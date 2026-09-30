export cooperative_wait

# Cooperative waiting: waiting for a GPU should not block the calling thread, so that other
# tasks can run in the meantime. To keep short waits fast, the object is first polled, and
# only then is the blocking wait handed to a worker thread while the calling task yields.


## fast path: polling

# handing the wait to a worker costs a couple of µs, or more when the worker thread has gone
# to sleep, so first poll the object: busy-waiting at first, then yielding to other tasks.
#
# the budget is a number of iterations, so how long it lasts depends on the machine and on
# the load: ~60 µs on an idle system, but longer when other tasks run, or the OS deschedules
# us, while yielding. that is intentional: a time-based budget performed worse on a loaded
# system, where waking up a worker that has gone to sleep gets expensive.
const SPIN_BUSY_ITERATIONS = 32
const SPIN_ITERATIONS = 256

function spin_until(isdone::F, obj) where {F}
    isdone(obj) && return true
    for i in 1:SPIN_ITERATIONS
        if i <= SPIN_BUSY_ITERATIONS
            ccall(:jl_cpu_pause, Cvoid, ())
            GC.safepoint()
        else
            yield()
        end
        isdone(obj) && return true
    end
    return false
end


## slow path: worker threads

# blocking waits are performed on a pool of foreign threads (which Julia adopts). each
# worker handles a single request at a time, so a long wait does not delay other ones. idle
# workers are reused last-in-first-out: the most recently used worker is likely still
# spinning in the scheduler, and does not need to be woken up by the OS.
#
# the number of workers is bounded, because drivers often spin while waiting, occupying a
# CPU core per worker. when all workers are busy, objects that can be polled are polled on
# the calling thread instead, until a worker becomes available. other objects have to wait
# for one.

mutable struct WaitRequest
    const wait::Any
    const obj::Any
    const done::Base.Event
    result::Any
    failed::Bool

    WaitRequest(wait, obj) = new(wait, obj, Base.Event(), nothing, false)
end

mutable struct WaitWorker
    const work::Base.Event      # autoreset
    request::Union{Nothing,WaitRequest}

    WaitWorker() = new(Base.Event(true), nothing)
end

const MAX_WAIT_WORKERS = 4
const wait_workers = WaitWorker[]       # all workers, keeping them rooted
const idle_wait_workers = WaitWorker[]
const wait_worker_lock = ReentrantLock()
const wait_worker_available = Threads.Condition(wait_worker_lock)

# how long to poll before checking again whether a worker has become available
const POLL_RETRY_NS = 100_000

function wait_worker_loop(data::Ptr{Cvoid})
    worker = unsafe_pointer_to_objref(data)::WaitWorker
    while true
        Base.wait(worker.work)
        request = worker.request::WaitRequest
        worker.request = nothing

        try
            # the worker was started in an older world
            request.result = Base.invokelatest(request.wait, request.obj)
        catch err
            request.result = err
            request.failed = true
        end

        # become available before waking the caller, so that it finds this worker when it
        # immediately waits again
        release_wait_worker(worker)
        notify(request.done)
    end
end

function release_wait_worker(worker::WaitWorker)
    @lock wait_worker_lock begin
        push!(idle_wait_workers, worker)
        notify(wait_worker_available)
    end
end

# needs to be called with `wait_worker_lock` held
function create_wait_worker()
    worker = WaitWorker()
    push!(wait_workers, worker)
    started = false
    try
        # we don't know what the size of uv_thread_t is, so reserve enough space
        tid = Ref{NTuple{32, UInt8}}(ntuple(i -> 0, 32))

        cb = @cfunction(wait_worker_loop, Cvoid, (Ptr{Cvoid},))
        err = ccall(:uv_thread_create, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                    tid, cb, pointer_from_objref(worker))
        err == 0 || Base.uv_error("uv_thread_create", err)
        started = true
        ccall(:uv_thread_detach, Cint, (Ptr{Cvoid},), tid)
    finally
        started || pop!(wait_workers)
    end
    return worker
end

# hand a request to a worker, returning whether one was available. if all workers are busy,
# either wait for one to become available, or return `false`.
function submit!(state, request::WaitRequest, wait::Bool)
    @lock wait_worker_lock begin
        while true
            # handing over the request must not be interrupted halfway, or the worker would
            # be lost
            submitted = uninterruptible() do
                worker = if !isempty(idle_wait_workers)
                    pop!(idle_wait_workers)
                elseif length(wait_workers) < MAX_WAIT_WORKERS
                    create_wait_worker()
                else
                    return false
                end
                worker.request = request
                try
                    notify(worker.work)
                catch
                    # only acquiring the event's lock can be interrupted (e.g., by an
                    # exception scheduled onto this task), so nothing was published yet
                    worker.request = nothing
                    push!(idle_wait_workers, worker)
                    notify(wait_worker_available)
                    rethrow()
                end
                state.request = request
                return true
            end
            submitted && return true
            wait || return false
            Base.wait(wait_worker_available)
        end
    end
end


## entry point

mutable struct WaitState
    # the request submitted to a worker, if any
    request::Union{Nothing,WaitRequest}
    # the first interrupt, to be thrown once the wait has completed
    interrupt::Union{Nothing,InterruptException}
end

# returns the completed request, or `nothing` if polling found `obj` to have completed
function wait_cooperatively(wait, obj, isdone, spin, state)
    # when resuming an interrupted wait, keep waiting for the submitted request
    request = state.request
    if request === nothing
        if isdone !== nothing && spin && spin_until(isdone, obj)
            return nothing
        end

        request = WaitRequest(wait, obj)
        while !submit!(state, request, isdone === nothing)
            # all workers are busy: poll the object, periodically checking for a worker
            t0 = time_ns()
            while time_ns() - t0 < POLL_RETRY_NS
                isdone(obj) && return nothing
                yield()
            end
        end
    end

    Base.wait(request.done)
    return request
end

# run `f` in a scope that task cancellation does not apply to, and check afterwards whether
# the task was cancelled in the meantime
@static if isdefined(Base, :CANCEL_TOKEN)
    shielded(f) = Base.ScopedValues.with(f, Base.CANCEL_TOKEN => nothing)
    check_cancelled() = Base.checkcancel(Base.CANCEL_TOKEN[])
else
    shielded(f) = f()
    check_cancelled() = nothing
end

# run `f` without it being interrupted or cancelled
uninterruptible(f) = shielded(() -> disable_sigint(f))

"""
    cooperative_wait(wait, obj; isdone=nothing, spin=true, cancellable=false)

Wait for `obj` to complete without blocking the calling thread, so that other tasks can run
on it in the meantime.

`wait(obj)` should perform a blocking wait for `obj`. It is executed on a separate thread,
so it should block in a GC-safe manner (e.g., using [`@gcsafe_ccall`](@ref)) and set up any
thread-local state it relies on (e.g., the active context).

`isdone(obj)`, if given, should return whether `obj` has completed without blocking. It is
used to detect short operations without involving another thread (unless `spin=false`),
and to poll `obj` while all of these threads are busy. Objects that cannot be polled wait
for a thread to become available instead.

Returns `Some(wait(obj))`, or `nothing` if `wait` was not called because polling found `obj`
to have completed. In that case, calling `wait(obj)` returns without blocking, which may
still be needed, e.g., to synchronize memory or to check for errors. Errors thrown by `wait`
or `isdone` are rethrown.

If the wait is interrupted (i.e., an `InterruptException` is thrown) or the task is
cancelled, the default is to keep waiting, and only throw once `obj` has completed: the
operation may be using memory that the caller would release when unwinding. Note that `wait(obj)` may not
have been called by then. Waits with `cancellable=true` throw immediately, while the
operation may still be executing.

In finalizers, where it is not possible to switch tasks, and while generating output (e.g.,
during precompilation), `wait(obj)` is called on the calling thread instead.
"""
function cooperative_wait(wait::W, obj; isdone::D=nothing, spin::Bool=true,
                          cancellable::Bool=false) where {W,D}
    if GC.in_finalizer() || generating_output()
        return Some(wait(obj))
    end

    state = WaitState(nothing, nothing)
    request = if cancellable
        wait_cooperatively(wait, obj, isdone, spin, state)
    else
        wait_uninterrupted(wait, obj, isdone, spin, state)
    end
    request === nothing && return nothing
    request.failed && throw(request.result)
    return Some(request.result)
end

# an interrupt can be delivered wherever we yield or hit a safepoint. keep waiting, resuming
# from where we were, and throw the interrupt once done. cancellation is deferred by running
# in a shielded scope.
function wait_uninterrupted(wait, obj, isdone, spin, state)
    ret = shielded() do
        while true
            try
                return wait_cooperatively(wait, obj, isdone, spin, state)
            catch err
                err isa InterruptException || rethrow()
                state.interrupt === nothing && (state.interrupt = err)
            end
        end
    end
    state.interrupt === nothing || throw(state.interrupt)
    check_cancelled()
    return ret
end
