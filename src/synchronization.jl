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


## slow path: completion notifications

# some drivers can notify us when an operation completes, by calling back from a thread of
# their own. that avoids a worker thread, which matters for devices that execute on the
# host's CPU cores: waking a worker while the operation starts delays it considerably.
#
# the callback directly wakes the waiting task, like a worker does, instead of going through
# the event loop, which would add a thread hop (and depends on the thread running the event
# loop to be available). the calling thread is adopted by Julia, which is safe as long as
# driver calls that may be waiting on the callback are GC-safe.

# a spin lock-based condition, so that signalling from a driver thread never blocks it in
# Julia's scheduler (as a `ReentrantLock` could)
mutable struct Completion
    const cond::Base.ThreadSynchronizer
    @atomic state::Int      # one of the constants below

    Completion() = new(Base.ThreadSynchronizer(), PENDING)
end
const PENDING = 0
const SIGNALLED = 1
const RELEASED = 2          # the callback does not access the completion anymore

# completions whose waiter stopped waiting (e.g., because it was cancelled), kept alive
# until the driver has signalled them
const abandoned_completions = Completion[]
const abandoned_completions_lock = ReentrantLock()

function sweep_abandoned_completions()
    filter!(c -> (@atomic :acquire c.state) != RELEASED, abandoned_completions)
end

function abandon_completion(c::Completion)
    @lock abandoned_completions_lock begin
        sweep_abandoned_completions()
        push!(abandoned_completions, c)
    end
    return
end

"""
    GPUToolbox.signal_completion(payload::Ptr{Cvoid})

Signal that the operation a `subscribe` function (see [`cooperative_wait`](@ref)) registered
a notification for has completed, waking up the waiting task. This can be called from a
thread that is not managed by Julia (e.g., one owned by the driver), and does not throw.
"""
function signal_completion(payload::Ptr{Cvoid})
    c = unsafe_pointer_to_objref(payload)::Completion
    # releasing the lock runs pending finalizers. that should not happen here, as the driver
    # may be holding locks a finalizer needs, so defer them to a thread managed by Julia.
    # (debug builds of Julia do run them when finalizers are re-enabled below.)
    ccall(:jl_gc_disable_finalizers_internal, Cvoid, ())
    lock(c.cond)
    @atomic c.state = SIGNALLED
    notify(c.cond)
    unlock(c.cond)
    ccall(:jl_gc_enable_finalizers_internal, Cvoid, ())
    # this is the last access, after which the completion may be freed
    @atomic :release c.state = RELEASED
    return
end

# register a completion notification, or return the one registered already when resuming
# an interrupted wait
function subscribe!(subscribe::F, obj, state) where {F}
    registered = state.completion
    registered === nothing || return registered

    # reclaim completions that were abandoned by earlier waits
    if !isempty(abandoned_completions)
        @lock abandoned_completions_lock sweep_abandoned_completions()
    end

    # registering the notification and keeping track of it must not be interrupted, or a
    # retry would register it again, or we would lose track of it. registration may invoke
    # the callback immediately, which is fine.
    c = Completion()
    uninterruptible() do
        GC.@preserve c subscribe(obj, pointer_from_objref(c))
        state.completion = c
    end
    return c
end

# returns the completion once it has been signalled
function wait_completion(subscribe::F, obj, state) where {F}
    try
        c = subscribe!(subscribe, obj, state)
        @lock c.cond while (@atomic c.state) == PENDING
            Base.wait(c.cond)
        end
        return c
    catch
        # when unwinding (e.g., because of an interrupt), the callback may still use the
        # completion, so keep it alive until then. an interrupted wait that is resumed
        # waits for the same completion, which is fine.
        registered = state.completion
        if registered !== nothing && (@atomic :acquire registered.state) != RELEASED
            uninterruptible(() -> abandon_completion(registered))
        end
        rethrow()
    end
end


## entry point

mutable struct WaitState
    # the request submitted to a worker, if any
    request::Union{Nothing,WaitRequest}
    # the completion notification registered, if any
    completion::Union{Nothing,Completion}
    # the first interrupt, to be thrown once the wait has completed
    interrupt::Union{Nothing,InterruptException}
end

# returns the completed request, or `nothing` if polling found `obj` to have completed
function wait_cooperatively(wait::W, obj, isdone::D, state) where {W,D}
    # when resuming an interrupted wait, keep waiting for the submitted request
    request = state.request
    if request === nothing
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
uninterruptible(f::F) where {F} = shielded(() -> disable_sigint(f))

"""
    cooperative_wait(wait, obj; subscribe=nothing, isdone=nothing, spin=true,
                     cancellable=false)

Wait for `obj` to complete without blocking the calling thread, so that other tasks can run
on it in the meantime.

`wait(obj)` should perform a blocking wait for `obj`. By default, it is executed on a
separate thread, so it should block in a GC-safe manner (e.g., using
[`@gcsafe_ccall`](@ref)) and set up any thread-local state it relies on (e.g., the active
context).

Alternatively, if the driver can notify completion by calling back, pass a function
`subscribe(obj, payload::Ptr{Cvoid})` that registers such a callback, which in turn calls
[`GPUToolbox.signal_completion(payload)`](@ref GPUToolbox.signal_completion). This avoids
involving a worker thread, which matters for devices that execute on the host's CPU cores,
where waking a worker delays the operation. `wait` is then only used where the calling
thread cannot switch tasks (see below). The callback:

- has to be called exactly once, also when the operation fails, and may be called before
  `subscribe` returns;
- must not be registered if `subscribe` throws;
- may be called from any thread, but must not let exceptions escape into the driver.

`signal_completion` does not run finalizers on the calling thread, except on debug builds
of Julia, so with those, drivers that hold locks while invoking callbacks may deadlock if a
finalizer calls into the driver. Don't use `subscribe` with such drivers there.

Registering the callback, and any other driver call that may wait for callbacks to finish,
should be GC-safe (e.g., using [`@gcsafe_ccall`](@ref)): a thread calling back into Julia
may have to wait for the garbage collector, which waits for all threads executing Julia
code. For example, with OpenCL:

```julia
function notify_completion(event::cl_event, status::Cint, payload::Ptr{Cvoid})
    GPUToolbox.signal_completion(payload)
    return
end
subscribe(event, payload) =
    clSetEventCallback(event, CL_COMPLETE,
                       @cfunction(notify_completion, Cvoid, (cl_event, Cint, Ptr{Cvoid})),
                       payload)
```

If the wait can be interrupted (see `cancellable`), the callback may be called after
`cooperative_wait` has returned, so it should not use anything else the caller may release
by then.

`isdone(obj)`, if given, should return whether `obj` has completed without blocking. It is
used to detect short operations without involving another thread, and to poll `obj` while
all worker threads are busy. Objects that cannot be polled wait for a thread to become
available instead.

With `spin=false`, `obj` is not polled before waiting as described above. For devices that
execute on the host's CPU cores, that is recommended: polling competes with the operation
for those cores.

Returns `Some(wait(obj))`, or `nothing` if `wait` was not called because polling or a
notification found `obj` to have completed. In that case, calling `wait(obj)` should not
block for long (it may have to wait for the notification callback to return), and may
still be needed, e.g., to synchronize memory or to check for errors. Errors thrown by
`wait`, `subscribe` or `isdone` are rethrown.

If the wait is interrupted (i.e., an `InterruptException` is thrown) or the task is
cancelled, the default is to keep waiting, and only throw once `obj` has completed: the
operation may be using memory that the caller would release when unwinding. Note that
`wait(obj)` may not have been called by then. Waits with `cancellable=true` throw
immediately, while the operation may still be executing.

In finalizers, where it is not possible to switch tasks, and while generating output (e.g.,
during precompilation), `wait(obj)` is called on the calling thread instead.
"""
function cooperative_wait(wait::W, obj; subscribe::S=nothing, isdone::D=nothing,
                          spin::Bool=true, cancellable::Bool=false) where {W,S,D}
    if GC.in_finalizer() || generating_output()
        return Some(wait(obj))
    end

    # fast path: poll the object, without allocating the state needed for the slow path.
    # like the slow path (see `wait_uninterrupted`), polling is shielded from cancellation,
    # and an interrupt is only thrown once the operation has completed, unless cancellable.
    interrupt = nothing
    if isdone !== nothing && spin
        done = try
            if cancellable
                spin_until(isdone, obj)
            else
                shielded(() -> spin_until(isdone, obj))
            end
        catch err
            (cancellable || !(err isa InterruptException)) && rethrow()
            interrupt = err
            false
        end
        if done
            cancellable || check_cancelled()
            return nothing
        end
    end

    # slow path: wait for a completion notification, or hand the wait to a worker thread
    state = WaitState(nothing, nothing, interrupt)
    slow_wait = if subscribe === nothing
        () -> wait_cooperatively(wait, obj, isdone, state)
    else
        () -> wait_completion(subscribe, obj, state)
    end
    ret = cancellable ? slow_wait() : wait_uninterrupted(slow_wait, state)
    ret isa WaitRequest || return nothing
    ret.failed && throw(ret.result)
    return Some(ret.result)
end

# an interrupt can be delivered wherever we yield or hit a safepoint. keep waiting, resuming
# from where we were, and throw the interrupt once done. cancellation is deferred by running
# in a shielded scope.
function wait_uninterrupted(slow_wait::F, state) where {F}
    ret = shielded() do
        while true
            try
                return slow_wait()
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
