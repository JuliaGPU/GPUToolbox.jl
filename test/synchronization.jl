# an operation that the test completes explicitly, polled with `isdone` and waited for with
# `blocking_wait`, which blocks GC-safely like a driver call would
mutable struct Operation
    @atomic done::Bool
    @atomic waited::Bool    # whether `blocking_wait` has been called
end
Operation() = Operation(false, false)
complete!(op::Operation) = (@atomic op.done = true; op)
complete_after!(op::Operation, secs) = (Timer(_ -> complete!(op), secs); op)
isdone(op::Operation) = @atomic op.done
function blocking_wait(op::Operation)
    @atomic op.waited = true
    while !isdone(op)
        @gcsafe_ccall uv_sleep(1::Cuint)::Cvoid
    end
    return :waited
end

function short_waits()
    for _ in 1:10
        cooperative_wait(blocking_wait, complete_after!(Operation(), 0.001); isdone)
    end
end

# wait for a worker to be waiting on `op`. the task that submitted it then waits too, as long
# as it runs on the current thread (i.e., it was created with `@async`).
waiting(op) = timedwait(() -> @atomic(op.waited), 30) === :ok

# signal a completion after a delay, from a thread that is not managed by Julia (like a
# driver's callback thread)
struct DelayedSignal
    payload::Ptr{Cvoid}
    ms::Cuint
end
function delayed_signal(arg::Ptr{DelayedSignal})
    signal = unsafe_load(arg)
    Libc.free(arg)
    @gcsafe_ccall uv_sleep(signal.ms::Cuint)::Cvoid
    GPUToolbox.signal_completion(signal.payload)
    return
end
function signal_later(payload, ms)
    arg = convert(Ptr{DelayedSignal}, Libc.malloc(sizeof(DelayedSignal)))
    unsafe_store!(arg, DelayedSignal(payload, ms))
    tid = Ref{NTuple{32, UInt8}}(ntuple(_ -> 0x0, 32))
    cb = @cfunction(delayed_signal, Cvoid, (Ptr{DelayedSignal},))
    err = ccall(:uv_thread_create, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), tid, cb, arg)
    @test err == 0
    ccall(:uv_thread_detach, Cint, (Ptr{Cvoid},), tid)
    return
end

@testset "cooperative_wait" begin
    # completed operations are detected by polling
    @test cooperative_wait(blocking_wait, complete!(Operation()); isdone) === nothing

    # without allocating (other than to shield from task cancellation, on Julia 1.14+)
    if !isdefined(Base, :CANCEL_TOKEN)
        op = complete!(Operation())
        fast_wait(op) = cooperative_wait(blocking_wait, op; isdone)
        fast_wait(op)
        @test @allocated(fast_wait(op)) == 0
    end

    # other tasks on this thread keep running while waiting on a worker
    for polled in (false, true)
        op = complete_after!(Operation(), 30)   # in case the thread is blocked
        t = @async cooperative_wait(blocking_wait, op; isdone=polled ? isdone : nothing,
                                    spin=false)
        try
            @test waiting(op)
            for _ in 1:100
                yield()
            end
            @test !isdone(op)
        finally
            complete!(op)
        end
        @test fetch(t) == Some(:waited)
    end

    # errors are rethrown
    @test_throws ErrorException("oops") cooperative_wait(_ -> error("oops"), nothing)
    @test_throws InterruptException cooperative_wait(_ -> throw(InterruptException()), nothing)
    @test_throws ErrorException("oops") cooperative_wait(blocking_wait, Operation();
                                                         isdone=_ -> error("oops"))

    # functions defined after the workers were started can be used
    f = @eval _ -> :new_function
    @test cooperative_wait(f, nothing) == Some(:new_function)

    # a long wait does not delay other ones, even with more waits than worker threads
    for polled in (false, true)
        long = Operation()
        waiter = Threads.@spawn cooperative_wait(blocking_wait, long;
                                                 isdone=polled ? isdone : nothing)
        shorts = Task[]
        try
            for _ in 1:8
                push!(shorts, Threads.@spawn short_waits())
            end
            @test timedwait(() -> all(istaskdone, shorts), 30) === :ok
            @test !istaskdone(waiter)
        finally
            complete!(long)
        end
        foreach(wait, shorts)
        @test fetch(waiter) == Some(:waited)
    end

    # interrupted waits keep waiting until the operation completes, unless cancellable
    for polled in (false, true), cancellable in (false, true)
        op = Operation()
        t = @async cooperative_wait(blocking_wait, op; cancellable, spin=false,
                                    isdone=polled ? isdone : nothing)
        try
            @test waiting(op)
            schedule(t, InterruptException(); error=true)
            if cancellable
                @test timedwait(() -> istaskdone(t), 30) === :ok
                @test !isdone(op)
            else
                for _ in 1:100
                    yield()
                end
                @test !istaskdone(t)
            end
        finally
            complete!(op)
        end
        @test_throws TaskFailedException wait(t)
        @test t.exception isa InterruptException
    end

    # the same goes for interrupts while polling
    for cancellable in (false, true)
        op = Operation()
        interrupted = Ref(false)
        function interrupting_isdone(op)
            interrupted[] || (interrupted[] = true; throw(InterruptException()))
            return isdone(op)
        end
        t = @async cooperative_wait(blocking_wait, op; cancellable,
                                    isdone=interrupting_isdone)
        try
            if cancellable
                @test timedwait(() -> istaskdone(t), 30) === :ok
                @test !isdone(op)
            else
                @test waiting(op)
                @test !istaskdone(t)
            end
        finally
            complete!(op)
        end
        @test_throws TaskFailedException wait(t)
        @test t.exception isa InterruptException
    end

    # the same goes for task cancellation
    if isdefined(Base, :CancellationTokenSource)
        src = Base.CancellationTokenSource()
        op = Operation()
        t = Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.CancellationToken(src)) do
            @async cooperative_wait(blocking_wait, op)
        end
        try
            @test waiting(op)
            Base.cancel!(src)
            for _ in 1:100
                yield()
            end
            @test !istaskdone(t)
        finally
            complete!(op)
        end
        @test_throws TaskFailedException wait(t)

        # also when polling finds the operation to have completed
        src = Base.CancellationTokenSource()
        Base.cancel!(src)
        op = complete!(Operation())
        t = Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.CancellationToken(src)) do
            @async cooperative_wait(blocking_wait, op; isdone)
        end
        @test_throws TaskFailedException wait(t)
        @test t.exception isa Base.CancellationRequest
    end

    # completion notifications from the driver, instead of a worker
    let
        # the callback may be invoked before registration returns
        subscribed = Ref(0)
        immediate = function (op, payload)
            subscribed[] += 1
            GPUToolbox.signal_completion(payload)
        end
        @test cooperative_wait(blocking_wait, Operation(); subscribe=immediate) === nothing
        @test subscribed[] == 1

        # or later, from another thread, while other tasks on this thread keep running
        ticks = Ref(0)
        ticker = @async while ticks[] >= 0
            ticks[] += 1
            sleep(0.001)
        end
        op = Operation()
        @test cooperative_wait(blocking_wait, op; subscribe=(_, p) -> signal_later(p, 100),
                               isdone, spin=false) === nothing
        @test !@atomic(op.waited)
        @test ticks[] > 0
        ticks[] = -1
        wait(ticker)

        # errors while registering are rethrown
        failing = (_, _) -> error("oops")
        @test_throws ErrorException("oops") cooperative_wait(blocking_wait, Operation();
                                                             subscribe=failing)

        # interrupted waits keep waiting until notified, unless cancellable, in which case
        # a later notification is harmless
        for cancellable in (false, true)
            notified = Base.Event()
            payload = Ref{Ptr{Cvoid}}(C_NULL)
            subscribe = (_, p) -> (payload[] = p; notify(notified))
            t = @async cooperative_wait(blocking_wait, Operation(); subscribe, cancellable)
            wait(notified)
            yield()
            schedule(t, InterruptException(); error=true)
            if cancellable
                @test timedwait(() -> istaskdone(t), 30) === :ok
                GC.gc()
                signal_later(payload[], 10)
                sleep(0.1)
            else
                for _ in 1:100
                    yield()
                end
                @test !istaskdone(t)
                signal_later(payload[], 10)
            end
            @test_throws TaskFailedException wait(t)
            @test t.exception isa InterruptException
        end

        # the same goes for task cancellation
        if isdefined(Base, :CancellationTokenSource)
            for cancellable in (false, true)
                src = Base.CancellationTokenSource()
                notified = Base.Event()
                payload = Ref{Ptr{Cvoid}}(C_NULL)
                subscribe = (_, p) -> (payload[] = p; notify(notified))
                token = Base.CancellationToken(src)
                t = Base.ScopedValues.with(Base.CANCEL_TOKEN => token) do
                    @async cooperative_wait(blocking_wait, Operation(); subscribe,
                                            cancellable)
                end
                wait(notified)
                Base.cancel!(src)
                if cancellable
                    @test timedwait(() -> istaskdone(t), 30) === :ok
                    GC.gc()
                    signal_later(payload[], 10)
                    sleep(0.1)
                else
                    for _ in 1:100
                        yield()
                    end
                    @test !istaskdone(t)
                    signal_later(payload[], 10)
                end
                @test_throws TaskFailedException wait(t)
            end
        end
    end

    # finalizers cannot switch tasks, so they wait on the calling thread
    ret = Ref{Any}()
    @noinline function finalized_object()
        obj = Ref(0)
        finalizer(obj) do _
            ret[] = cooperative_wait(blocking_wait, complete!(Operation()))
        end
        return
    end
    finalized_object()
    GC.gc(); GC.gc()
    @test isassigned(ret) && ret[] == Some(:waited)
end
