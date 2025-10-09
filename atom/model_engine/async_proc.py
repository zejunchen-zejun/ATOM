import logging
import multiprocessing
import pickle
import queue
import signal
import threading
import weakref
from contextlib import ExitStack
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from threading import Thread
from typing import List

import zmq
import zmq.asyncio

from atom.utils import (
    get_mp_context,
    get_open_zmq_ipc_path,
    init_exit_handler,
    make_zmq_socket,
    resolve_obj_by_qualname,
    shutdown_all_processes,
)

logger = logging.getLogger("atom")


class AsyncIOProc:

    def __init__(
        self,
        label: str,
        label_shm: str,
        io_addrs: tuple[str, str],
        sync_event: Event,
        # runner class and its args
        runner_qualname: str,  # atom.model_engine.model_runner.ModelRunner
        *args,
        **kwargs,
    ):
        self.label = f"AsyncIOProc({label})"
        self.sync_event = sync_event
        self.io_addrs = io_addrs
        self.io_queues = queue.Queue(), queue.Queue()
        self.io_threads = []
        self.shm = SharedMemory(name=f"atom_{label_shm}_shm")
        init_exit_handler(self)

        # make sure exit handler is set before runner is created
        runner_class = resolve_obj_by_qualname(runner_qualname)  # type: ignore
        self.runners = []
        self.runners.append(runner_class(*args, **kwargs))
        for addr, q, func in zip(
            self.io_addrs,
            self.io_queues,
            [self.recv_input_from_socket, self.send_output_to_socket],
        ):
            if addr is None:
                continue
            t = threading.Thread(target=func, args=(addr, q), daemon=True)
            t.start()
            self.io_threads.append(t)
        self.busy_loop()

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        logger.debug(f"{self.label}: Shutting down runner...")
        for el in self.runners:
            el.exit()
        self.shm.close()
        for t in self.io_threads:
            t.join(timeout=0.5)

    def recv_input_from_socket(self, addr: str, input_queue: queue.Queue):
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
            socket.send(b"")
            poller.register(socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input socket connected")

            while self.still_running:
                for socket, _ in poller.poll(timeout=1000):
                    serialized_obj = socket.recv(copy=False)
                    input = pickle.loads(serialized_obj)
                    input_queue.put_nowait(input)

    def send_output_to_socket(self, addr: str, output_queue: queue.Queue):
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            while True:
                result = output_queue.get()
                serialized_obj = pickle.dumps(result)
                socket.send(serialized_obj)

    def busy_loop(self):
        while True:
            func_name, args = self.get_func()
            for runner in self.runners:
                func = getattr(runner, func_name, None)
                if func is None:
                    continue
                out = func(*args)
                if out is not None:
                    self.io_queues[1].put_nowait(out)
            if func_name == "exit":
                break
        logger.debug(f"{self.label}: exit busy_loop...")

    def get_func(self):
        self.sync_event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.sync_event.clear()
        return method_name, args


class AsyncIOProcManager:

    def __init__(self, finalizer, proc_num: int, runner: str, *args):
        self.parent_finalizer = finalizer
        io_addrs = [get_open_zmq_ipc_path(), get_open_zmq_ipc_path()]
        self.procs = []
        self.sync_events = []
        ctx = get_mp_context()
        self.mp_ctx = ctx
        self.runner_label = runner.split(".")[-1]
        self.label = f"AsyncIOProcManager({self.runner_label})"
        self.create_shared_memory()
        self.still_running = True
        init_exit_handler(self)
        for i in range(proc_num):
            label = f"ModelRunner{i}/{proc_num}"
            addrs = (
                io_addrs if i == 0 else [io_addrs[0], None]
            )  # only get output from rank 0
            sync_event = ctx.Event()
            process = ctx.Process(
                target=AsyncIOProc,
                name=label,
                args=(label, self.runner_label, addrs, sync_event, runner, i, *args),
            )
            process.start()
            self.procs.append(process)
            self.sync_events.append(sync_event)

        self.zmq_ctx = zmq.Context(io_threads=2)
        self.outputs_queue = queue.Queue()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            name=f"{self.label}_output_thread",
            args=(io_addrs[1],),
            daemon=True,
        )
        self.output_thread.start()
        self.monitor_procs()

    def create_shared_memory(self):
        name = f"atom_{self.runner_label}_shm"
        try:
            self.shm = SharedMemory(name=name, create=True, size=2**20)
        except FileExistsError:
            existing_shm = SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
            self.shm = SharedMemory(name=name, create=True, size=2**20)

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        # 1. kill all runners
        logger.info(f"{self.label}: shutdown all runners...")
        shutdown_all_processes(self.procs, allowed_seconds=1)
        self.shm.unlink()
        self.procs = []
        self.output_thread.join()
        logger.info(f"{self.label}: All runners are shutdown.")
        # 2. put a None to unblock call_func
        self.outputs_queue.put_nowait(SystemExit())
        # 3. call parent finalizer
        self.parent_finalizer()

    def process_output_sockets(self, output_address: str):
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue

                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)  # type: ignore
                self.outputs_queue.put_nowait(obj)
        finally:
            # Close sockets.
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: output thread exit")

    def call_func(self, func_name: str, *args, wait_out: bool = False):
        logger.debug(f"{self.label}: call_func {func_name} {args}")
        n = pickle.dumps((func_name, *args))
        self.shm.buf[0:4] = len(n).to_bytes(4, "little")
        self.shm.buf[4 : len(n) + 4] = n
        for ev in self.sync_events:
            ev.set()
        if wait_out:
            ret = self.outputs_queue.get()
            if isinstance(ret, SystemExit):
                raise ret
            return ret

    def monitor_procs(self):
        self_ref = weakref.ref(self)
        procs = self.procs
        self.keep_monitoring = True

        def monitor_engine_cores():
            sentinels = [proc.sentinel for proc in procs]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or not _self.keep_monitoring:
                return
            proc_name = next(proc.name for proc in procs if proc.sentinel == died[0])
            logger.error(
                f"{self.label}: [{proc_name}] proc died unexpectedly, shutting down.",
            )
            _self.exit()

        Thread(
            target=monitor_engine_cores, daemon=True, name=f"{self.runner_label}Monitor"
        ).start()
