import logging
import multiprocessing
import pickle
import queue
import signal
import sys
import weakref
from threading import Thread
from typing import List

import zmq
import zmq.asyncio

from atom.config import Config
from atom.model_engine.engine_core import EngineCore, EngineCoreRequestType
from atom.model_engine.sequence import Sequence
from atom.utils import (
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
    shutdown_all_processes,
)

logger = logging.getLogger("atom")


class CoreManager:
    def __init__(self, config: Config):
        self.label = "Engine Core Mgr"
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if config.asyncio_mode else sync_ctx
        self.outputs_queue = queue.Queue[List[Sequence]]()
        engine_core_process, addresses = launch_engine_core(config)
        self.engine_core_process = engine_core_process

        input_address = addresses["input_address"]
        output_address = addresses["output_address"]
        self.input_socket = make_zmq_socket(
            self.ctx, input_address, zmq.ROUTER, bind=True
        )
        identity, _ = self.input_socket.recv_multipart()
        self.engine_core_identity = identity
        self.output_socket = make_zmq_socket(self.ctx, output_address, zmq.PULL)

        shutdown_path = get_open_zmq_inproc_path()
        self.shutdown_path = shutdown_path

        def process_outputs_socket():
            assert isinstance(self.output_socket, zmq.Socket)
            shutdown_socket = self.ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(self.output_socket, zmq.POLLIN)
                logger.debug(f"{self.label}: output thread started")
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        logger.debug(
                            f"{self.label}: output thread receive shutdown signal"
                        )
                        break

                    obj = self.output_socket.recv(copy=False)
                    request_type, seqs = pickle.loads(obj)
                    if request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(
                            f"{self.label}: output thread receive SHUTDOWN request"
                        )
                        self._shutdown_engine_core()
                        self.outputs_queue.put_nowait(
                            SystemExit(f"{self.label}: shutdown")
                        )
                        break
                    elif request_type == EngineCoreRequestType.ADD:
                        # logger.info(f"Engine core output sequence id: {seq.id}")
                        self.outputs_queue.put_nowait(seqs)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                self.output_socket.close(linger=0)

        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()
        self._finalizer = weakref.finalize(self, self.close)

    def close(self):
        self._shutdown_engine_core()
        self.input_socket.close()
        if self.shutdown_path and self.output_queue_thread:
            with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                shutdown_sender.connect(self.shutdown_path)
                # Send shutdown signal.
                shutdown_sender.send(b"")
        self.output_queue_thread.join()

    def add_request(self, seq: Sequence):
        logger.debug(f"{self.label}: Add request, sequence id: {seq.id}")
        self.input_socket.send_multipart(
            [self.engine_core_identity, pickle.dumps((EngineCoreRequestType.ADD, seq))],
            copy=False,
        )

    def _shutdown_engine_core(self):
        if self.engine_core_process is not None and self.engine_core_process.is_alive():
            self.input_socket.send_multipart(
                [
                    self.engine_core_identity,
                    pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)),
                ],
                copy=False,
            )

    def get_output(self) -> List[Sequence]:
        seqs = self.outputs_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    def is_rest(self):
        return not self.outputs_queue.empty()

    def is_alive(self):
        return self.engine_core_process is not None


def launch_engine_core(config: Config):
    input_address = get_open_zmq_ipc_path()
    output_address = get_open_zmq_ipc_path()
    import torch
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method("spawn", force=False)
    process = multiprocessing.Process(
        target=EngineCore.run_engine,
        name=f"EngineCore",
        kwargs={
            "config": config,
            "input_address": input_address,
            "output_address": output_address,
        },
    )
    process.start()
    return process, {"input_address": input_address, "output_address": output_address}
