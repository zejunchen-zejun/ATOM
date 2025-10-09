import contextlib
import ipaddress
import logging
import multiprocessing
import os
import signal
import socket
import tempfile
import time
from functools import lru_cache
from multiprocessing.context import ForkContext, SpawnContext
from multiprocessing.process import BaseProcess
from typing import Any, Iterator, Optional, Sequence, Union
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import psutil
import torch
import zmq
import zmq.asyncio

logger = logging.getLogger("atom")


def get_mp_context() -> Union[ForkContext, SpawnContext]:
    """Get a multiprocessing context with 'spawn' start method."""
    return multiprocessing.get_context("spawn")


def shutdown_all_processes(procs: list[BaseProcess], allowed_seconds: int = 2):
    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # Allow 5 seconds for remaining procs to terminate.
    deadline = time.monotonic() + allowed_seconds
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def split_host_port(host_port: str) -> tuple[str, int]:
    # ipv6
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        host = host[1:]
        port = port.split(":")[1]
        return host, int(port)
    else:
        host, port = host_port.split(":")
        return host, int(port)


def join_host_port(host: str, port: int) -> str:
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    return _get_open_port()


def get_open_ports_list(count: int = 5) -> list[int]:
    """Get a list of open ports."""
    ports = set()
    while len(ports) < count:
        ports.add(get_open_port())
    return list(ports)


def _get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


@lru_cache()
def get_zmq_base_path() -> str:
    return tempfile.gettempdir()


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = get_zmq_base_path()
    return f"ipc://{base_rpc_path}/{uuid4()}"


def get_engine_client_zmq_addr(local_only: bool, host: str, port: int = 0) -> str:
    """Assign a new ZMQ socket address.

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP
    address (port == 0 means assign an available port)."""

    return (
        get_open_zmq_ipc_path()
        if local_only
        else (get_tcp_uri(host, port or get_open_port()))
    )


def close_sockets(sockets: Sequence[Union[zmq.Socket, zmq.asyncio.Socket]]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


def make_zmq_path(scheme: str, host: str, port: Optional[int] = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: Union[zmq.asyncio.Context, zmq.Context],  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    identity: Optional[bytes] = None,
    linger: Optional[int] = None,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


def init_exit_handler(self: Any):
    import weakref

    self.still_running = True
    self._finalizer = weakref.finalize(self, self.exit)

    def signal_handler(signum, frame):
        sig_str = signal.Signals(signum).name
        msg = f"{self.label}: received signal {signum} ({sig_str}), exiting..."
        logger.info(msg)
        self._finalizer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    linger: int = 0,
    identity: Optional[bytes] = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type, bind=bind, identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)


class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(
        self,
        *size: Union[int, torch.SymInt],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = True,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: Optional[int] = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: Optional[int] = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


import copy
import dataclasses
import importlib
import time
from contextlib import contextmanager
from typing import Any, Union

import torch
from packaging import version
from packaging.version import Version

from atom.config import CompilationConfig, CompilationLevel, Config

context_manager = None
torch_compile_start_time: float = 0.0


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.ops._C.weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]],
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")


@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    # including the splitting ops
    num_piecewise_graphs_seen: int = 0
    # not including the splitting ops
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    num_gpu_runner_capture_triggers: int = 0
    # Number of CUDAGraphs captured
    num_cudagraph_captured: int = 0
    # InductorAdapter.compile calls
    num_inductor_compiles: int = 0
    # EagerAdapter.compile calls
    num_eager_compiles: int = 0
    # The number of time vLLM's compiler cache entry was updated
    num_cache_entries_updated: int = 0
    # The number of standalone_compile compiled artifacts saved
    num_compiled_artifacts_saved: int = 0
    # Number of times a model was loaded with CompilationLevel.DYNAMO_AS_IS
    dynamo_as_is_count: int = 0

    def clone(self) -> "CompilationCounter":
        return copy.deepcopy(self)

    @contextmanager
    def expect(self, **kwargs):
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, (
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


compilation_counter = CompilationCounter()


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def start_monitoring_torch_compile(vllm_config: Config):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = vllm_config.compilation_config
    if (
        compilation_config.level == CompilationLevel.PIECEWISE
        and compilation_config.debug_dump_path
    ):
        import depyf

        path = os.path.join(compilation_config.debug_dump_path, "rank_0")
        # f"rank_{vllm_config.parallel_config.rank}")
        global context_manager
        context_manager = depyf.prepare_debug(path)
        context_manager.__enter__()


def end_monitoring_torch_compile(vllm_config: Config):
    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE:
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


def getLogger():
    global logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if int(os.environ.get("AITER_LOG_MORE", 0)):
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)03d - %(processName)s:%(process)d - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                fmt="[%(name)s] %(message)s",
            )
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(console_handler)
        if hasattr(torch._dynamo.config, "ignore_logger_methods"):
            torch._dynamo.config.ignore_logger_methods = (
                logging.Logger.info,
                logging.Logger.warning,
                logging.Logger.debug,
                logger.warning,
                logger.info,
                logger.debug,
            )

    return logger


logger = getLogger()
