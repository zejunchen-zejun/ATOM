# Kimi-K3 PD 1P1D on MI455X（Helios，KV 走 UALink）

目标：P 和 D 都用 TP8 + DCP8（每个实例占两台 tray），MLPerf Endpoints agentic c1024，按模拟时的 10 分钟窗口测。
镜像：`rocm/fw-bringup:gfx1250-atom-k3-20261116`（/app/ATOM c7a0003c）；1113（746624d4）也能用。问题记录在 `ISSUES.md`（内部运行记录，公开的 recipe 里不带）。

## 跑通 455 K3 1P1D 需要的代码改动

### 1. ATOM：三个 commit（两个功能、一个修复）

- `zejunchen-zejun/ATOM` 的分支 `zejun/k3-455-1p1d-ualink`，基于 1113 的 746624d4：b0fdb59d、9eddd1ca、`recipes/k3-455-pd/`（这份 recipe）、23cde1ae（修复）。
- 1116 镜像用的是同样三个 commit cherry-pick 到 c7a0003c 上的版本：8c66e0a9、9fc963d7、ec4a797c，`atom-k3pd-pd.bundle` 打包的就是这个。

| commit | 解决的问题 | 改动 |
|---|---|---|
| `b0fdb59d` fix(moriio): reach a producer whose TP group spans hosts | P 的 TP8 跨两台 tray（`--tensor-parallel-size-local 4`），但 P 的 engine core 上报的 `remote_host` 只有 node0 的地址。D 的 rank 4-7 会把 write request 和 done-notify 发到 P node0 的 `handshake_port + r`，那里没有进程在监听 | `kv_transfer_config` 新增 `remote_tp_hosts: {P 的 remote_host: [按 TP rank 顺序排列的 P host]}`。consumer 的每个 rank 用 `tp_rank // (remote_tp_size / len(hosts))` 找到对端 rank 所在的 host，write request、done-notify、read handshake、read notify 都按这个地址发。不在 map 里的 producer 行为不变。`moriio_connector.py`，加 1 个测试 |
| `9eddd1ca` feat(moriio): KV transfer over the UALink fabric (backend "fabric") | 用户要求 KV 走 UALink，不走 RDMA。原来的 moriio 只有 RDMA | `kv_transfer_config.backend`：`"rdma"`（默认，行为不变）或 `"fabric"`，后者用 mori 的 FABRIC backend，GPU 到 GPU 直接走 UALink（要求同一个 vPOD），不经过 NIC。FABRIC 只能 export fabric VMM 内存，所以 `ModelRunner` 在 moriio + fabric 时，paged KV pool 和 per-request state（KDA conv/SSM，每个 slot 每个 rank 56,171,520 B）都从 mori 的 fabric MemPool 分配（`make_fabric_mem_pool` + `torch.cuda.use_mem_pool`），pool 跟 runner 同生命周期。fabric import 会把每个注册 range 背后的整块 allocation 都映射一遍，所以每个 tensor 只注册成一个 range（RDMA 那种 2 GiB 切块会让 KV pool 被映射 N 次）。2 GiB 的 state-slot 上限只对 RDMA 生效。改动在 `moriio_connector.py`、`model_runner.py`，新增 `tests/test_moriio_fabric.py` |
| `23cde1ae` / `ec4a797c` fix(moriio): keep answering a held-back consumer request's remote lookup | D 在重复做 P 已经做过的 prefill | 见下 |

第三个 commit（修复）：`get_num_new_matched_tokens` 原来只在第一次被问时回答"需要远端 KV"，D 满载时请求被挡回 waiting，再问就变成本地 prefill。R7 里 41% 的请求因此在 D 上被重算（见 ISSUES.md）。

单测：容器内 147 passed（加上修复的两个用例后 149 passed）。实机情况：P 和 D 都单独完整起来过，fabric MemPool 分配和注册都正常（R4 的 P、R5 的 D），**但 P 和 D 还没同时在线过，真正的端到端 KV 传输还没跑过**（原因见 `ISSUES.md`：几次都是硬件 MCE 把一侧打掉）。

### 2. Launcher：镜像 kit 的 `k3_server_dspark.sh` → `k3pd_server.sh`（`make_pd_launcher.py` 自动改）

- `ATOM_COMMIT`、`ATOM_DIR` 改成 PD tree（容器内 `/root/.cache/atom-pd` @ 9eddd1ca）
- 新增 `K3_KV_TRANSFER_CONFIG`，原样传给 `--kv-transfer-config`；不设置时与原命令一致

### 3. 运行配置（`pd_up.sh`、`up.sh`、`router.sh`）

- P：`{"kv_role":"kv_producer","kv_connector":"moriio","backend":"fabric","moriio_mode":"write","handshake_port":6301}`
- D：同上，`kv_role` 改为 `kv_consumer`，再加 `"remote_tp_hosts":{"<P0 IP>":["<P0 IP>","<P1 IP>"]}`
- 必须用 write 模式：K3 的 KDA per-request state，moriio 只能 push
- `ATOM_ENABLE_REPLAYSSM=0`：moriio 不搬 ReplaySSM buffer
- `ATOM_DISABLE_MMAP=true`、`ATOM_LOADER_PREFETCH=false`，启动前 drop_caches，再加上 `memguard.sh`：0.5 TB 内存的 tray 上 page cache 会打满，引发 KFD userptr 驱逐风暴（R1）
- `K3_DSPARK=0`；其余沿用镜像 recipe：agentx、16384 batched tokens、FP8 KV block 128、MoE TP8、custom AR 8 MiB
- P 的 max-num-seqs 64，D 的 512
- Router：`atomesh --pd-disaggregation --backend atom`，P 用 `dp_sticky`（按 X-Session-ID），D 用 `round_robin`

### 4. 机器和运维（不是代码改动，但跑通离不开）

- UALink：参与的 tray 必须一起做 DC power cycle、一起 `modprobe amdgpu`。不是同一批起来的 tray 之间做 import 会 oops，进而 panic。对应 `recover.sh`（在不参与服务的机器上执行）
- 宕机后要清掉 Triton 和 torch.compile 缓存（`clean_caches.sh`），否则会读到写了一半的缓存
- 有两台 tray 在 TP8 第一次 forward 时出现过致命 CPU MCE：平台直接复位，journal/pstore/kdump 都没有记录，只有下次启动时的 BERT 记录（BMC 串口日志 `obmc-console-host0.log` 里能看到）

## 文件

| 文件 | 作用 |
|---|---|
| `config.env` | 版本、路径，P/D/router 放在哪几台机器，BMC 地址格式（所有脚本都 source 它；公开的 recipe 里机器名、IP、BMC 都是占位符） |
| `prep_container.sh`、`make_pd_launcher.sh`、`make_pd_launcher.py` | 建 `k3pd` 容器，在其中准备 PD tree（bundle → commit）和 launcher，预先 JIT fabric_copy |
| `atom-k3pd-pd.bundle`、`ATOM_PD_COMMIT` | ATOM 的两个 commit（git bundle）。公开的 recipe 里不带 bundle，可在 ATOM 仓库里生成：`git branch k3pd-multinode-pd 9eddd1ca && git bundle create atom-k3pd-pd.bundle 746624d4..k3pd-multinode-pd` |
| `preflight.sh` | 每台检查：4 个 GPU、无进程、vPOD 1 ACTIVE、numa_balancing 0、容器在跑、ATOM commit 正确 |
| `pd_up.sh`、`up.sh`、`down.sh`、`pd_down.sh` | 起停 P/D（每台各起一半 TP rank）。down 先发 SIGTERM，超时再 SIGKILL，且只对该容器自己的 PID 操作 |
| `router.sh` | atomesh PD router |
| `run_pd.sh`、`mkyaml.sh`、`gate.py`、`summarize.py` | 一次完整测量：起服务 → 等 health → router → gate → warmup → 10 分钟窗口 → 汇总 |
| `memguard.sh` | 空闲内存低于 64 GB 时 drop_caches；KFD 驱逐风暴时停掉本机的 server |
| `clean_caches.sh` | 清掉容器内的 JIT 和编译缓存 |
| `recover.sh <tag>` | 在不参与服务的机器上执行：四台一起冷启动（BMC 地址格式和用户名在 config.env）、一起加载驱动，两两跑 UALink 测试，然后自动起一轮测试 |
| `py/sitecustomize.py`、`py/pd_steps.py` | 每步的 step log 及其分析 |
