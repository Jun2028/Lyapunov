
'''
PBS Scheduler
'''



from logging import getLogger
import os, sys, socket, signal, subprocess
import torch
import torch.distributed as dist

logger = getLogger()

# ---------------- signals ----------------

def _sigusr1(signum, frame):
    logger.warning("SIGUSR1 received; performing clean exit for PBS preemption")
    # PBS requeues job on SIGUSR1 if configured with -r y flag
    sys.exit(-1)

def _sigterm(signum, frame):
    logger.warning("SIGTERM received from PBS; initiating clean shutdown")
    # Perform cleanup if needed before termination
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass
    sys.exit(0)  # Exit gracefully

def init_signal_handler():
    """Handle preemption/time-limit signals on PBS."""
    try: signal.signal(signal.SIGUSR1, _sigusr1) #“When this process receives the SIGUSR1 signal, do not use the default action (which is to terminate). Instead, run the function _sigusr1.”
    except Exception: pass
    try: signal.signal(signal.SIGTERM, _sigterm)
    except Exception: pass
    logger.warning("Signal handlers ready (USR1, TERM).")

# ---------------- helpers ----------------

def _pbs_nodes():
    '''
    Get the list of nodes allocated to this job by PBS.
    '''
    nf = os.getenv("PBS_NODEFILE")
    if nf and os.path.exists(nf):
        seen, nodes = set(), []
        with open(nf) as f:
            for line in f:
                h = line.strip()
                if h and h not in seen:
                    seen.add(h); nodes.append(h)
        return nodes
    return [socket.gethostname()]

def _infer_gpus_per_node():
    '''
    Infer the number of GPUs per node.
    '''
    v = os.getenv("GPUS_PER_NODE")
    if v and v.isdigit():
        return int(v)
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [x for x in cvd.split(",") if x.strip() != ""] #split and filter out empty strings
        if ids: 
            return len(ids)
    try:
        out = subprocess.check_output(["nvidia-smi","-L"], stderr=subprocess.DEVNULL).decode()
        return sum(1 for l in out.splitlines() if "GPU " in l)
    except Exception:
        return 0

# ---------------- main init ----------------

def init_distributed_mode(params):
    """
    PBS / torchrun DDP setup. Mirrors the API used by train.py.
    Sets:
      job_id: PBS job identifier
      n_nodes: Number of allocated nodes
      n_gpu_per_node: GPUs available per node
      world_size: Total number of processes
      global_rank: Global process rank
      local_rank: Local process rank
      is_master: Whether this is the master process
      multi_gpu: Whether using multiple GPUs
      master_addr: Address of the master node
      master_port: Port for distributed coordination
    """
    # Identify context
    nodes = _pbs_nodes()
    params.job_id = os.getenv("PBS_JOBID", "local")
    params.n_nodes = len(nodes)
    params.n_gpu_per_node = _infer_gpus_per_node()

    # torchrun envs
    params.world_size = int(os.getenv("WORLD_SIZE", "1"))
    params.global_rank = int(os.getenv("RANK", "0"))
    params.local_rank = int(os.getenv("LOCAL_RANK", os.getenv("LOCAL_RANK", "0")))
    params.is_master = params.global_rank == 0
    params.multi_gpu = params.world_size > 1

    # rendezvous info (for logging)
    params.master_addr = os.getenv("MASTER_ADDR", nodes[0])
    params.master_port = int(os.getenv("MASTER_PORT", "29500"))

    # device
    if not getattr(params, "cpu", False) and torch.cuda.is_available():
        torch.cuda.set_device(params.local_rank)

    # init process group
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed (env://).")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(init_method="env://", backend=backend)
        if params.is_master:
            logger.info(
                f"DDP ready | WORLD_SIZE={params.world_size} "
                f"RANK={params.global_rank} LOCAL_RANK={params.local_rank} "
                f"NNODES={params.n_nodes} GPUS_PER_NODE={params.n_gpu_per_node}"
            )
