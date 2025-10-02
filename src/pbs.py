
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
    logger.warning("SIGUSR1 received; exiting for scheduler to resubmit if configured.")
    sys.exit(-1) #force exit with exit code -1

    #This needs to be rewritten to use the same logic as in slurm.py

def _sigterm(signum, frame):
    logger.warning("SIGTERM received; bypassing.") # do nothing, 
    #
    # This better be changed

def init_signal_handler():
    """Handle preemption/time-limit signals on PBS."""
    try: signal.signal(signal.SIGUSR1, _sigusr1) #“When this process receives the SIGUSR1 signal, do not use the default action (which is to terminate). Instead, run the function _sigusr1.”
    except Exception: pass
    try: signal.signal(signal.SIGTERM, _sigterm)
    except Exception: pass
    logger.warning("Signal handlers ready (USR1, TERM).")

# ---------------- helpers ----------------

def _pbs_nodes():
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
    v = os.getenv("GPUS_PER_NODE")
    if v and v.isdigit():
        return int(v)
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [x for x in cvd.split(",") if x.strip() != ""]
        if ids: return len(ids)
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
      is_slurm_job, job_id, n_nodes, n_gpu_per_node, world_size,
      global_rank, local_rank, is_master, multi_gpu, master_addr, master_port.
    Also backfills SLURM_* envs from PBS_* for utils.get_dump_path().
    """
    # Backfill env so existing utils pick up job ids
    if "PBS_JOBID" in os.environ and "SLURM_JOB_ID" not in os.environ:
        os.environ["SLURM_JOB_ID"] = os.environ["PBS_JOBID"]
    if "PBS_ARRAYID" in os.environ and "SLURM_ARRAY_TASK_ID" not in os.environ:
        os.environ["SLURM_ARRAY_TASK_ID"] = os.environ["PBS_ARRAYID"]

    # Identify context
    nodes = _pbs_nodes()
    params.is_slurm_job = False                         # avoid problem with existing code
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
