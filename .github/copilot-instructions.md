# Copilot Instructions for Lyapunov Codebase

## Project Overview
# Copilot Instructions for Lyapunov Codebase

## Project Overview
This repository implements symbolic transformer models for discovering global Lyapunov functions, as described in the NeurIPS 2024 paper "Symbolic Transformers for Global Lyapunov Function Discovery" (DOI: 10.48550/arXiv.2406.12345). The codebase is organized for dataset generation, cleaning, and model training/evaluation, primarily using PyTorch.

## Architecture & Key Components
- **src/**: Core logic
  - `envs/ode.py`: Environment and argument registration (see `register_args` for all CLI flags)
  - `model/transformer.py`: Transformer model implementation
  - `trainer.py`: Training loop and evaluation
  - `SOS_utils.py`: Sum-of-squares checking utilities
  - `utils.py`, `logger.py`, `optim.py`, `slurm.py`: Supporting utilities
- **benchmarks/**: Contains benchmark datasets (BPoly, FBarr, FLyap, FSOSTOOL)
- **train.py**: Main entry for dataset generation and training. Uses many CLI flags (see `get_parser` for full list).
- **create_dataset.py**: Cleans and merges generated datasets before training.

## Developer Workflows
- **Environment Setup**: Use `Lyapunov_python_3_9.yml` (local) or `Lyapunov_python_3_10.yml` (cluster) to create a conda environment. Activate with `conda activate Lyapunov`.
- **Dataset Generation**: Run `train.py` with `--export_data true` and other flags to generate samples. Example:
  ```
  python train.py --dump_path /path/to/storage/ --export_data true --cpu true --reload_data '' --env_base_seed -1 --num_workers 20
  ```
- **Dataset Cleaning**: Edit paths in `create_dataset.py` as needed, then run:
  ```
  python create_dataset.py
  ```
  Produces `.train`, `.cleaned.valid`, `.cleaned.test` files.
- **Training**: Launch with `train.py`, specifying datasets and benchmarks in `--reload_data`. Example:
  ```
  python train.py --reload_data "ode_lyapunov,/path/to/dataset.train,/path/to/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL"
  ```
- **Flags**: Most workflow flags are defined in `ode.py` (`register_args`) and `train.py` (`get_parser`).

## Project-Specific Patterns & Conventions
- **Data Flow**: Generation → Cleaning → Training. Always clean datasets before training.
- **Benchmarks**: Always reference benchmarks via relative paths in CLI flags.
- **Sum-of-Squares Checking**: Use flags like `--lyap_SOS_checker` and `--lyap_SOS_fwd_gen` for SOS-based evaluation/generation.
- **Model Architecture**: Transformer parameters (layers, heads, embedding dim) are set via CLI flags.
- **Storage**: Output paths are controlled by `--dump_path` and appear in `.stderr` logs.

## Integration & External Dependencies
- **PyTorch**: Main ML framework.
- **dReal**: Required for environments such as `envs/ode.py` when using the `--lyap_SOS_checker` or `--lyap_SOS_fwd_gen` flags. Note: dReal is not natively supported on Windows; Windows users should use WSL (Windows Subsystem for Linux) or a Linux cluster, or disable SOS checking by omitting related flags.
- **Conda**: Required for environment management.

## Examples
- Dataset generation:
  ```python
  python train.py --export_data true --dump_path /tmp/dataset --cpu true --num_workers 8
  ```
- Training:
  ```python
  python train.py --reload_data "ode_lyapunov,/tmp/dataset.train,/tmp/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL" --batch_size 4 --n_enc_layers 6
  ```

## References
- For full flag documentation, see `register_args` in `src/envs/ode.py` and `get_parser` in `train.py`.
- For benchmarks, see files in `benchmarks/`.
- For model details, see `src/model/transformer.py`.

If any section is unclear or missing details, please specify which workflows, flags, or architectural choices need further documentation.
