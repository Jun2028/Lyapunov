# Copilot Instructions for Lyapunov Codebase

## Project Overview
# Copilot Instructions for Lyapunov Codebase

## Project Overview
This repository implements symbolic transformer models for discovering global Lyapunov functions, as described in the NeurIPS 2024 paper "Symbolic Transformers for Global Lyapunov Function Discovery" (DOI: 10.48550/arXiv.2406.12345). The codebase is organized for dataset generation, cleaning, and model training/evaluation, primarily using PyTorch.

## Architecture & Key Components
  - `envs/ode.py`: Environment and argument registration (see `register_args` for all CLI flags)
  - `model/transformer.py`: Transformer model implementation
  - `trainer.py`: Training loop and evaluation
  - `SOS_utils.py`: Sum-of-squares checking utilities
  - `utils.py`, `logger.py`, `optim.py`, `slurm.py`: Supporting utilities

## Developer Workflows
  ```
  python train.py --dump_path /path/to/storage/ --export_data true --cpu true --reload_data '' --env_base_seed -1 --num_workers 20
  ```
  ```
  python create_dataset.py
  ```
  Produces `.train`, `.cleaned.valid`, `.cleaned.test` files.
  ```
  python train.py --reload_data "ode_lyapunov,/path/to/dataset.train,/path/to/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL"
  ```

## Project-Specific Patterns & Conventions

## Integration & External Dependencies

## Examples
  ```python
  python train.py --export_data true --dump_path /tmp/dataset --cpu true --num_workers 8
  ```
  ```python
  python train.py --reload_data "ode_lyapunov,/tmp/dataset.train,/tmp/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL" --batch_size 4 --n_enc_layers 6
  ```

## References

If any section is unclear or missing details, please specify which workflows, flags, or architectural choices need further documentation.
