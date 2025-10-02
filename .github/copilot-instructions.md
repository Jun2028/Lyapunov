# Copilot Instructions for Lyapunov Codebase

## Project Overview
This repository implements symbolic transformer models for discovering global Lyapunov functions, as described in the NeurIPS 2024 paper "Symbolic Transformers for Global Lyapunov Function Discovery". The system generates training data for polynomial/non-polynomial dynamical systems, trains transformers to predict Lyapunov functions, and evaluates using Sum-of-Squares (SOS) checkers.

## Architecture & Key Components
- **`train.py`**: Main entry point with dual modes - data generation (`--export_data true`) or training (`--export_data false`)
- **`src/envs/ode.py`**: ODEEnvironment class that generates symbolic dynamical systems and Lyapunov functions. Contains all CLI flags in `register_args()` method (2600+ lines)
- **`src/model/transformer.py`**: Custom transformer with multihead attention, positional encodings, and beam search for sequence-to-sequence translation
- **`src/trainer.py`**: Training loop with gradient accumulation, distributed training support, and early stopping based on validation metrics
- **`src/SOS_utils.py`**: Sum-of-squares polynomial verification for checking Lyapunov function validity
- **`create_dataset.py`**: Post-processing script for deduplication, cleaning, and train/validation/test splits

## Critical Developer Workflows

### 1. Data Generation (requires high CPU count)
```bash
python train.py --dump_path /path/to/storage/ --export_data true --cpu true --reload_data '' --env_base_seed -1 --num_workers 20
```
Uses JSON configs: `generate_bwd_poly.json`, `generate_bwd_nonpoly.json`, `generate_fwd_poly.json`

### 2. Dataset Creation (mandatory step)
```bash
python create_dataset.py  # Edit paths in file first
```
Produces: `.train`, `.cleaned.valid`, `.cleaned.test` files. Must run after generation, before training.

### 3. Training
```bash
python train.py --reload_data "ode_lyapunov,/path/to/dataset.train,/path/to/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL"
```

## Project-Specific Patterns & Conventions

### Data Format
- Input/Output: Tab-separated `system \t lyapunov_function` 
- Symbolic expressions use prefix notation: `+ * INT+ 2 ^ x0 INT+ 3 ^ x1 INT+ 2` = `2*x0^3 + x1^2`
- Special tokens: `<SPECIAL_3>` separates system equations, `INT+`/`INT-` for positive/negative integers
- Benchmark files (e.g., `benchmarks/BPoly`) contain pre-computed test cases

### Environment Configuration Pattern
- All environment flags centralized in `src/envs/ode.py::register_args()` 
- Key flags: `lyap_polynomial_H`, `lyap_pure_polynomial`, `lyap_SOS_checker`, `lyap_SOS_fwd_gen`
- Generation modes: backward (from known Lyapunov function) vs forward (guess-and-check with SOS)
- Model flags in `train.py::get_parser()`: architecture, training, beam search parameters

### Training Configuration
- Uses JSON files for hyperparameter sweeps (see `train.json`)
- Evaluation-only mode: `--eval_only` with `--eval_from_exp` to load pickled parameters
- Distributed training via `init_distributed_mode()` and PyTorch DDP
- Early stopping on `valid_ode_lyapunov_beam_acc` metric by default

## Integration & External Dependencies
- **SymPy**: Symbolic mathematics, expression manipulation and simplification
- **dReal**: SMT solver for constraint satisfaction (Linux/Mac only)
- **PyTorch**: Deep learning framework with custom transformer implementation  
- **Conda environments**: `Lyapunov_python_3_9.yml` (local) or `Lyapunov_python_3_10.yml` (cluster)
- **SOS checking**: Custom implementation in `SOS_utils.py` for polynomial verification

## Essential Development Notes
- Clear SymPy cache periodically (`CLEAR_SYMPY_CACHE_FREQ = 10000`) to prevent memory leaks
- Forward generation with SOS checking is significantly slower than backward generation
- Batch size constraints: Use small `batch_size` (4) for training, larger `batch_size_eval` (16) for evaluation
- Path templating: Update paths in `create_dataset.py` and JSON configs before use
- Debug mode: `--debug` flag sets `exp_name="debug"` and random `exp_id`
