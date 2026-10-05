# Interleaved reasoning and reasoning probe

Python workflows for interleaved reasoning rollouts, PPO training, and the
separate reasoning probe. The original notebooks are retained in
`original_notebooks/`.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | The entry point: load data/model, collect rollouts, train, and save. |
| `interleaved_reasoning/model.py` | Model, tokenizer, LoRA, and PPO value head. |
| `interleaved_reasoning/data.py` | Load and group the CSV; exclude ambiguous tasks. |
| `interleaved_reasoning/prompts.py` | Training prompts and input formatting. |
| `interleaved_reasoning/rollouts.py` | Answer checks, rewards, rollout collection, and trajectory export. |
| `interleaved_reasoning/ppo.py` | PPO loss, optimization loop, checkpoints, and final adapter export; imported by `train.py`. |
| `interleaved_reasoning/generation.py` | Reusable greedy response generation for the PPO model wrapper. |
| `probe.py` | Run the reasoning probe and save parsed responses. |
| `paths.json` | Input, output, and checkpoint locations. |
| `paths.py` | Resolve paths and provide optional Colab utilities. |
| `check_kk_rollout_results.ipynb` | Check saved verifier results. |

Notebook previews, bare inspection expressions, toy checks, sample comparisons,
commented-out alternative implementations, and their CLI options have been
removed from the Python files. Model selection, prompts, hyperparameters,
reward calculations, sampling, and PPO updates retain the notebook logic.
Print statements inside the retained original functions are preserved.

## Setup and paths

Use the working notebook's model environment where possible. Dependencies are
listed in `requirements.txt`; their exact versions were not recorded in the
notebooks. Model execution requires a compatible GPU/quantization setup.
Python 3.12+ also supports the unchanged reference notebooks.

```bash
python -m pip install -r requirements.txt
```

Edit `paths.json` before running. Relative paths are resolved from its directory;
absolute paths also work. For local settings, copy it to `paths.local.json` and
pass `--paths paths.local.json`. Local settings, datasets, and generated outputs
are excluded from Git by `.gitignore`.

The training CSV must contain `task_id`, `quiz_text`, `reasoning_step`,
`sub_answer`, `row_claims`, and `final_answer`, using the original puzzle text
and claim formats. The probe expects the original `engaged_tests_all.json`
structure. Data and model weights are not included.

## Run

Run `train.py` for the training workflow. `interleaved_reasoning/ppo.py` holds
the PPO implementation and is imported by `train.py`; it is not a second
training command. Its code is unchanged from the former `training.py`.

The entry point separates argument parsing, the training workflow, and optional
Colab setup. The workflow reads in order: check paths, load model/data,
collect or load rollouts, train, and save.

```bash
python train.py
python probe.py
```

Both commands accept `--paths paths.local.json`. Run the stages separately with:

```bash
python train.py --stage rollouts
python train.py --stage train
```

Training reuses saved trajectories. Keep their tensor device compatible with
the current model environment. Set `resume_from_checkpoint` when needed; the
original resume behavior is retained. It does not restore optimizer or random
state, and an end-of-epoch checkpoint repeats that saved epoch.

`export_copy_dir` is optional: leave it `null` or choose a second destination for
the final adapter/tokenizer export. The reusable `generate_response()` function
in `generation.py` expects a PPO model wrapper and prompt IDs; its device uses
the original CUDA-if-available selection and must match the supplied model.

For Colab, use mounted Drive paths in the JSON and pass `--mount-drive` if a
mount is needed. `--disconnect` explicitly disconnects the runtime after success.
These actions are not run when importing the project.

## Results notebook

Set `verifier_data_dir` to the results bundle containing `data/`, `tests/`,
`parser_outputs/`, and `labels_and_reviews/`. Run
`check_kk_rollout_results.ipynb` with this project folder as its working directory
so it can import `paths.py`. It uses the project `paths.json`.

The working results notebook retains its earlier print cleanup and configured
`DATA_DIR`. Its saved outputs have not been rerun. All notebooks are unchanged
by the Python cleanup; untouched source copies remain in `original_notebooks/`.

## Validation

The cleanup was checked against the preceding package for retained function
source and computational behavior. Syntax, CLI, path, data, parsing, and stage
wiring checks were performed. Model operations in wiring checks were mocked;
real GPU model loading, generation, and training were not run here.
