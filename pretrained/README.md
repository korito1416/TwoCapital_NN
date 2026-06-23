# Pretrained Checkpoints

This folder contains the minimal TensorFlow checkpoint files needed to reproduce
the current training workflow without machine-local absolute paths.

## Folders

- `nber_legacy/` is the four-regime NBER warm start that the six model classes
  used to load from `/project/lhansen/Cap_NN_oldVersion/...`.
- `two_stage_tech_base/` is the six-regime two-stage technology checkpoint used
  as the common pretrained root for downstream one-jump, intensity, and
  fine-tuning runs.

Only loadable checkpoint files and small metadata files are included:
`checkpoint`, `*.index`, `*.data-*`, `params*.txt`, and `training_history.csv`.
Large plots, TensorBoard logs, simulations, and diagnostics remain in ignored
`output*` folders.

## Usage

The six files in `models/` now resolve the NBER warm start through
`models/pretrained_paths.py`. By default they use:

```bash
pretrained/nber_legacy
```

To override it:

```bash
export NBER_PRETRAINED_FOLDER=/path/to/nber_legacy
```

For runs that take a `PRETRAINED_FOLDER`, use the bundled base checkpoint:

```bash
export PRETRAINED_FOLDER="$PWD/pretrained/two_stage_tech_base"
```

For simulation from the bundled base:

```bash
python models/SimulationStochasticJumps.py \
  --export-folder pretrained/two_stage_tech_base \
  --xi 0.05
```

Historical `params.txt` files are kept as provenance and may still contain old
absolute paths. The loading code uses the checkpoint prefixes in this folder.

## Integrity

Verify the bundle after clone with:

```bash
cd pretrained
sha256sum -c MANIFEST.sha256
```
