# Manifest

## Core implementation

- `src/main.cpp`: object-level spatial calibration, geometric filtering, multi-candidate pose correction, GTSAM factor graph optimization, trust-region selection, and metric output.
- `CMakeLists.txt`: standalone CMake build entry.
- `train/run_gtsam_full_eval.py`: batch evaluation launcher.
- `train/summarize_gtsam_metrics.py`: CSV aggregation and summary helper.

## Build inputs expected at runtime

- Exported DAIR-V2X or V2X-Real world-scene folders with `fused_inference_results.json`.
- Local GTSAM, Pangolin, Eigen, and nlohmann-json installations.
- Optional environment variables:
  - `GEOCALIB_EXTRA_INCLUDE_DIR`: additional include directory for local headers.
  - `GEOCALIB_DLL_PATHS`: runtime DLL search paths separated by the platform path separator.

## Not included

Datasets, exported point clouds, build directories, screenshots, checkpoints, caches, figure scripts, rendered images, and document files are not included.
