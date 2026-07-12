# GeoCalib-R

Core code for robust object-level spatial calibration in vehicle-infrastructure cooperative perception.

## Contents

- `src/main.cpp`: object-level factor construction, residual filtering, multi-candidate SE(2) correction, sliding-window GTSAM optimization, and trust-region output selection.
- `CMakeLists.txt`: standalone build configuration.
- `train/run_gtsam_full_eval.py`: batch evaluation launcher.
- `train/summarize_gtsam_metrics.py`: metric aggregation helper.

## Dependencies

- C++17 compiler
- GTSAM
- Pangolin
- Eigen
- nlohmann-json
- Python 3 for evaluation scripts

Set optional local paths through environment variables instead of editing source files:

```powershell
$env:GEOCALIB_EXTRA_INCLUDE_DIR = "<extra_include_dir>"
$env:GEOCALIB_DLL_PATHS = "<path1>;<path2>"
```

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release --target v2x_gtsam_viewer
```

## Evaluation

```powershell
python train/run_gtsam_full_eval.py --viewer build/Release/v2x_gtsam_viewer.exe --skip-v2x
python train/run_gtsam_full_eval.py --viewer build/Release/v2x_gtsam_viewer.exe --skip-dair --per-sequence-v2x
```

