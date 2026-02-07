# Event-Based Vision - Prophesee SDK Standalone

This folder contains a complete standalone copy of the Prophesee Metavision SDK for event-based vision processing, ready to use locally or upload to cloud environments.

## Contents

- **`.venv/`** - Python 3.9 virtual environment with all dependencies installed
- **`bin/`** - Native DLL binaries required by the SDK
- **`sdk/`** - Complete Prophesee SDK Python packages (also copied to `.venv/Lib/site-packages/`)
- **`samples/`** - All SDK sample scripts organized by category:
  - `analytics/` - Vibration estimation, frequency analysis
  - `cv/` - Optical flow, tracking
  - `cv3d/` - 3D model tracking
  - `core/` - Core event processing
  - `core_ml/` - Machine learning demos
  - And more...

## Setup

### Local Windows Use

```powershell
# Add binaries to PATH
$env:PATH = "C:\Users\z003n5uc\Desktop\event-based\bin;" + $env:PATH

# Activate virtual environment
.\\.venv\Scripts\Activate.ps1

# Verify installation
python -c "import metavision_core; print('SDK loaded:', metavision_core.__file__)"
```

### Cloud/Remote Use

1. **Upload this entire folder** to your cloud environment
2. Ensure Python 3.9 is available
3. Activate the venv: `source .venv/bin/activate` (Linux) or `.venv\Scripts\activate` (Windows)
4. Set PATH to include the `bin/` folder for native libraries
5. Run samples from the `samples/` directory

## Running Samples

### Vibration Estimation
```powershell
python samples\analytics\python_samples\metavision_vibration_estimation\metavision_vibration_estimation.py --help
```

### Sparse Optical Flow
```powershell
python samples\cv\python_samples\metavision_sparse_optical_flow\metavision_sparse_optical_flow.py --help
```

### 3D Model Tracking
```powershell
python samples\cv3d\python_samples\metavision_model_3d_tracking\metavision_model_3d_tracking.py --help
```

## Installed Python Packages

- numpy==1.24.4
- opencv-python==4.11.0.86
- h5py==3.14.0
- scikit-video==1.1.11
- scipy==1.13.1
- pillow==11.3.0
- Prophesee Metavision SDK (all modules)

## Notes

- Requires Python 3.9 (SDK compiled for this version)
- Native DLLs in `bin/` must be accessible via PATH
- Samples expect event camera data files (.raw or .hdf5) or a connected Prophesee camera
- For cloud use, you may need to install additional system dependencies for OpenCV and HDF5

## Cloud Platform Specifics

### Claude Code (Web)
- Upload this folder as a workspace
- SDK packages are already in `.venv/Lib/site-packages`
- May have limited access to native DLLs (some functionality may require local execution)

### Google Colab / Jupyter
```python
import sys
sys.path.insert(0, '/path/to/event-based/sdk')
```

### Docker
Copy the entire folder and use Python 3.9 base image. Ensure all system libraries for OpenCV are installed.
