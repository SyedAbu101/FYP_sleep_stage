# Development Environment

**Technical specifications for reproducing this project**

---

## 💻 System Information

### Operating System
- **Platform:** macOS
- **Version:** macOS Sequoia (Darwin 25.3.0)
- **Architecture:** ARM64 (Apple Silicon) or x86_64 (Intel)

### Python Environment
- **Python Version:** 3.12.x (or 3.8+)
- **Package Manager:** pip
- **Virtual Environment:** venv (recommended)

### Development Tools
- **IDE/Editor:**
  - Primary: VS Code / PyCharm / Jupyter Notebook
  - Terminal: macOS Terminal / iTerm2
  - Version Control: Git

### Hardware (Development)
- **Processor:** Apple M-series or Intel
- **RAM:** 16 GB (minimum 8 GB)
- **Storage:** 50 GB free space (for data)

---

## 📦 Python Dependencies

### Core Libraries

```
numpy>=1.21.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
scikit-learn>=1.0.0    # Machine learning
xgboost>=1.7.0         # Gradient boosting
scipy>=1.7.0           # Scientific computing
matplotlib>=3.4.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
tqdm>=4.62.0           # Progress bars
pyedflib>=0.1.30       # EDF file reading
```

### Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"
```

---

## 🔧 Hardware Platform (Deployment)

### Target Device: Seeed XIAO nRF52840 Sense

**Specifications:**
- **MCU:** Nordic nRF52840 (ARM Cortex-M4F @ 64 MHz)
- **Flash:** 1 MB
- **RAM:** 256 KB
- **IMU:** LSM6DS3 (6-axis, I2C)
- **Microphone:** PDM microphone
- **Connectivity:** Bluetooth 5.0, USB-C
- **Power:** 3.3V, ~15mA active

**Development Environment:**
- **IDE:** Arduino IDE 2.x
- **Board Package:** Seeed nRF52 Boards (via Board Manager)
- **Libraries:**
  - Seeed_Arduino_LSM6DS3 (IMU)
  - Wire (I2C communication)

---

## 🛠️ IDE Configuration

### VS Code (Recommended)

**Extensions:**
```
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance (Microsoft)
- GitLens (optional)
- Markdown All in One (optional)
```

**Settings (`.vscode/settings.json`):**
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

### Jupyter Notebook

**Installation:**
```bash
pip install jupyter notebook
jupyter notebook
```

**Useful Extensions:**
- jupyterlab-git
- jupyterlab-toc (table of contents)

### PyCharm

**Configuration:**
- Python Interpreter: venv/bin/python
- Project Structure: Mark `src/` as Sources Root
- Enable: Scientific mode for data analysis

---

## 📊 Data Storage

### Directory Structure

```
~/Documents/University/Y3S2/FYP/
├── Fresh_Start/                    # Original working directory
│   ├── PSG_Data/                   # ~2 GB (44 patients)
│   ├── PillowClip_Data/            # ~5 GB (44 patients)
│   └── *.pkl                       # Processed datasets (4-16 MB each)
│
└── Sleep_Stage_Classifier_Clean/  # Clean GitHub repo
    ├── data/                       # Data directory (not in Git)
    ├── src/                        # Source code
    ├── models/                     # Saved models
    └── results/                    # Generated results
```

**Total Storage Required:**
- Raw data: ~7 GB
- Processed data: ~50 MB
- Models: ~1 MB
- Code + docs: ~10 MB

---

## 🔄 Version Control

### Git Configuration

```bash
# Set up user (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize repository
git init
git add .
git commit -m "Initial commit"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/sleep-stage-classifier.git
git branch -M main
git push -u origin main
```

### Git LFS (for large files, if needed)

```bash
# Install Git LFS
brew install git-lfs  # macOS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "models/*.json"
git add .gitattributes
```

---

## 🧪 Testing Environment

### Validation

**Test script:**
```bash
# Test Python environment
python -c "import numpy, pandas, sklearn, xgboost; print('✓ All imports successful')"

# Test data loading
python -c "import pickle; data = pickle.load(open('models/sleep_dataset_optimized.pkl', 'rb')); print(f'✓ Dataset loaded: {data[\"n_features\"]} features')"

# Test model
python -c "import pickle; model = pickle.load(open('models/xgboost_best.pkl', 'rb')); print('✓ Model loaded')"
```

### Performance Benchmarks

**Expected training times (on MacBook Pro M1/M2):**
- Data loading: ~1 second
- Feature engineering: ~5 seconds
- XGBoost training (5-fold CV): ~5 minutes
- LSTM training: ~30 minutes
- Feature importance: ~2 minutes

---

## 📝 Known Issues & Solutions

### Issue 1: Scikit-learn Version Mismatch

**Error:**
```
InconsistentVersionWarning: Trying to unpickle estimator from version X when using version Y
```

**Solution:**
```bash
pip install scikit-learn==1.5.2  # Match training version
```

### Issue 2: Memory Error on Large Dataset

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Load dataset in chunks or reduce features
import gc
gc.collect()  # Force garbage collection
```

### Issue 3: XGBoost Not Found

**Error:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
pip install xgboost
# Or for Apple Silicon:
pip install xgboost --no-binary :all:
```

### Issue 4: Arduino Compilation Error

**Error:**
```
fatal error: LSM6DS3.h: No such file or directory
```

**Solution:**
```
Arduino IDE → Sketch → Include Library → Manage Libraries
Search: "Seeed Arduino LSM6DS3" → Install
```

---

## 🔍 Reproducibility

### Ensuring Reproducible Results

**Set random seeds:**
```python
import numpy as np
import random

# Set seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# XGBoost
model = XGBClassifier(random_state=SEED)

# Scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
```

### Cross-Platform Compatibility

**Differences to note:**
- File paths: Use `pathlib.Path()` or `os.path.join()`
- Line endings: Git handles automatically
- Python pickle: May differ between Python versions
- XGBoost models: Should be compatible across platforms

**Recommendation:** Export to ONNX for maximum compatibility
```python
# Optional: Export to ONNX for cross-platform use
import onnxmltools
from onnxmltools.convert import convert_xgboost

onnx_model = convert_xgboost(xgb_model)
onnxmltools.utils.save_model(onnx_model, 'model.onnx')
```

---

## 📚 Additional Resources

### Documentation
- **XGBoost:** https://xgboost.readthedocs.io/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Arduino nRF52:** https://wiki.seeedstudio.com/XIAO_BLE/
- **Pandas:** https://pandas.pydata.org/docs/

### Tutorials
- XGBoost parameter tuning
- Patient-level cross-validation
- Embedded ML deployment

### Community
- Stack Overflow (tag: `xgboost`, `scikit-learn`)
- GitHub Issues
- NTU CS forums

---

## ✅ Environment Setup Checklist

**Development Environment:**
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Data directory structure set up
- [ ] Git configured
- [ ] IDE configured

**Hardware Environment:**
- [ ] Arduino IDE 2.x installed
- [ ] Seeed nRF52 board package installed
- [ ] LSM6DS3 library installed
- [ ] XIAO nRF52840 board connected
- [ ] Drivers installed (if needed)

**Verification:**
- [ ] `python -c "import xgboost; print(xgboost.__version__)"` works
- [ ] Sample dataset loads successfully
- [ ] Test script runs without errors
- [ ] Arduino sketch compiles successfully

---

**Last Updated:** March 2025
**Compatibility:** Python 3.8+, macOS/Linux/Windows, Arduino IDE 2.x
