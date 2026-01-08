# Deployment Guide for Plotly Cloud

This guide explains how to deploy the ComfortRoom app to Plotly Cloud with optional ML framework dependencies.

## Overview

The app supports three ML models:
- **Random Forest (scikit-learn)** - Always available, lightweight
- **TensorFlow Neural Network** - Optional, requires tensorflow package
- **PyTorch Neural Network** - Optional, requires torch package

## Deployment Configurations

### Option 1: Lightweight Deployment (Recommended for Plotly Cloud)

Use the standard `requirements.txt` which includes only scikit-learn. This avoids the large torch and tensorflow packages that can cause deployment issues.

**Features:**
- Random Forest model (fully functional)
- Fast deployment
- Lower memory footprint
- No installation issues

**To deploy:**
```bash
# requirements.txt already configured for lightweight deployment
# Just deploy normally - torch and tensorflow are excluded
```

The app will automatically:
- Detect missing libraries at runtime
- Disable unavailable model options in the UI
- Default to Random Forest
- Show clear messages if users try to select unavailable models

### Option 2: Full Deployment (Local or Cloud with Large Instance)

For local development or cloud instances with sufficient resources, you can install all dependencies including torch and tensorflow.

**To install all dependencies locally:**
```bash
pip install -r requirements-full.txt
```

## How It Works

### 1. Optional Import Handling

The app uses try-except blocks to gracefully handle missing libraries:

```python
# TensorFlow (Neural Network)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
```

### 2. Dynamic UI Updates

The app automatically adjusts the UI based on available libraries:
- Disables unavailable model options in dropdowns
- Shows "(Not Available)" labels for missing models
- Defaults to the best available model (PT > NN > RF)

### 3. Runtime Fallbacks

When a model is not available:
- The training functions return `None` with error info
- UI displays helpful warning messages
- Users are guided to select available models

## Files

- **requirements.txt** - Lightweight dependencies (scikit-learn only)
- **requirements-full.txt** - Full dependencies (includes torch + tensorflow)

## Switching Between Configurations

### For Plotly Cloud Deployment:
Ensure `requirements.txt` does NOT include torch or tensorflow:
```bash
# These should NOT be in requirements.txt for Plotly Cloud
# tensorflow>=2.18.0  # REMOVED
# torch>=2.0.0        # REMOVED
```

### For Local Development with All Features:
```bash
pip install -r requirements-full.txt
```

## Troubleshooting

### Issue: Models show as "Not Available"
**Cause:** The required library (torch or tensorflow) is not installed.
**Solution:** This is expected on Plotly Cloud with the lightweight deployment. Use Random Forest, or install full dependencies locally.

### Issue: Deployment fails with "Package too large"
**Cause:** torch or tensorflow are in requirements.txt
**Solution:** Remove them from requirements.txt and deploy with scikit-learn only.

### Issue: Model performance differs between deployments
**Cause:** Different models available in different environments
**Solution:** This is expected. Random Forest provides good performance for production deployment.

## Model Performance Comparison

All three models provide similar accuracy for the comfort prediction task:
- **Random Forest**: Fast training, good accuracy, lightweight
- **TensorFlow NN**: Similar accuracy, larger package size
- **PyTorch NN**: Similar accuracy, larger package size

For production deployment, Random Forest is recommended as it provides the best balance of performance, size, and reliability.

## Environment Variables (Optional)

You can force specific models to be disabled even if libraries are available:

```bash
export DISABLE_TENSORFLOW=1
export DISABLE_PYTORCH=1
```

## Testing Before Deployment

Test with lightweight dependencies locally:

```bash
# Create a new virtual environment
python -m venv venv-minimal
source venv-minimal/bin/activate  # On Windows: venv-minimal\Scripts\activate

# Install lightweight dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Verify that:
1. App starts without errors
2. Random Forest model is available
3. TensorFlow and PyTorch show as "Not Available"
4. All features work with Random Forest

## Support

If you encounter deployment issues:
1. Check that requirements.txt does not include torch or tensorflow
2. Verify the app works locally with lightweight dependencies
3. Check Plotly Cloud logs for any error messages
