# Cross-Sensor Fingerprint Matcher

A Python-based demo application for matching **contactless fingerprint photos** with **contact-based sensor scans** using SIFT/ORB feature matching.

![Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)

## 🎯 Overview

This application demonstrates the challenges and techniques involved in cross-sensor fingerprint matching - a critical problem in biometric interoperability. It bridges the gap between:

- **Contactless fingerprints**: Captured via smartphone cameras or digital cameras
- **Contact-based fingerprints**: Captured via traditional touch sensors (like those used in border control, banking, etc.)

## 🚀 Quick Start

### 1. Clone/Download the Repository

```bash
cd fingerprint_matcher
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Demo

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📸 Usage Options

### Option A: Upload Your Own Images (No Dataset Required)

1. Select "**Upload Images**" in the sidebar
2. Upload a contactless fingerprint photo (from a smartphone camera)
3. Upload a contact-based fingerprint scan (from a sensor)
4. Click "**Run Matching Algorithm**"

### Option B: Use the PolyU Dataset

If you have access to the [PolyU Contactless 2D to Contact-based 2D Fingerprint Database](http://www.comp.polyu.edu.hk/~csajaykr/fingerprint.htm):

1. Place the dataset folders in the parent directory:
   ```
   parent_folder/
   ├── fingerprint_matcher/          # This application
   ├── contactless_2d_fingerprint_images/
   ├── contact-based_fingerprints/
   └── processed_contactless_2d_fingerprint_images/
   ```
2. Select "**PolyU Dataset Browser**" in the sidebar
3. Browse and select fingerprint pairs by finger ID

## 🔧 Features

### Preprocessing Pipeline
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Gabor Filtering**: Ridge pattern enhancement
- **Bilateral Filtering**: Edge-preserving noise reduction
- **Binarization**: Adaptive thresholding
- **Skeletonization**: Ridge thinning to 1-pixel width

### Matching Algorithms
- **SIFT**: Scale-Invariant Feature Transform (more accurate, slower)
- **ORB**: Oriented FAST and Rotated BRIEF (faster, patent-free)
- **Lowe's Ratio Test**: Filter ambiguous matches
- **Homography Filtering**: Geometric consistency via RANSAC

### Visualization
- Side-by-side enhanced images
- Keypoint correspondence map
- Match statistics and confidence scores

## 📊 Command-Line Testing

You can also run matching from the command line:

```bash
# Match two images
python matcher.py <contactless_image> <contact_image> [sift|orb|both]

# Example
python matcher.py photo.jpg scan.jpg sift
```

Run the test suite:

```bash
python test_matching.py
```

## ⚠️ Important Notes

### Cross-Sensor Matching is Challenging

The accuracy of this demo is limited because:

1. **Domain Gap**: Contactless photos and contact scans have fundamentally different characteristics
2. **Scale/Perspective**: Photos lack the "flattening" effect of touch sensors
3. **Feature Mismatch**: SIFT/ORB features are general-purpose, not fingerprint-specific

For production-level accuracy, consider:
- Deep learning approaches (Siamese networks)
- Minutiae-based matching (MCC descriptors)
- Domain adaptation techniques

### Reference

This demo is designed for the PolyU database:

> Chenhao Lin, Ajay Kumar, "Matching Contactless and Contact-based Conventional Fingerprint Images for Biometrics Identification," IEEE Transactions on Image Processing, vol. 27, pp. 2008-2021, April 2018.

## 📁 Project Structure

```
fingerprint_matcher/
├── app.py              # Streamlit web application
├── preprocessing.py    # Image enhancement pipeline
├── matcher.py          # SIFT/ORB feature matching
├── test_matching.py    # Test scenarios
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── samples/            # Sample images for testing
    ├── contactless_sample.png
    └── contact_sample.png
```

## 🛠️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| Matching Algorithm | SIFT, ORB, or Both | SIFT |
| Enhancement Level | light, standard, heavy | standard |
| Ratio Threshold | Lowe's ratio test (0.5-0.9) | 0.8 |
| Use Homography | Enable RANSAC filtering | True |
| Match Threshold | Min inliers for match decision | 15 |

## 📝 License

This code is provided for research and educational purposes. The PolyU database has its own usage restrictions - please refer to their license terms.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Deep learning matching models
- Minutiae extraction and matching
- Better preprocessing for low-quality images
- Additional fingerprint databases support
