# Cross-Sensor Fingerprint Matching: Experiment Report

**Project:** Contactless to Contact-Based Fingerprint Matching  
**Date:** January 21, 2026  
**Dataset:** PolyU Contactless 2D to Contact-based 2D Fingerprint Database

---

## 1. Executive Summary

This report documents our experiments in matching **contactless fingerprint photos** captured via digital cameras with **contact-based fingerprint scans** from touch sensors. We implemented a feature-based matching pipeline using SIFT and ORB algorithms and evaluated performance on the PolyU dataset.

### Key Findings

| Metric | SIFT | ORB |
|--------|------|-----|
| Best Accuracy | 66.67% | 66.67% |
| Avg Genuine Inliers | 3.26 | 2.74 |
| Avg Impostor Inliers | 3.02 | 2.58 |
| Discrimination Ratio | 1.08x | 1.06x |

> **Conclusion:** Feature-based matching (SIFT/ORB) provides limited discrimination for cross-sensor fingerprint matching due to the significant domain gap between contactless photos and contact sensor scans.

---

## 2. Problem Statement

### 2.1 Background

Traditional fingerprint recognition systems use **contact-based sensors** where users press their finger against a surface. However, there's growing interest in **contactless fingerprint capture** using smartphone cameras due to:

- Hygiene concerns (no surface contact)
- Convenience (use existing smartphones)
- Cost reduction (no specialized hardware)

### 2.2 Challenge: Cross-Sensor Interoperability

The core challenge is matching fingerprints across different capture modalities:

| Aspect | Contactless (Camera) | Contact-Based (Sensor) |
|--------|---------------------|------------------------|
| Resolution | 1400×900 pixels (varied) | 328×356 pixels (fixed) |
| Capture | Finger photographed at distance | Finger pressed on surface |
| Distortion | Perspective, rotation | Elastic deformation |
| Lighting | Variable, shadows | Controlled illumination |
| Ridge Clarity | Depends on focus/lighting | Consistent ridge capture |

### 2.3 Research Question

> Can traditional computer vision feature matching (SIFT/ORB) effectively match contactless fingerprint photos with contact-based sensor scans?

---

## 3. Methodology

### 3.1 Dataset

**PolyU Contactless 2D to Contact-based 2D Fingerprint Database**

- **Source:** Hong Kong Polytechnic University
- **Total Images:** 5,952 fingerprint images
- **Subjects:** 336 different fingers
- **Samples per finger:** 6 images each
- **Two Sessions:** With 2-24 month gap

Reference:
> Chenhao Lin, Ajay Kumar, "Matching Contactless and Contact-based Conventional Fingerprint Images for Biometrics Identification," IEEE TIP, 2018.

### 3.2 Proposed Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Grayscale Conversion                                         │
│  2. Intensity Normalization                                      │
│  3. CLAHE Enhancement (Contrast Limited Adaptive Histogram Eq.)  │
│  4. Gabor Filtering (Ridge Pattern Enhancement)                  │
│  5. Resolution Normalization (400×400)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                            │
├─────────────────────────────────────────────────────────────────┤
│  • SIFT: Scale-Invariant Feature Transform (10,000 keypoints)   │
│  • ORB: Oriented FAST and Rotated BRIEF (10,000 keypoints)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MATCHING & FILTERING                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Brute-Force KNN Matching (k=2)                               │
│  2. Lowe's Ratio Test (threshold=0.8)                            │
│  3. Homography Estimation with RANSAC                            │
│  4. Inlier Count as Match Score                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Preprocessing Details

| Stage | Technique | Purpose |
|-------|-----------|---------|
| CLAHE | clipLimit=3.0, tileSize=8×8 | Enhance local contrast without amplifying noise |
| Gabor | 8 orientations, σ=4.0 | Respond to ridge patterns at multiple angles |
| Bilateral Filter | d=9, σ_color=75 | Reduce noise while preserving ridge edges |
| Resize | 400×400 pixels | Normalize resolution for cross-sensor matching |

### 3.4 Feature Matching Details

**SIFT Configuration:**
- `nfeatures=10000`
- `contrastThreshold=0.02`
- `edgeThreshold=15`

**ORB Configuration:**
- `nfeatures=10000`
- `scaleFactor=1.1`
- `nlevels=12`

**Matching:**
- Lowe's Ratio Test threshold: 0.8
- Homography filtering with RANSAC (5.0 reprojection error)

---

## 4. Experimental Setup

### 4.1 Experiment Design

| Experiment | Description | Sample Size |
|------------|-------------|-------------|
| **Genuine Matches** | Same finger, corresponding samples (contactless p1.bmp ↔ contact X_1.jpg) | 50 pairs |
| **Impostor Matches** | Different fingers (contactless from finger A ↔ contact from finger B) | 100 pairs |

### 4.2 Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **True Positive (TP)** | Genuine pair correctly identified as match |
| **False Negative (FN)** | Genuine pair incorrectly rejected |
| **True Negative (TN)** | Impostor pair correctly rejected |
| **False Positive (FP)** | Impostor pair incorrectly accepted |
| **Accuracy** | (TP + TN) / Total |
| **FAR (False Accept Rate)** | FP / (FP + TN) |
| **FRR (False Reject Rate)** | FN / (FN + TP) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |

---

## 5. Results

### 5.1 Inlier Distribution

#### SIFT Method
```
Genuine Matches:   avg=3.26, min=1, max=6
Impostor Matches:  avg=3.02, min=0, max=6
Separation Ratio:  1.08x
```

#### ORB Method
```
Genuine Matches:   avg=2.74, min=0, max=7
Impostor Matches:  avg=2.58, min=0, max=7
Separation Ratio:  1.06x
```

### 5.2 Performance at Different Thresholds

#### SIFT Results

| Threshold | TP | FN | TN | FP | Accuracy | FAR | FRR |
|-----------|----|----|----|----|----------|-----|-----|
| 1 | 50 | 0 | 8 | 92 | 38.67% | 92.00% | 0.00% |
| 2 | 43 | 7 | 21 | 79 | 42.67% | 79.00% | 14.00% |
| 3 | 36 | 14 | 33 | 67 | 46.00% | 67.00% | 28.00% |
| 4 | 28 | 22 | 53 | 47 | 54.00% | 47.00% | 44.00% |
| **5** | **5** | **45** | **85** | **15** | **60.00%** | **15.00%** | **90.00%** |
| 7 | 0 | 50 | 100 | 0 | 66.67% | 0.00% | 100.00% |

#### ORB Results

| Threshold | TP | FN | TN | FP | Accuracy | FAR | FRR |
|-----------|----|----|----|----|----------|-----|-----|
| 1 | 39 | 11 | 21 | 79 | 40.00% | 79.00% | 22.00% |
| 2 | 36 | 14 | 38 | 62 | 49.33% | 62.00% | 28.00% |
| 3 | 27 | 23 | 43 | 57 | 46.67% | 57.00% | 46.00% |
| 4 | 20 | 30 | 60 | 40 | 53.33% | 40.00% | 60.00% |
| **5** | **10** | **40** | **85** | **15** | **63.33%** | **15.00%** | **80.00%** |
| 7 | 1 | 49 | 98 | 2 | 66.00% | 2.00% | 98.00% |

### 5.3 Confusion Matrix (SIFT at Threshold=4)

This threshold provides the best trade-off between FAR and FRR:

```
                      Predicted
                 Match      No Match
              ┌──────────┬──────────┐
Actual Match  │    28    │    22    │
              ├──────────┼──────────┤
Actual No     │    47    │    53    │
Match         └──────────┴──────────┘

Accuracy:  54.00%
Precision: 37.33%
Recall:    56.00%
F1-Score:  44.80%
```

### 5.4 Key Observations

1. **Low Discrimination:** Genuine and impostor inlier counts overlap significantly (both averaging ~3 inliers)

2. **Trade-off Challenge:** 
   - Low threshold → High recall but many false positives
   - High threshold → Low false positives but rejects almost all genuine pairs

3. **Best Operating Point:** Threshold=4-5 provides ~54-60% accuracy with balanced FAR/FRR

4. **Method Comparison:** SIFT slightly outperforms ORB (3.26 vs 2.74 avg genuine inliers)

---

## 6. Analysis: Why Results Are Limited

### 6.1 Domain Gap Problem

The fundamental challenge is the **domain gap** between capture modalities:

| Factor | Impact |
|--------|--------|
| **Elastic Deformation** | Contact sensors flatten the finger; photos preserve 3D curvature |
| **Ridge Appearance** | Sensors capture ridges as dark lines; photos show ridges with highlights/shadows |
| **Scale Variation** | Camera distance varies; sensors have fixed capture area |
| **Perspective** | Photos have angle distortion; sensors are orthographic |

### 6.2 Feature Matching Limitations

SIFT/ORB are designed for **general image matching**, not fingerprints:

- Features are detected on arbitrary image structures, not specifically ridges
- Fingerprint-specific features (minutiae: ridge endings, bifurcations) are ignored
- No domain adaptation to bridge contactless↔contact appearance gap

### 6.3 Comparison with State-of-the-Art

| Approach | Typical Accuracy | Notes |
|----------|-----------------|-------|
| Our SIFT/ORB | ~54-60% | General feature matching |
| Traditional Minutiae | ~70-80% | Requires good segmentation |
| Deep Learning (Siamese) | ~90-95% | Requires training data |
| Lin & Kumar (2018) | ~95%+ | Domain-adaptive deep features |

---

## 7. Recommendations for Improvement

### 7.1 Short-Term Improvements

1. **Better Preprocessing:**
   - Fingerprint segmentation (remove background)
   - Ridge orientation estimation
   - Frequency-based enhancement

2. **Minutiae-Based Matching:**
   - Extract ridge endings and bifurcations
   - Use MCC (Minutia Cylinder Code) descriptors
   - Geometric alignment before matching

### 7.2 Long-Term Improvements

1. **Deep Learning Approaches:**
   - Siamese networks for similarity learning
   - Domain adaptation (contactless → contact)
   - Triplet loss for discriminative embeddings

2. **Reference Implementation:**
   - The original paper (Lin & Kumar, 2018) uses learned descriptors
   - Consider fine-tuning on PolyU dataset

---

## 8. Conclusion

Our experiments demonstrate that **traditional feature matching (SIFT/ORB) is insufficient** for cross-sensor fingerprint matching between contactless photos and contact-based scans. 

The key findings are:
- Maximum accuracy achieved: **~66%** (essentially rejecting all matches)
- Balanced accuracy at threshold=4: **~54%** 
- Genuine/impostor separation ratio: **~1.08x** (very low)

For production use, this approach would need to be replaced with:
1. Minutiae-based matching with specialized extractors
2. Deep learning models trained for cross-sensor matching
3. Domain adaptation techniques

The implemented Streamlit demo serves as an **educational tool** to visualize the matching process and understand the challenges of cross-sensor fingerprint interoperability.

---

## Appendix A: Code Files

| File | Description |
|------|-------------|
| `preprocessing.py` | Image enhancement pipeline |
| `matcher.py` | SIFT/ORB feature matching |
| `app.py` | Streamlit web demo |
| `run_experiments.py` | Experiment script |
| `experiment_results.json` | Raw results data |

## Appendix B: How to Reproduce

```bash
# Clone repository
git clone https://github.com/viswarup/fingerprint-matcher-track-C.git
cd fingerprint-matcher-track-C

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run experiments (requires PolyU dataset in parent folder)
python run_experiments.py

# Launch demo
streamlit run app.py
```

---

*Report generated from experiments on PolyU Contactless 2D to Contact-based 2D Fingerprint Database*
