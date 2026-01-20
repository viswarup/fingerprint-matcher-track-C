"""
Cross-Sensor Fingerprint Matcher - Streamlit Demo Application

A web-based demo for matching contactless fingerprint photos with
contact-based sensor scans using SIFT/ORB feature matching.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os

# Import our custom modules
from preprocessing import FingerprintPreprocessor, preprocess_from_bytes
from matcher import FingerprintMatcher, match_from_bytes

# Page configuration
st.set_page_config(
    page_title="Cross-Sensor Fingerprint Matcher",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .failure-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_dataset_images():
    """Get available images from the PolyU dataset."""
    base_path = Path(__file__).parent.parent
    
    contactless_path = base_path / "processed_contactless_2d_fingerprint_images" / "first_session"
    contact_path = base_path / "contact-based_fingerprints" / "first_session"
    
    contactless_images = {}
    contact_images = {}
    
    # Get contactless images
    if contactless_path.exists():
        for folder in sorted(contactless_path.iterdir()):
            if folder.is_dir() and folder.name.startswith('p'):
                finger_id = folder.name[1:]  # Remove 'p' prefix
                images = list(folder.glob('*.bmp'))
                if images:
                    contactless_images[finger_id] = images
    
    # Get contact images
    if contact_path.exists():
        for img_file in sorted(contact_path.glob('*.jpg')):
            parts = img_file.stem.split('_')
            if len(parts) == 2:
                finger_id = parts[0]
                if finger_id not in contact_images:
                    contact_images[finger_id] = []
                contact_images[finger_id].append(img_file)
    
    return contactless_images, contact_images


def load_image_from_path(path):
    """Load an image from file path."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def image_to_display(img):
    """Convert OpenCV image to format suitable for Streamlit display."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    # Header
    st.markdown('<p class="main-header">🔐 Cross-Sensor Fingerprint Matcher</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Match contactless fingerprint photos with contact-based sensor scans using SIFT/ORB feature matching</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Matching algorithm
        method = st.selectbox(
            "Matching Algorithm",
            ["SIFT", "ORB", "Both (Compare)"],
            help="SIFT is more accurate but slower. ORB is faster but less robust."
        )
        method_key = method.lower().split()[0]
        
        # Enhancement level
        enhance_level = st.select_slider(
            "Enhancement Level",
            options=["light", "standard", "heavy"],
            value="standard",
            help="Higher enhancement may help with low-quality images but adds processing time."
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            ratio_threshold = st.slider(
                "Lowe's Ratio Threshold",
                min_value=0.5,
                max_value=0.9,
                value=0.75,
                step=0.05,
                help="Lower values are stricter (fewer but better matches)"
            )
            
            use_homography = st.checkbox(
                "Use Homography Filtering",
                value=True,
                help="Filter matches using geometric consistency"
            )
            
            match_threshold = st.slider(
                "Match Decision Threshold",
                min_value=5,
                max_value=50,
                value=15,
                help="Minimum inlier count to consider a match"
            )
        
        st.divider()
        
        # Input source selection
        st.header("📁 Input Source")
        
        # Check if PolyU dataset exists
        base_path = Path(__file__).parent.parent
        dataset_exists = (base_path / "processed_contactless_2d_fingerprint_images").exists()
        
        if dataset_exists:
            input_options = ["Upload Images", "Use Sample Images", "PolyU Dataset Browser"]
        else:
            input_options = ["Upload Images", "Use Sample Images"]
        
        input_source = st.radio(
            "Choose input source:",
            input_options
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🖼️ Match Fingerprints", "📊 Preprocessing Pipeline", "ℹ️ About"])
    
    with tab1:
        img1_raw = None
        img2_raw = None
        
        if input_source == "Upload Images":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📱 Contactless Source")
                file1 = st.file_uploader(
                    "Upload contactless fingerprint photo",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    key="contactless"
                )
                if file1:
                    img1_bytes = file1.read()
                    nparr = np.frombuffer(img1_bytes, np.uint8)
                    img1_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    file1.seek(0)  # Reset file pointer
                    st.image(image_to_display(img1_raw), caption="Uploaded contactless image", use_container_width=True)
            
            with col2:
                st.subheader("👆 Contact Target")
                file2 = st.file_uploader(
                    "Upload contact-based fingerprint scan",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    key="contact"
                )
                if file2:
                    img2_bytes = file2.read()
                    nparr = np.frombuffer(img2_bytes, np.uint8)
                    img2_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    file2.seek(0)
                    st.image(image_to_display(img2_raw), caption="Uploaded contact image", use_container_width=True)
        
        elif input_source == "Use Sample Images":
            # Load included sample images
            samples_path = Path(__file__).parent / "samples"
            
            st.info("📁 Using included sample fingerprint images for demonstration.")
            
            # Get available samples
            contactless_samples = list(samples_path.glob("contactless_*.bmp")) + list(samples_path.glob("contactless_*.png"))
            contact_samples = list(samples_path.glob("contact_*.jpg")) + list(samples_path.glob("contact_*.png"))
            
            if contactless_samples and contact_samples:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📱 Contactless Sample")
                    sample_names1 = [p.name for p in contactless_samples]
                    selected_sample1 = st.selectbox("Select contactless sample:", sample_names1, key="sample_cl")
                    sample_path1 = contactless_samples[sample_names1.index(selected_sample1)]
                    img1_raw = load_image_from_path(sample_path1)
                    st.image(image_to_display(img1_raw), caption=selected_sample1, use_container_width=True)
                
                with col2:
                    st.subheader("👆 Contact Sample")
                    sample_names2 = [p.name for p in contact_samples]
                    selected_sample2 = st.selectbox("Select contact sample:", sample_names2, key="sample_ct")
                    sample_path2 = contact_samples[sample_names2.index(selected_sample2)]
                    img2_raw = load_image_from_path(sample_path2)
                    st.image(image_to_display(img2_raw), caption=selected_sample2, use_container_width=True)
                
                st.caption("💡 Tip: Sample 1 pairs (contactless_sample1 + contact_sample1) are from the same finger.")
            else:
                st.error("❌ Sample images not found in the 'samples' folder. Please use the Upload option.")
        
        elif input_source == "PolyU Dataset Browser":  # Dataset Browser
            contactless_images, contact_images = get_dataset_images()
            
            if not contactless_images or not contact_images:
                st.warning("⚠️ PolyU dataset not found in the expected location. Please use the upload option.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📱 Contactless Source")
                    
                    # Select finger ID
                    finger_ids = sorted(contactless_images.keys(), key=lambda x: int(x))
                    selected_finger = st.selectbox(
                        "Select Finger ID",
                        finger_ids,
                        key="finger_select"
                    )
                    
                    # Select sample
                    samples = contactless_images.get(selected_finger, [])
                    sample_names = [p.name for p in samples]
                    selected_sample1 = st.selectbox(
                        "Select Sample",
                        sample_names,
                        key="sample1_select"
                    )
                    
                    if selected_sample1:
                        sample_path = samples[sample_names.index(selected_sample1)]
                        img1_raw = load_image_from_path(sample_path)
                        st.image(image_to_display(img1_raw), caption=f"Finger {selected_finger} - {selected_sample1}", use_container_width=True)
                
                with col2:
                    st.subheader("👆 Contact Target")
                    
                    # Option to match same or different finger
                    match_same = st.checkbox("Match with same finger", value=True)
                    
                    if match_same:
                        contact_finger = selected_finger
                    else:
                        contact_finger_ids = sorted(contact_images.keys(), key=lambda x: int(x))
                        contact_finger = st.selectbox(
                            "Select Contact Finger ID",
                            contact_finger_ids,
                            key="contact_finger_select"
                        )
                    
                    contact_samples = contact_images.get(contact_finger, [])
                    contact_sample_names = [p.name for p in contact_samples]
                    
                    if contact_sample_names:
                        selected_sample2 = st.selectbox(
                            "Select Contact Sample",
                            contact_sample_names,
                            key="sample2_select"
                        )
                        
                        if selected_sample2:
                            sample_path2 = contact_samples[contact_sample_names.index(selected_sample2)]
                            img2_raw = load_image_from_path(sample_path2)
                            st.image(image_to_display(img2_raw), caption=f"Finger {contact_finger} - {selected_sample2}", use_container_width=True)
                    else:
                        st.warning(f"No contact images found for finger {contact_finger}")
        
        st.divider()
        
        # Matching button
        if img1_raw is not None and img2_raw is not None:
            if st.button("🚀 Run Matching Algorithm", type="primary", use_container_width=True):
                with st.spinner("Processing images and extracting features..."):
                    # Initialize preprocessor and matcher
                    preprocessor = FingerprintPreprocessor()
                    matcher = FingerprintMatcher(method=method_key, ratio_threshold=ratio_threshold)
                    
                    # Preprocess images
                    img1_processed = preprocessor.get_for_matching(img1_raw, 'contactless', enhance_level)
                    img2_processed = preprocessor.get_for_matching(img2_raw, 'contact', enhance_level)
                    
                    # Run matching
                    result = matcher.match_images(img1_processed, img2_processed, method_key, use_homography)
                    
                    # Get statistics
                    if 'best' in result:
                        stats = result['best']['stats']
                        display_result = result['best']
                    else:
                        stats = result['stats']
                        display_result = result
                    
                    is_match = stats['inlier_count'] >= match_threshold
                    
                    # Display results
                    st.divider()
                    st.header("📊 Matching Results")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Keypoints Found", stats['keypoint_count'])
                    with col2:
                        st.metric("Good Matches", stats['match_count'])
                    with col3:
                        st.metric("Inlier Matches", stats['inlier_count'])
                    with col4:
                        st.metric("Confidence", stats['confidence'])
                    
                    # Match decision
                    st.divider()
                    if is_match:
                        st.markdown(
                            '<div class="success-box">✅ IDENTITY VERIFIED - Fingerprints Match!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="failure-box">❌ IDENTITY DENIED - No Match Found</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Show enhanced images
                    st.divider()
                    st.subheader("🔍 Enhanced Images")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img1_processed, caption="Enhanced Contactless", use_container_width=True)
                    with col2:
                        st.image(img2_processed, caption="Enhanced Contact", use_container_width=True)
                    
                    # Show match visualization
                    st.divider()
                    st.subheader("🔗 Keypoint Correspondence Map")
                    match_img = matcher.draw_matches(img1_processed, img2_processed, display_result)
                    st.image(image_to_display(match_img), caption="Feature matches between images", use_container_width=True)
                    
                    # Show both methods if 'both' was selected
                    if method_key == 'both':
                        st.divider()
                        st.subheader("📈 Method Comparison")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**SIFT Results**")
                            sift_stats = result['sift']['stats']
                            st.write(f"- Keypoints: {sift_stats['keypoint_count']}")
                            st.write(f"- Good Matches: {sift_stats['match_count']}")
                            st.write(f"- Inliers: {sift_stats['inlier_count']}")
                            st.write(f"- Confidence: {sift_stats['confidence']}")
                        
                        with col2:
                            st.markdown("**ORB Results**")
                            orb_stats = result['orb']['stats']
                            st.write(f"- Keypoints: {orb_stats['keypoint_count']}")
                            st.write(f"- Good Matches: {orb_stats['match_count']}")
                            st.write(f"- Inliers: {orb_stats['inlier_count']}")
                            st.write(f"- Confidence: {orb_stats['confidence']}")
        else:
            st.info("👆 Please upload or select two fingerprint images to begin matching.")
    
    with tab2:
        st.header("🔬 Preprocessing Pipeline Visualization")
        st.markdown("""
        This tab shows the step-by-step preprocessing applied to fingerprint images
        before feature extraction. Understanding these stages helps optimize matching performance.
        """)
        
        # Allow image upload for preprocessing demo
        demo_file = st.file_uploader(
            "Upload an image to see preprocessing stages",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="preprocess_demo"
        )
        
        if demo_file:
            nparr = np.frombuffer(demo_file.read(), np.uint8)
            demo_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            img_type = st.radio(
                "Image type:",
                ["contactless", "contact"],
                horizontal=True
            )
            
            preprocessor = FingerprintPreprocessor()
            
            if img_type == "contactless":
                stages = preprocessor.preprocess_contactless(demo_img, 'heavy')
            else:
                stages = preprocessor.preprocess_contact(demo_img, 'heavy')
            
            st.subheader("📸 Original Image")
            st.image(image_to_display(demo_img), use_container_width=True)
            
            st.subheader("🔄 Processing Stages")
            
            # Display each stage
            stage_names = {
                'grayscale': '1️⃣ Grayscale Conversion',
                'normalized': '2️⃣ Intensity Normalization',
                'enhanced': '3️⃣ CLAHE Enhancement',
                'filtered': '4️⃣ Noise Filtering',
                'gabor': '5️⃣ Gabor Ridge Enhancement',
                'binary': '6️⃣ Binarization',
                'skeleton': '7️⃣ Skeletonization'
            }
            
            cols = st.columns(3)
            for idx, (key, title) in enumerate(stage_names.items()):
                if key in stages:
                    with cols[idx % 3]:
                        st.markdown(f"**{title}**")
                        st.image(stages[key], use_container_width=True)
    
    with tab3:
        st.header("ℹ️ About This Demo")
        
        st.markdown("""
        ### 🎯 Purpose
        
        This application demonstrates **cross-sensor fingerprint matching** - the challenge of 
        matching fingerprints captured via contactless photography with traditional contact-based 
        sensor scans. This is a critical problem in biometric interoperability.
        
        ### 🔧 Technical Approach
        
        The solution uses a three-stage pipeline:
        
        1. **Pre-processing**: Enhance fingerprint ridge patterns using:
           - CLAHE (Contrast Limited Adaptive Histogram Equalization)
           - Gabor filtering for ridge enhancement
           - Bilateral filtering for noise reduction
        
        2. **Feature Extraction**: Detect and describe local features using:
           - **SIFT** (Scale-Invariant Feature Transform) - robust to scale/rotation changes
           - **ORB** (Oriented FAST and Rotated BRIEF) - faster, patent-free alternative
        
        3. **Matching**: Find correspondences between feature sets:
           - FLANN/Brute-force matching for nearest neighbors
           - Lowe's ratio test to filter ambiguous matches
           - Homography estimation with RANSAC for geometric consistency
        
        ### 📚 Dataset
        
        This demo is designed for the **PolyU Contactless 2D to Contact-based 2D Fingerprint 
        Images Database** from The Hong Kong Polytechnic University.
        
        **Reference:**
        > Chenhao Lin, Ajay Kumar, "Matching Contactless and Contact-based Conventional 
        > Fingerprint Images for Biometrics Identification," IEEE Transactions on Image 
        > Processing, vol. 27, pp. 2008-2021, April 2018.
        
        ### 🔑 Key Challenges
        
        - **Scale differences**: Contactless photos have varying distances to the finger
        - **Perspective distortion**: Photos lack the "flattening" of touch sensors
        - **Lighting variations**: Uneven illumination in photos vs. controlled sensors
        - **Resolution mismatch**: High-res photos vs. lower-res sensor scans
        
        ### 💡 Tips for Best Results
        
        1. Use **SIFT** for more accurate matching (slower)
        2. Use **ORB** for faster processing (less robust)
        3. Increase enhancement level for low-quality images
        4. Lower the ratio threshold for stricter matching
        5. Enable homography filtering to reduce false matches
        """)


if __name__ == "__main__":
    main()
