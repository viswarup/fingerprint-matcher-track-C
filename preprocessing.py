"""
Fingerprint Image Preprocessing Module

This module provides comprehensive preprocessing functions for fingerprint images,
designed to bridge the gap between contactless photos and contact-based sensor scans.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte


class FingerprintPreprocessor:
    """
    A class to preprocess fingerprint images for feature matching.
    
    Supports multiple enhancement techniques optimized for cross-sensor matching.
    """
    
    def __init__(self, clahe_clip_limit=3.0, clahe_tile_size=(8, 8)):
        """
        Initialize the preprocessor with configurable parameters.
        
        Args:
            clahe_clip_limit: Contrast limiting threshold for CLAHE
            clahe_tile_size: Size of grid for histogram equalization
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, 
            tileGridSize=clahe_tile_size
        )
    
    def load_image(self, image_path):
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image in BGR format, or None if loading fails
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def to_grayscale(self, image):
        """
        Convert image to grayscale if needed.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    def normalize_intensity(self, image):
        """
        Normalize image intensity to full 0-255 range.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Intensity-normalized image
        """
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def apply_clahe(self, image):
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        CLAHE enhances local contrast while limiting noise amplification,
        making ridge patterns more visible in fingerprint images.
        
        Args:
            image: Grayscale input image
            
        Returns:
            CLAHE-enhanced image
        """
        return self.clahe.apply(image)
    
    def apply_bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter for edge-preserving smoothing.
        
        Reduces noise while preserving ridge edges, crucial for 
        maintaining fingerprint minutiae detail.
        
        Args:
            image: Grayscale input image
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Bilaterally filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def apply_gabor_filter(self, image, orientations=8, frequency=0.1):
        """
        Apply Gabor filter bank to enhance ridge patterns.
        
        Gabor filters are particularly effective for fingerprint 
        ridge enhancement as they respond to specific orientations.
        
        Args:
            image: Grayscale input image
            orientations: Number of filter orientations
            frequency: Spatial frequency of the filter
            
        Returns:
            Gabor-enhanced image (maximum response across orientations)
        """
        filtered = np.zeros_like(image, dtype=np.float32)
        
        for theta_idx in range(orientations):
            theta = theta_idx * np.pi / orientations
            kernel = cv2.getGaborKernel(
                (21, 21), 
                sigma=4.0, 
                theta=theta, 
                lambd=1/frequency, 
                gamma=0.5, 
                psi=0
            )
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            filtered = np.maximum(filtered, response)
        
        # Normalize to 0-255 range
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        return filtered.astype(np.uint8)
    
    def apply_gaussian_blur(self, image, kernel_size=5):
        """
        Apply Gaussian blur for noise reduction.
        
        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel (must be odd)
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_adaptive_threshold(self, image, block_size=11, c=2):
        """
        Apply adaptive thresholding for binarization.
        
        Uses local mean for threshold calculation, handling uneven 
        illumination common in contactless fingerprint photos.
        
        Args:
            image: Grayscale input image
            block_size: Size of neighborhood for threshold calculation
            c: Constant subtracted from the mean
            
        Returns:
            Binary image
        """
        return cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size, c
        )
    
    def apply_otsu_threshold(self, image):
        """
        Apply Otsu's automatic thresholding for binarization.
        
        Automatically determines optimal threshold that minimizes 
        intra-class variance.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Binary image
        """
        _, binary = cv2.threshold(
            image, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary
    
    def apply_morphological_ops(self, image, operation='close', kernel_size=3):
        """
        Apply morphological operations to clean up binary images.
        
        Args:
            image: Binary input image
            operation: 'open', 'close', 'erode', 'dilate'
            kernel_size: Size of the structuring element
            
        Returns:
            Morphologically processed image
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        if operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'erode':
            return cv2.erode(image, kernel)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def apply_skeletonization(self, binary_image):
        """
        Apply morphological skeletonization to thin ridges.
        
        Reduces ridge thickness to 1 pixel, making minutiae 
        (ridge endings and bifurcations) easier to detect.
        
        Args:
            binary_image: Binary input image (ridges = white)
            
        Returns:
            Skeletonized image
        """
        # Ensure binary format (0 and 1)
        binary = (binary_image > 127).astype(np.uint8)
        
        # Apply skeletonization
        skeleton = skeletonize(binary).astype(np.uint8) * 255
        
        return skeleton
    
    def resize_image(self, image, target_size=None, scale_factor=None):
        """
        Resize image to target size or by scale factor.
        
        Args:
            image: Input image
            target_size: Tuple (width, height) for target dimensions
            scale_factor: Float to scale image by
            
        Returns:
            Resized image
        """
        if target_size:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        elif scale_factor:
            return cv2.resize(
                image, None, 
                fx=scale_factor, fy=scale_factor, 
                interpolation=cv2.INTER_AREA
            )
        return image
    
    def preprocess_contactless(self, image, enhance_level='standard'):
        """
        Full preprocessing pipeline for contactless fingerprint photos.
        
        Args:
            image: Input BGR or grayscale image
            enhance_level: 'light', 'standard', or 'heavy'
            
        Returns:
            Dictionary with preprocessing stages:
                - grayscale: Grayscale conversion
                - normalized: Intensity normalized
                - enhanced: CLAHE enhanced
                - filtered: Noise filtered
                - binary: Binarized (optional)
                - skeleton: Skeletonized (optional)
        """
        result = {}
        
        # Convert to grayscale
        gray = self.to_grayscale(image)
        result['grayscale'] = gray
        
        # Normalize intensity
        normalized = self.normalize_intensity(gray)
        result['normalized'] = normalized
        
        # Apply CLAHE for contrast enhancement
        enhanced = self.apply_clahe(normalized)
        result['enhanced'] = enhanced
        
        # Apply bilateral filtering for noise reduction
        filtered = self.apply_bilateral_filter(enhanced)
        result['filtered'] = filtered
        
        if enhance_level in ['standard', 'heavy']:
            # Apply Gabor filtering for ridge enhancement
            gabor = self.apply_gabor_filter(filtered)
            result['gabor'] = gabor
        
        if enhance_level == 'heavy':
            # Binarization
            binary = self.apply_adaptive_threshold(filtered)
            binary = self.apply_morphological_ops(binary, 'close', 3)
            result['binary'] = binary
            
            # Skeletonization
            skeleton = self.apply_skeletonization(binary)
            result['skeleton'] = skeleton
        
        return result
    
    def preprocess_contact(self, image, enhance_level='standard'):
        """
        Full preprocessing pipeline for contact-based fingerprint scans.
        
        Contact-based images typically have better quality but may 
        need less aggressive enhancement.
        
        Args:
            image: Input BGR or grayscale image
            enhance_level: 'light', 'standard', or 'heavy'
            
        Returns:
            Dictionary with preprocessing stages
        """
        result = {}
        
        # Convert to grayscale
        gray = self.to_grayscale(image)
        result['grayscale'] = gray
        
        # Normalize intensity
        normalized = self.normalize_intensity(gray)
        result['normalized'] = normalized
        
        # Apply lighter CLAHE for contact images
        light_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = light_clahe.apply(normalized)
        result['enhanced'] = enhanced
        
        # Light Gaussian blur to reduce sensor noise
        filtered = self.apply_gaussian_blur(enhanced, 3)
        result['filtered'] = filtered
        
        if enhance_level in ['standard', 'heavy']:
            gabor = self.apply_gabor_filter(filtered, orientations=8)
            result['gabor'] = gabor
        
        if enhance_level == 'heavy':
            binary = self.apply_otsu_threshold(filtered)
            binary = self.apply_morphological_ops(binary, 'close', 2)
            result['binary'] = binary
            
            skeleton = self.apply_skeletonization(binary)
            result['skeleton'] = skeleton
        
        return result
    
    def get_for_matching(self, image, image_type='contactless', enhance_level='standard', 
                          target_size=(400, 400)):
        """
        Get the best preprocessed version of an image for feature matching.
        
        Args:
            image: Input image (BGR or grayscale)
            image_type: 'contactless' or 'contact'
            enhance_level: 'light', 'standard', or 'heavy'
            target_size: Target size for resolution normalization (width, height)
            
        Returns:
            Best preprocessed image for feature extraction
        """
        if image_type == 'contactless':
            stages = self.preprocess_contactless(image, enhance_level)
        else:
            stages = self.preprocess_contact(image, enhance_level)
        
        # Return the enhanced or gabor-filtered image for feature matching
        if 'gabor' in stages and enhance_level != 'light':
            result = stages['gabor']
        else:
            result = stages['enhanced']
        
        # Normalize resolution for cross-sensor matching
        if target_size:
            result = self.resize_image(result, target_size)
        
        return result


def preprocess_image(image_path, image_type='contactless', enhance_level='standard'):
    """
    Convenience function to preprocess an image from file.
    
    Args:
        image_path: Path to the image file
        image_type: 'contactless' or 'contact'
        enhance_level: 'light', 'standard', or 'heavy'
        
    Returns:
        Preprocessed image ready for feature matching
    """
    preprocessor = FingerprintPreprocessor()
    image = preprocessor.load_image(image_path)
    return preprocessor.get_for_matching(image, image_type, enhance_level)


def preprocess_from_bytes(image_bytes, image_type='contactless', enhance_level='standard'):
    """
    Preprocess an image from bytes (for Streamlit file uploads).
    
    Args:
        image_bytes: Raw image bytes
        image_type: 'contactless' or 'contact'
        enhance_level: 'light', 'standard', or 'heavy'
        
    Returns:
        Preprocessed image ready for feature matching
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image from bytes")
    
    preprocessor = FingerprintPreprocessor()
    return preprocessor.get_for_matching(image, image_type, enhance_level)


if __name__ == "__main__":
    # Test the preprocessing module
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        preprocessor = FingerprintPreprocessor()
        img = preprocessor.load_image(img_path)
        
        print(f"Image loaded: {img.shape}")
        
        stages = preprocessor.preprocess_contactless(img, 'heavy')
        
        for stage_name, stage_img in stages.items():
            print(f"  {stage_name}: {stage_img.shape}")
            cv2.imwrite(f"test_{stage_name}.png", stage_img)
        
        print("Preprocessing complete. Check output files.")
    else:
        print("Usage: python preprocessing.py <image_path>")
