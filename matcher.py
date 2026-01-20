"""
Fingerprint Feature Matching Module

This module provides SIFT and ORB-based feature matching for fingerprint images,
optimized for cross-sensor matching between contactless photos and contact scans.
"""

import cv2
import numpy as np
from preprocessing import FingerprintPreprocessor, preprocess_from_bytes


class FingerprintMatcher:
    """
    A class to match fingerprint images using feature-based methods.
    
    Supports SIFT, ORB, and hybrid matching approaches.
    """
    
    def __init__(self, method='sift', ratio_threshold=0.8):
        """
        Initialize the matcher with configurable parameters.
        
        Args:
            method: 'sift', 'orb', or 'both'
            ratio_threshold: Lowe's ratio test threshold (lower = stricter)
        """
        self.method = method.lower()
        self.ratio_threshold = ratio_threshold
        self.preprocessor = FingerprintPreprocessor()
        
        # Initialize feature detectors with more features for better matching
        self.sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=15)
        self.orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.1, nlevels=12)
        
        # Initialize matchers
        # FLANN parameters for SIFT (float descriptors)
        self.flann_sift = cv2.FlannBasedMatcher(
            {'algorithm': 1, 'trees': 5},  # FLANN_INDEX_KDTREE
            {'checks': 50}
        )
        
        # FLANN parameters for ORB (binary descriptors)
        self.flann_orb = cv2.FlannBasedMatcher(
            {'algorithm': 6, 'table_number': 6, 'key_size': 12, 'multi_probe_level': 1},
            {'checks': 50}
        )
        
        # Brute-force matchers
        self.bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def extract_sift_features(self, image):
        """
        Extract SIFT keypoints and descriptors from an image.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def extract_orb_features(self, image):
        """
        Extract ORB keypoints and descriptors from an image.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_with_ratio_test(self, des1, des2, matcher, k=2):
        """
        Match descriptors using Lowe's ratio test.
        
        The ratio test filters out ambiguous matches by comparing
        the distance to the best match vs. second-best match.
        
        Args:
            des1: Descriptors from first image
            des2: Descriptors from second image
            matcher: OpenCV matcher to use
            k: Number of nearest neighbors
            
        Returns:
            List of good matches that pass the ratio test
        """
        if des1 is None or des2 is None:
            return []
        
        if len(des1) < k or len(des2) < k:
            return []
        
        try:
            matches = matcher.knnMatch(des1, des2, k=k)
        except cv2.error:
            # Fallback to brute-force if FLANN fails
            return []
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_homography_inliers(self, kp1, kp2, good_matches, min_matches=4):
        """
        Compute homography and count inlier matches.
        
        Uses RANSAC to find geometrically consistent matches,
        filtering out outliers that don't fit the transformation.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            good_matches: List of matches after ratio test
            min_matches: Minimum matches needed for homography
            
        Returns:
            Tuple of (inlier_matches, homography_matrix, mask)
        """
        if len(good_matches) < min_matches:
            return good_matches, None, None
        
        # Extract match point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is None:
            return good_matches, None, None
        
        # Filter matches by inlier mask
        inlier_matches = [m for m, inlier in zip(good_matches, mask.ravel()) if inlier]
        
        return inlier_matches, H, mask
    
    def calculate_match_score(self, good_matches, kp1, kp2, inlier_matches=None):
        """
        Calculate a normalized match score.
        
        Args:
            good_matches: Matches after ratio test
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            inlier_matches: Matches after homography filtering (optional)
            
        Returns:
            Dictionary with match statistics and scores
        """
        n_keypoints = min(len(kp1), len(kp2))
        n_good_matches = len(good_matches)
        n_inliers = len(inlier_matches) if inlier_matches else n_good_matches
        
        if n_keypoints == 0:
            return {
                'match_count': 0,
                'inlier_count': 0,
                'keypoint_count': 0,
                'match_ratio': 0.0,
                'inlier_ratio': 0.0,
                'score': 0.0,
                'confidence': 'Very Low'
            }
        
        match_ratio = n_good_matches / n_keypoints * 100
        inlier_ratio = n_inliers / max(n_good_matches, 1) * 100 if n_good_matches > 0 else 0
        
        # Combined score considering both match count and inlier ratio
        score = (n_inliers * 0.7 + match_ratio * 0.3)
        
        # Determine confidence level
        if n_inliers >= 50 and inlier_ratio >= 60:
            confidence = 'Very High'
        elif n_inliers >= 30 and inlier_ratio >= 50:
            confidence = 'High'
        elif n_inliers >= 15 and inlier_ratio >= 40:
            confidence = 'Medium'
        elif n_inliers >= 5:
            confidence = 'Low'
        else:
            confidence = 'Very Low'
        
        return {
            'match_count': n_good_matches,
            'inlier_count': n_inliers,
            'keypoint_count': n_keypoints,
            'match_ratio': round(match_ratio, 2),
            'inlier_ratio': round(inlier_ratio, 2),
            'score': round(score, 2),
            'confidence': confidence
        }
    
    def match_sift(self, img1, img2, use_homography=True):
        """
        Match two images using SIFT features.
        
        Args:
            img1: First preprocessed image
            img2: Second preprocessed image
            use_homography: Whether to filter with homography
            
        Returns:
            Dictionary with matching results
        """
        # Extract SIFT features
        kp1, des1 = self.extract_sift_features(img1)
        kp2, des2 = self.extract_sift_features(img2)
        
        # Match with ratio test
        good_matches = self.match_with_ratio_test(des1, des2, self.bf_sift)
        
        # Filter with homography if requested
        inlier_matches = good_matches
        homography = None
        if use_homography and len(good_matches) >= 4:
            inlier_matches, homography, _ = self.compute_homography_inliers(
                kp1, kp2, good_matches
            )
        
        # Calculate score
        stats = self.calculate_match_score(good_matches, kp1, kp2, inlier_matches)
        
        return {
            'method': 'SIFT',
            'keypoints1': kp1,
            'keypoints2': kp2,
            'good_matches': good_matches,
            'inlier_matches': inlier_matches,
            'homography': homography,
            'stats': stats
        }
    
    def match_orb(self, img1, img2, use_homography=True):
        """
        Match two images using ORB features.
        
        Args:
            img1: First preprocessed image
            img2: Second preprocessed image
            use_homography: Whether to filter with homography
            
        Returns:
            Dictionary with matching results
        """
        # Extract ORB features
        kp1, des1 = self.extract_orb_features(img1)
        kp2, des2 = self.extract_orb_features(img2)
        
        # Match with ratio test
        good_matches = self.match_with_ratio_test(des1, des2, self.bf_orb)
        
        # Filter with homography if requested
        inlier_matches = good_matches
        homography = None
        if use_homography and len(good_matches) >= 4:
            inlier_matches, homography, _ = self.compute_homography_inliers(
                kp1, kp2, good_matches
            )
        
        # Calculate score
        stats = self.calculate_match_score(good_matches, kp1, kp2, inlier_matches)
        
        return {
            'method': 'ORB',
            'keypoints1': kp1,
            'keypoints2': kp2,
            'good_matches': good_matches,
            'inlier_matches': inlier_matches,
            'homography': homography,
            'stats': stats
        }
    
    def match_images(self, img1, img2, method=None, use_homography=True):
        """
        Match two preprocessed fingerprint images.
        
        Args:
            img1: First preprocessed image (contactless)
            img2: Second preprocessed image (contact)
            method: 'sift', 'orb', or None (use default)
            use_homography: Whether to use homography filtering
            
        Returns:
            Dictionary with matching results
        """
        method = method or self.method
        
        if method == 'sift':
            return self.match_sift(img1, img2, use_homography)
        elif method == 'orb':
            return self.match_orb(img1, img2, use_homography)
        elif method == 'both':
            sift_result = self.match_sift(img1, img2, use_homography)
            orb_result = self.match_orb(img1, img2, use_homography)
            
            # Choose the result with more inliers
            if sift_result['stats']['inlier_count'] >= orb_result['stats']['inlier_count']:
                best = sift_result
            else:
                best = orb_result
            
            return {
                'sift': sift_result,
                'orb': orb_result,
                'best': best
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def draw_matches(self, img1, img2, result, draw_all=False):
        """
        Draw matches between two images for visualization.
        
        Args:
            img1: First image
            img2: Second image
            result: Matching result dictionary
            draw_all: If True, draw all good matches; if False, only inliers
            
        Returns:
            Image with drawn matches
        """
        # Handle 'both' method result
        if 'best' in result:
            result = result['best']
        
        matches_to_draw = result['good_matches'] if draw_all else result['inlier_matches']
        
        # Convert grayscale to BGR for colored visualization
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()
        
        if len(img2.shape) == 2:
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = img2.copy()
        
        # Draw matches
        match_img = cv2.drawMatches(
            img1_color, result['keypoints1'],
            img2_color, result['keypoints2'],
            matches_to_draw, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_img
    
    def is_match(self, stats, threshold=15):
        """
        Determine if the match result indicates a genuine match.
        
        Args:
            stats: Match statistics dictionary
            threshold: Minimum inlier count for a match
            
        Returns:
            Boolean indicating if the fingerprints match
        """
        return stats['inlier_count'] >= threshold and stats['confidence'] in ['High', 'Very High']


def match_fingerprints(img1_path, img2_path, method='sift', enhance_level='standard'):
    """
    Convenience function to match two fingerprint images from file paths.
    
    Args:
        img1_path: Path to first image (contactless)
        img2_path: Path to second image (contact)
        method: 'sift', 'orb', or 'both'
        enhance_level: 'light', 'standard', or 'heavy'
        
    Returns:
        Dictionary with matching results and statistics
    """
    # Initialize components
    preprocessor = FingerprintPreprocessor()
    matcher = FingerprintMatcher(method=method)
    
    # Load and preprocess images
    img1_raw = preprocessor.load_image(img1_path)
    img2_raw = preprocessor.load_image(img2_path)
    
    img1 = preprocessor.get_for_matching(img1_raw, 'contactless', enhance_level)
    img2 = preprocessor.get_for_matching(img2_raw, 'contact', enhance_level)
    
    # Match
    result = matcher.match_images(img1, img2, method)
    
    # Get stats
    if 'best' in result:
        stats = result['best']['stats']
    else:
        stats = result['stats']
    
    return {
        'result': result,
        'stats': stats,
        'is_match': matcher.is_match(stats),
        'img1_preprocessed': img1,
        'img2_preprocessed': img2
    }


def match_from_bytes(img1_bytes, img2_bytes, method='sift', enhance_level='standard'):
    """
    Match fingerprints from image bytes (for Streamlit uploads).
    
    Args:
        img1_bytes: Bytes of first image (contactless)
        img2_bytes: Bytes of second image (contact)
        method: 'sift', 'orb', or 'both'
        enhance_level: 'light', 'standard', or 'heavy'
        
    Returns:
        Dictionary with matching results
    """
    # Preprocess from bytes
    img1 = preprocess_from_bytes(img1_bytes, 'contactless', enhance_level)
    img2 = preprocess_from_bytes(img2_bytes, 'contact', enhance_level)
    
    # Match
    matcher = FingerprintMatcher(method=method)
    result = matcher.match_images(img1, img2, method)
    
    # Get stats
    if 'best' in result:
        stats = result['best']['stats']
    else:
        stats = result['stats']
    
    return {
        'result': result,
        'stats': stats,
        'is_match': matcher.is_match(stats),
        'img1_preprocessed': img1,
        'img2_preprocessed': img2
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) > 3 else 'sift'
        
        print(f"Matching: {img1_path} vs {img2_path}")
        print(f"Method: {method}")
        print("-" * 50)
        
        result = match_fingerprints(img1_path, img2_path, method)
        
        stats = result['stats']
        print(f"Keypoints found: {stats['keypoint_count']}")
        print(f"Good matches: {stats['match_count']}")
        print(f"Inlier matches: {stats['inlier_count']}")
        print(f"Match ratio: {stats['match_ratio']}%")
        print(f"Inlier ratio: {stats['inlier_ratio']}%")
        print(f"Score: {stats['score']}")
        print(f"Confidence: {stats['confidence']}")
        print(f"Is Match: {result['is_match']}")
        
        # Save visualization
        matcher = FingerprintMatcher()
        match_img = matcher.draw_matches(
            result['img1_preprocessed'],
            result['img2_preprocessed'],
            result['result']
        )
        cv2.imwrite("match_result.png", match_img)
        print("\nVisualization saved to: match_result.png")
    else:
        print("Usage: python matcher.py <image1_path> <image2_path> [method]")
        print("  method: sift (default), orb, or both")
