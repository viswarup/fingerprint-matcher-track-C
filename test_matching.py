"""
Test Script: Cross-Sensor Fingerprint Matching Scenarios

This script tests the fingerprint matching pipeline on various scenarios:
1. Genuine matches (same finger, different samples)
2. Impostor matches (different fingers)
3. Multi-method comparison (SIFT vs ORB)
"""

import os
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import FingerprintPreprocessor
from matcher import FingerprintMatcher, match_fingerprints


def print_separator():
    print("=" * 70)


def print_result(label, result):
    stats = result['stats']
    print(f"\n{label}")
    print("-" * 50)
    print(f"  Keypoints found:  {stats['keypoint_count']}")
    print(f"  Good matches:     {stats['match_count']}")
    print(f"  Inlier matches:   {stats['inlier_count']}")
    print(f"  Match ratio:      {stats['match_ratio']}%")
    print(f"  Inlier ratio:     {stats['inlier_ratio']}%")
    print(f"  Score:            {stats['score']}")
    print(f"  Confidence:       {stats['confidence']}")
    print(f"  Is Match:         {'✅ YES' if result['is_match'] else '❌ NO'}")


def run_tests():
    base_path = Path(__file__).parent.parent
    
    contactless_path = base_path / "processed_contactless_2d_fingerprint_images" / "first_session"
    contact_path = base_path / "contact-based_fingerprints" / "first_session"
    
    print("\n" + "=" * 70)
    print("🔐 CROSS-SENSOR FINGERPRINT MATCHING - TEST SCENARIOS")
    print("=" * 70)
    
    # ==========================
    # TEST 1: Genuine Matches (Same Finger)
    # ==========================
    print_separator()
    print("\n📋 TEST 1: GENUINE MATCHES (Same finger, different samples)")
    print("   Expected: Should show higher match scores for same-finger pairs")
    print_separator()
    
    genuine_pairs = [
        ("p1", "p1.bmp", "1_1.jpg", "Finger 1: Sample 1 vs Sample 1"),
        ("p1", "p2.bmp", "1_2.jpg", "Finger 1: Sample 2 vs Sample 2"),
        ("p1", "p3.bmp", "1_3.jpg", "Finger 1: Sample 3 vs Sample 3"),
        ("p10", "p1.bmp", "10_1.jpg", "Finger 10: Sample 1 vs Sample 1"),
        ("p50", "p1.bmp", "50_1.jpg", "Finger 50: Sample 1 vs Sample 1"),
    ]
    
    genuine_scores = []
    for folder, cl_file, ct_file, label in genuine_pairs:
        cl_path = contactless_path / folder / cl_file
        ct_path = contact_path / ct_file
        
        if cl_path.exists() and ct_path.exists():
            result = match_fingerprints(str(cl_path), str(ct_path), 'sift', 'standard')
            genuine_scores.append(result['stats']['inlier_count'])
            print_result(label, result)
        else:
            print(f"\n⚠️ Files not found for: {label}")
    
    # ==========================
    # TEST 2: Impostor Matches (Different Fingers)
    # ==========================
    print_separator()
    print("\n📋 TEST 2: IMPOSTOR MATCHES (Different fingers)")
    print("   Expected: Should show lower match scores for different-finger pairs")
    print_separator()
    
    impostor_pairs = [
        ("p1", "p1.bmp", "2_1.jpg", "Finger 1 vs Finger 2"),
        ("p1", "p1.bmp", "10_1.jpg", "Finger 1 vs Finger 10"),
        ("p1", "p1.bmp", "50_1.jpg", "Finger 1 vs Finger 50"),
        ("p10", "p1.bmp", "20_1.jpg", "Finger 10 vs Finger 20"),
        ("p50", "p1.bmp", "100_1.jpg", "Finger 50 vs Finger 100"),
    ]
    
    impostor_scores = []
    for folder, cl_file, ct_file, label in impostor_pairs:
        cl_path = contactless_path / folder / cl_file
        ct_path = contact_path / ct_file
        
        if cl_path.exists() and ct_path.exists():
            result = match_fingerprints(str(cl_path), str(ct_path), 'sift', 'standard')
            impostor_scores.append(result['stats']['inlier_count'])
            print_result(label, result)
        else:
            print(f"\n⚠️ Files not found for: {label}")
    
    # ==========================
    # TEST 3: Method Comparison (SIFT vs ORB)
    # ==========================
    print_separator()
    print("\n📋 TEST 3: METHOD COMPARISON (SIFT vs ORB)")
    print("   Comparing matching methods on the same image pair")
    print_separator()
    
    test_cl = contactless_path / "p5" / "p1.bmp"
    test_ct = contact_path / "5_1.jpg"
    
    if test_cl.exists() and test_ct.exists():
        # SIFT
        result_sift = match_fingerprints(str(test_cl), str(test_ct), 'sift', 'standard')
        print_result("SIFT Method (Finger 5)", result_sift)
        
        # ORB
        result_orb = match_fingerprints(str(test_cl), str(test_ct), 'orb', 'standard')
        print_result("ORB Method (Finger 5)", result_orb)
    
    # ==========================
    # TEST 4: Enhancement Level Comparison
    # ==========================
    print_separator()
    print("\n📋 TEST 4: ENHANCEMENT LEVEL COMPARISON")
    print("   Testing different preprocessing intensities")
    print_separator()
    
    test_cl = contactless_path / "p3" / "p1.bmp"
    test_ct = contact_path / "3_1.jpg"
    
    if test_cl.exists() and test_ct.exists():
        for level in ['light', 'standard', 'heavy']:
            result = match_fingerprints(str(test_cl), str(test_ct), 'sift', level)
            print_result(f"Enhancement: {level.upper()} (Finger 3)", result)
    
    # ==========================
    # SUMMARY
    # ==========================
    print_separator()
    print("\n📊 SUMMARY")
    print_separator()
    
    if genuine_scores and impostor_scores:
        avg_genuine = sum(genuine_scores) / len(genuine_scores)
        avg_impostor = sum(impostor_scores) / len(impostor_scores)
        
        print(f"\n  Average Genuine Inliers:  {avg_genuine:.1f}")
        print(f"  Average Impostor Inliers: {avg_impostor:.1f}")
        print(f"  Separation Ratio:         {avg_genuine / max(avg_impostor, 0.1):.2f}x")
        
        if avg_genuine > avg_impostor:
            print("\n  ✅ System shows discriminative power (genuine > impostor)")
        else:
            print("\n  ⚠️ System may need tuning (low separation)")
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_tests()
