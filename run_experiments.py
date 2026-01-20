"""
Comprehensive Experiment Script for Cross-Sensor Fingerprint Matching

This script runs extensive experiments on the PolyU dataset to evaluate
the matching performance with detailed metrics and confusion matrix.
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import FingerprintPreprocessor
from matcher import FingerprintMatcher, match_fingerprints


def run_comprehensive_experiments():
    """Run comprehensive experiments on the PolyU dataset."""
    
    base_path = Path(__file__).parent.parent
    contactless_path = base_path / "processed_contactless_2d_fingerprint_images" / "first_session"
    contact_path = base_path / "contact-based_fingerprints" / "first_session"
    
    print("\n" + "=" * 80)
    print("📊 CROSS-SENSOR FINGERPRINT MATCHING - COMPREHENSIVE EXPERIMENTS")
    print("=" * 80)
    
    # Collect available fingers
    finger_ids = []
    for folder in contactless_path.iterdir():
        if folder.is_dir() and folder.name.startswith('p'):
            fid = folder.name[1:]
            # Check if corresponding contact image exists
            if (contact_path / f"{fid}_1.jpg").exists():
                finger_ids.append(fid)
    
    finger_ids = sorted(finger_ids, key=lambda x: int(x))
    print(f"\n📁 Found {len(finger_ids)} fingers with both contactless and contact images")
    
    # Select sample for experiments (50 fingers for reasonable runtime)
    sample_size = min(50, len(finger_ids))
    sample_fingers = finger_ids[:sample_size]
    print(f"📌 Using {sample_size} fingers for experiments")
    
    results = {
        'sift': {'genuine': [], 'impostor': []},
        'orb': {'genuine': [], 'impostor': []}
    }
    
    # ========================================
    # EXPERIMENT 1: GENUINE MATCHES
    # ========================================
    print("\n" + "-" * 80)
    print("🔬 EXPERIMENT 1: GENUINE MATCHES (Same finger, corresponding samples)")
    print("-" * 80)
    
    genuine_count = 0
    for finger_id in sample_fingers:
        cl_path = contactless_path / f"p{finger_id}" / "p1.bmp"
        ct_path = contact_path / f"{finger_id}_1.jpg"
        
        if cl_path.exists() and ct_path.exists():
            # Test with SIFT
            result_sift = match_fingerprints(str(cl_path), str(ct_path), 'sift', 'standard')
            results['sift']['genuine'].append({
                'finger_id': finger_id,
                'inliers': result_sift['stats']['inlier_count'],
                'score': result_sift['stats']['score'],
                'confidence': result_sift['stats']['confidence']
            })
            
            # Test with ORB
            result_orb = match_fingerprints(str(cl_path), str(ct_path), 'orb', 'standard')
            results['orb']['genuine'].append({
                'finger_id': finger_id,
                'inliers': result_orb['stats']['inlier_count'],
                'score': result_orb['stats']['score'],
                'confidence': result_orb['stats']['confidence']
            })
            
            genuine_count += 1
            if genuine_count % 10 == 0:
                print(f"  Processed {genuine_count}/{sample_size} genuine pairs...")
    
    print(f"  ✅ Completed {genuine_count} genuine match experiments")
    
    # ========================================
    # EXPERIMENT 2: IMPOSTOR MATCHES
    # ========================================
    print("\n" + "-" * 80)
    print("🔬 EXPERIMENT 2: IMPOSTOR MATCHES (Different fingers)")
    print("-" * 80)
    
    # Generate impostor pairs (different finger IDs)
    impostor_pairs = []
    for i, f1 in enumerate(sample_fingers):
        # Compare with 2 different random fingers
        other_fingers = [f for f in sample_fingers if f != f1]
        random.seed(42 + i)  # Reproducible
        selected = random.sample(other_fingers, min(2, len(other_fingers)))
        for f2 in selected:
            impostor_pairs.append((f1, f2))
    
    impostor_count = 0
    for f1, f2 in impostor_pairs:
        cl_path = contactless_path / f"p{f1}" / "p1.bmp"
        ct_path = contact_path / f"{f2}_1.jpg"
        
        if cl_path.exists() and ct_path.exists():
            # Test with SIFT
            result_sift = match_fingerprints(str(cl_path), str(ct_path), 'sift', 'standard')
            results['sift']['impostor'].append({
                'finger1': f1,
                'finger2': f2,
                'inliers': result_sift['stats']['inlier_count'],
                'score': result_sift['stats']['score'],
                'confidence': result_sift['stats']['confidence']
            })
            
            # Test with ORB
            result_orb = match_fingerprints(str(cl_path), str(ct_path), 'orb', 'standard')
            results['orb']['impostor'].append({
                'finger1': f1,
                'finger2': f2,
                'inliers': result_orb['stats']['inlier_count'],
                'score': result_orb['stats']['score'],
                'confidence': result_orb['stats']['confidence']
            })
            
            impostor_count += 1
            if impostor_count % 20 == 0:
                print(f"  Processed {impostor_count}/{len(impostor_pairs)} impostor pairs...")
    
    print(f"  ✅ Completed {impostor_count} impostor match experiments")
    
    # ========================================
    # ANALYZE RESULTS
    # ========================================
    print("\n" + "=" * 80)
    print("📈 RESULTS ANALYSIS")
    print("=" * 80)
    
    for method in ['sift', 'orb']:
        print(f"\n{'=' * 40}")
        print(f"METHOD: {method.upper()}")
        print(f"{'=' * 40}")
        
        genuine_inliers = [r['inliers'] for r in results[method]['genuine']]
        impostor_inliers = [r['inliers'] for r in results[method]['impostor']]
        
        avg_genuine = sum(genuine_inliers) / max(len(genuine_inliers), 1)
        avg_impostor = sum(impostor_inliers) / max(len(impostor_inliers), 1)
        
        print(f"\n📊 Inlier Statistics:")
        print(f"  Genuine matches:  avg={avg_genuine:.2f}, min={min(genuine_inliers)}, max={max(genuine_inliers)}")
        print(f"  Impostor matches: avg={avg_impostor:.2f}, min={min(impostor_inliers)}, max={max(impostor_inliers)}")
        
        # Test different thresholds
        print(f"\n📊 Performance at Different Thresholds:")
        print(f"  {'Threshold':<12} {'TP':<8} {'FN':<8} {'TN':<8} {'FP':<8} {'Accuracy':<10} {'FAR':<8} {'FRR':<8}")
        print(f"  {'-'*76}")
        
        best_threshold = 0
        best_accuracy = 0
        best_metrics = {}
        
        for threshold in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
            # True Positives: Genuine pairs correctly identified as match
            tp = sum(1 for i in genuine_inliers if i >= threshold)
            # False Negatives: Genuine pairs incorrectly rejected
            fn = len(genuine_inliers) - tp
            # True Negatives: Impostor pairs correctly rejected
            tn = sum(1 for i in impostor_inliers if i < threshold)
            # False Positives: Impostor pairs incorrectly accepted
            fp = len(impostor_inliers) - tn
            
            total = tp + fn + tn + fp
            accuracy = (tp + tn) / total * 100 if total > 0 else 0
            far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0  # False Accept Rate
            frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0  # False Reject Rate
            
            print(f"  {threshold:<12} {tp:<8} {fn:<8} {tn:<8} {fp:<8} {accuracy:<10.2f} {far:<8.2f} {frr:<8.2f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_metrics = {
                    'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
                    'accuracy': accuracy, 'far': far, 'frr': frr,
                    'precision': tp / (tp + fp) * 100 if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                }
        
        print(f"\n⭐ Best Threshold: {best_threshold} (Accuracy: {best_accuracy:.2f}%)")
        print(f"\n📊 Confusion Matrix at Best Threshold ({best_threshold}):")
        print(f"                    Predicted")
        print(f"                  Match    No Match")
        print(f"  Actual Match     {best_metrics['tp']:<8} {best_metrics['fn']:<8}")
        print(f"  Actual No Match  {best_metrics['fp']:<8} {best_metrics['tn']:<8}")
        
        print(f"\n📊 Key Metrics at Best Threshold:")
        print(f"  Accuracy:  {best_metrics['accuracy']:.2f}%")
        print(f"  Precision: {best_metrics['precision']:.2f}%")
        print(f"  Recall:    {best_metrics['recall']:.2f}%")
        print(f"  FAR:       {best_metrics['far']:.2f}%")
        print(f"  FRR:       {best_metrics['frr']:.2f}%")
        
        # Store for report
        results[method]['best_threshold'] = best_threshold
        results[method]['best_metrics'] = best_metrics
        results[method]['avg_genuine'] = avg_genuine
        results[method]['avg_impostor'] = avg_impostor
    
    # Save results to JSON
    output_path = Path(__file__).parent / "experiment_results.json"
    with open(output_path, 'w') as f:
        # Convert to serializable format
        serializable = {
            'sift': {
                'genuine_count': len(results['sift']['genuine']),
                'impostor_count': len(results['sift']['impostor']),
                'avg_genuine_inliers': results['sift']['avg_genuine'],
                'avg_impostor_inliers': results['sift']['avg_impostor'],
                'best_threshold': results['sift']['best_threshold'],
                'best_metrics': results['sift']['best_metrics']
            },
            'orb': {
                'genuine_count': len(results['orb']['genuine']),
                'impostor_count': len(results['orb']['impostor']),
                'avg_genuine_inliers': results['orb']['avg_genuine'],
                'avg_impostor_inliers': results['orb']['avg_impostor'],
                'best_threshold': results['orb']['best_threshold'],
                'best_metrics': results['orb']['best_metrics']
            }
        }
        json.dump(serializable, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n" + "=" * 80)
    print("Experiments complete!")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    run_comprehensive_experiments()
