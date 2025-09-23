# src/debug_preprocessing_steps.py
import numpy as np
from physionet_loader import PhysioNetLoader
from ppg_preprocessor import PPGPreprocessor

def debug_preprocessing_pipeline():
    """Step-by-step debugging of preprocessing pipeline"""
    
    print("🔍 Debugging preprocessing pipeline step by step...")
    
    # Load original data
    loader = PhysioNetLoader()
    signals, metadata = loader.load_all_ppg_signals()
    
    # Test on a few problematic records
    test_records = ['a397l', 'a645l', 'b753l']  # From your error log
    
    preprocessor = PPGPreprocessor()
    
    for record_name in test_records:
        if record_name in signals:
            print(f"\n📋 Debugging {record_name}:")
            original_signal = signals[record_name]
            
            # Step 1: Check original signal
            print(f"   Original signal - Length: {len(original_signal)}")
            print(f"   Original NaN count: {np.sum(~np.isfinite(original_signal))}")
            print(f"   Original range: [{np.min(original_signal[np.isfinite(original_signal)]):.4f}, {np.max(original_signal[np.isfinite(original_signal)]):.4f}]")
            
            # Step 2: After bandpass filter
            try:
                filtered = preprocessor.apply_bandpass_filter(original_signal)
                print(f"   After filtering - NaN count: {np.sum(~np.isfinite(filtered))}")
                print(f"   After filtering - Range: [{np.min(filtered[np.isfinite(filtered)]):.4f}, {np.max(filtered[np.isfinite(filtered)]):.4f}]")
            except Exception as e:
                print(f"   ❌ Filter error: {e}")
                continue
            
            # Step 3: After artifact removal
            try:
                clean = preprocessor.remove_artifacts(filtered)
                print(f"   After artifact removal - NaN count: {np.sum(~np.isfinite(clean))}")
                print(f"   After artifact removal - Range: [{np.min(clean[np.isfinite(clean)]):.4f}, {np.max(clean[np.isfinite(clean)]):.4f}]")
            except Exception as e:
                print(f"   ❌ Artifact removal error: {e}")
                continue
            
            # Step 4: After normalization
            try:
                normalized = preprocessor.normalize_signal(clean)
                print(f"   After normalization - NaN count: {np.sum(~np.isfinite(normalized))}")
                print(f"   After normalization - Range: [{np.min(normalized[np.isfinite(normalized)]):.4f}, {np.max(normalized[np.isfinite(normalized)]):.4f}]")
                
                # Check for problematic normalization
                if np.all(clean[np.isfinite(clean)] == clean[np.isfinite(clean)][0]):
                    print("   ⚠️  Signal has all identical values - normalization will create NaN!")
                
            except Exception as e:
                print(f"   ❌ Normalization error: {e}")

if __name__ == "__main__":
    debug_preprocessing_pipeline()
