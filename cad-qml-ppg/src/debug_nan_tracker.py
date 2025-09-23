# src/debug_nan_tracker.py
import numpy as np
import pandas as pd
from pathlib import Path

def debug_nan_sources():
    """Debug where NaN values are being introduced in preprocessing"""
    
    # Load the preprocessed data file
    from physionet_loader import PATHS
    preprocessed_path = PATHS.PROCESSED_DATA_DIR / "preprocessed_ppg_signals.npz"
    
    if not preprocessed_path.exists():
        print("❌ No preprocessed data found")
        return
    
    print("🔍 Debugging NaN sources in preprocessed data...")
    data = np.load(preprocessed_path, allow_pickle=True)
    
    record_names = data['record_names']
    
    nan_analysis = []
    
    for i, record_name in enumerate(record_names):
        try:
            # Check each signal
            signal_key = f'normalized_{record_name}'
            
            if signal_key in data:
                signal = data[signal_key]
                
                # Analyze NaN presence
                nan_count = np.sum(~np.isfinite(signal))
                total_count = len(signal)
                nan_percentage = (nan_count / total_count) * 100 if total_count > 0 else 0
                
                # Check signal statistics
                if total_count > 0:
                    has_nan = nan_count > 0
                    all_same_value = len(np.unique(signal[np.isfinite(signal)])) <= 1
                    min_val = np.min(signal[np.isfinite(signal)]) if np.any(np.isfinite(signal)) else np.nan
                    max_val = np.max(signal[np.isfinite(signal)]) if np.any(np.isfinite(signal)) else np.nan
                    
                    nan_analysis.append({
                        'record': record_name,
                        'total_samples': total_count,
                        'nan_count': nan_count,
                        'nan_percentage': round(nan_percentage, 2),
                        'has_nan': has_nan,
                        'all_same_value': all_same_value,
                        'min_val': min_val,
                        'max_val': max_val,
                        'range': max_val - min_val if np.isfinite(min_val) and np.isfinite(max_val) else np.nan
                    })
        
        except Exception as e:
            print(f"Error analyzing {record_name}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(nan_analysis)
    
    print(f"\n📊 NaN Analysis Results:")
    print(f"   Total records analyzed: {len(df)}")
    print(f"   Records with NaN: {df['has_nan'].sum()}")
    print(f"   Records with all same values: {df['all_same_value'].sum()}")
    print(f"   Records with zero range: {df['range'].isna().sum()}")
    
    # Show problematic records
    problematic = df[df['has_nan'] | df['all_same_value'] | df['range'].isna()]
    
    if len(problematic) > 0:
        print(f"\n⚠️  Problematic records:")
        print(problematic[['record', 'nan_count', 'nan_percentage', 'all_same_value', 'range']])
        
        # Save detailed analysis
        problematic.to_csv(PATHS.PROCESSED_DATA_DIR / "nan_analysis.csv", index=False)
        print(f"\n💾 Detailed analysis saved to: nan_analysis.csv")
    
    return df

if __name__ == "__main__":
    debug_nan_sources()
