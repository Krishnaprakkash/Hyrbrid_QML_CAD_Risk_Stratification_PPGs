import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_view_processed_data():
    """Load and view the saved processed data"""
    
    # Load the saved data
    data_path = Path("data/processed/physionet_ppg_signals.npz")
    
    if not data_path.exists():
        print("No processed data found. Run physionet_loader.py first.")
        return
    
    # Load the data
    data = np.load(data_path, allow_pickle=True)
    
    # Show what's in the file
    print("📁 Loaded processed data:")
    print(f"   Keys: {list(data.keys())}")
    
    # Get signal names
    signal_names = data['signal_names']
    print(f"\n📊 Found {len(signal_names)} signals:")
    
    # Show details of first few signals
    for i, name in enumerate(signal_names[:5]):
        signal = data[f'signal_{name}']
        print(f"  {name}: {len(signal)} samples")
        if i == 0:
            print(f"    Sample values: {signal[:10]}")

if __name__ == "__main__":
    load_and_view_processed_data()
