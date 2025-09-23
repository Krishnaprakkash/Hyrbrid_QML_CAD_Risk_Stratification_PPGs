# src/visualize_ppg.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from physionet_loader import PhysioNetLoader

def plot_ppg_overview():
    """Create comprehensive PPG visualization"""
    print("📊 Loading PPG data for visualization...")
    
    # Load the data
    loader = PhysioNetLoader()
    ppg_records = loader.find_ppg_records()
    
    if not ppg_records:
        print("❌ No PPG records found")
        return
        
    signals, metadata = loader.load_all_ppg_signals()
    
    # Plot multiple signals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    signal_names = list(signals.keys())[:4]  # First 4 signals
    
    print(f"🎨 Plotting {len(signal_names)} sample signals...")
    
    for i, name in enumerate(signal_names):
        signal = signals[name]
        meta = metadata[name]
        
        # Time axis
        time = np.arange(len(signal)) / meta['sampling_rate']
        
        # Plot
        axes[i].plot(time, signal, 'b-', linewidth=0.8)
        axes[i].set_title(f'{name} - {meta["channel_name"]}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('PPG Amplitude')
        axes[i].grid(True, alpha=0.3)
        
        # Limit to first 20 seconds for clarity
        max_time = min(20, time[-1])
        axes[i].set_xlim(0, max_time)
    
    plt.suptitle('PhysioNet PPG Signals Overview', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_signal_statistics():
    """Plot statistics of all PPG signals"""
    print("📈 Creating statistical plots...")
    
    loader = PhysioNetLoader()
    loader.find_ppg_records()
    signals, metadata = loader.load_all_ppg_signals()
    
    # Collect statistics
    durations = [meta['duration'] for meta in metadata.values()]
    signal_lengths = [len(sig) for sig in signals.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Duration histogram
    axes[0].hist(durations, bins=20, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Signal Duration (seconds)')
    axes[0].set_ylabel('Number of Signals')
    axes[0].set_title('PPG Signal Duration Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Length histogram  
    axes[1].hist(signal_lengths, bins=20, alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('Signal Length (samples)')
    axes[1].set_ylabel('Number of Signals')
    axes[1].set_title('PPG Signal Length Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🎨 PhysioNet PPG Data Visualizer")
    print("=" * 40)
    
    try:
        plot_ppg_overview()
        plot_signal_statistics()
        print("✅ Visualization complete!")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
