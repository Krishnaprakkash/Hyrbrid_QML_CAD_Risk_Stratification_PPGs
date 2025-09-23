import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class ProjectPaths:
    """Centralized path management for portability"""
    
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).parent.parent  # Goes up from src/ to project root
        
        # All paths relative to project root
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.PHYSIONET_DIR = self.RAW_DATA_DIR / "physionet_2015"
        self.PHYSIONET_TRAINING = self.PHYSIONET_DIR / "training"
        
        # Create directories if they don't exist
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories"""
        for path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR]:
            path.mkdir(parents=True, exist_ok=True)

# Global path instance
PATHS = ProjectPaths()

class PhysioNetLoader:
    """Load and parse PhysioNet Challenge 2015 dataset with PPG signals"""
    
    def __init__(self, data_path: Optional[Path] = None):
        # Use centralized path configuration
        self.data_path = data_path or PATHS.PHYSIONET_TRAINING
        self.ppg_records = []
        self.sampling_rate = 250  # PhysioNet standard
        self.ppg_signals = {}
        self.labels = {}
        
        print(f"📂 PhysioNet data path: {self.data_path.relative_to(PATHS.PROJECT_ROOT)}")
        
    def find_ppg_records(self) -> List[str]:
        """Find all records that contain PPG signals"""
        print("🔍 Scanning for PPG records in PhysioNet dataset...")
        
        if not self.data_path.exists():
            print(f"⚠️  Data directory not found: {self.data_path.relative_to(PATHS.PROJECT_ROOT)}")
            print("   Please extract training.zip to this location")
            return []
        
        # Look for .hea files (headers)
        hea_files = list(self.data_path.glob("*.hea"))
        ppg_records = []
        
        print(f"   Found {len(hea_files)} total records, checking for PPG...")
        
        for hea_file in hea_files:
            try:
                record_name = hea_file.stem
                header = wfdb.rdheader(str(hea_file.with_suffix('')))
                
                # Check if PPG signal exists in this record
                signal_names = [sig.lower() for sig in header.sig_name]
                ppg_indices = []
                
                for i, name in enumerate(signal_names):
                    if any(keyword in name for keyword in ['ppg', 'pleth', 'photopleth']):
                        ppg_indices.append(i)
                
                if ppg_indices:
                    ppg_records.append({
                        'record_name': record_name,
                        'ppg_channels': ppg_indices,
                        'channel_names': [header.sig_name[i] for i in ppg_indices]
                    })
                    
            except Exception as e:
                continue
                
        print(f"✅ Found {len(ppg_records)} records with PPG signals")
        self.ppg_records = ppg_records
        return ppg_records
    
    def load_ppg_signal(self, record_info: Dict) -> Tuple[np.ndarray, Dict]:
        """Load PPG signal from a single record"""
        record_name = record_info['record_name']
        ppg_channels = record_info['ppg_channels']
        
        try:
            # Load the complete record
            record_path = str(self.data_path / record_name)
            record = wfdb.rdrecord(record_path)
            
            # Extract PPG channel(s)
            ppg_data = record.p_signal[:, ppg_channels[0]]  # Use first PPG channel
            
            # Get metadata
            metadata = {
                'record_name': record_name,
                'sampling_rate': record.fs,
                'duration': len(ppg_data) / record.fs,
                'channel_name': record_info['channel_names'][0],
                'units': record.units[ppg_channels[0]] if record.units else 'unknown'
            }
            
            return ppg_data, metadata
            
        except Exception as e:
            print(f"⚠️  Error loading {record_name}: {e}")
            return None, None
    
    def load_all_ppg_signals(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Load all PPG signals from the dataset"""
        if not self.ppg_records:
            self.find_ppg_records()
        
        print(f"📊 Loading {len(self.ppg_records)} PPG signals...")
        
        ppg_signals = {}
        metadata = {}
        
        for i, record_info in enumerate(self.ppg_records):
            record_name = record_info['record_name']
            
            signal, meta = self.load_ppg_signal(record_info)
            
            if signal is not None:
                ppg_signals[record_name] = signal
                metadata[record_name] = meta
                
                if (i + 1) % 50 == 0:
                    print(f"   Loaded {i + 1}/{len(self.ppg_records)} signals...")
        
        print(f"✅ Successfully loaded {len(ppg_signals)} PPG signals")
        
        self.ppg_signals = ppg_signals
        self.metadata = metadata
        
        return ppg_signals, metadata
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get summary statistics of the loaded dataset"""
        if not self.ppg_signals:
            print("No signals loaded. Run load_all_ppg_signals() first.")
            return None
        
        summary_data = []
        
        for record_name, signal in self.ppg_signals.items():
            meta = self.metadata[record_name]
            
            summary_data.append({
                'Record': record_name,
                'Duration (s)': f"{meta['duration']:.1f}",
                'Signal Length': len(signal),
                'Sampling Rate': meta['sampling_rate'],
                'Channel': meta['channel_name'],
                'Mean': f"{np.mean(signal):.3f}",
                'Std': f"{np.std(signal):.3f}",
                'Min': f"{np.min(signal):.3f}",
                'Max': f"{np.max(signal):.3f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def save_processed_data(self):
        """Save processed signals for future use"""
        if not self.ppg_signals:
            print("No signals to save. Load signals first.")
            return
        
        output_path = PATHS.PROCESSED_DATA_DIR / "physionet_ppg_signals.npz"
        
        # Save signals individually (they have different lengths)
        save_dict = {}
        
        # Add each signal with its record name as key
        for record_name, signal in self.ppg_signals.items():
            save_dict[f"signal_{record_name}"] = signal
        
        # Add metadata as separate entries
        save_dict['signal_names'] = list(self.ppg_signals.keys())
        
        # Save metadata info
        metadata_arrays = {}
        for record_name, meta in self.metadata.items():
            for key, value in meta.items():
                if key not in metadata_arrays:
                    metadata_arrays[key] = []
                metadata_arrays[key].append(value)
        
        # Add metadata arrays
        for key, values in metadata_arrays.items():
            save_dict[f"meta_{key}"] = np.array(values, dtype=object)
        
        # Save everything
        np.savez_compressed(output_path, **save_dict)
        
        print(f"💾 Saved {len(self.ppg_signals)} signals to: {output_path.relative_to(PATHS.PROJECT_ROOT)}")


if __name__ == "__main__":
    print("🧪 Testing PhysioNet Loader...")
    loader = PhysioNetLoader()
    
    # Find PPG records
    ppg_records = loader.find_ppg_records()
    
    if ppg_records:
        # Load all signals
        signals, metadata = loader.load_all_ppg_signals()
        
        # Show summary
        summary = loader.get_dataset_summary()
        print("\n📋 Dataset Summary:")
        print(summary.head(10))
        
        # Save processed data
        loader.save_processed_data()
        
        print(f"\n✅ Successfully processed {len(signals)} PPG signals!")
    else:
        print("❌ No PPG records found. Check dataset extraction.")

