# src/ppg_preprocessor.py (CONCISE VERSION)
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, detrend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from physionet_loader import PhysioNetLoader, PATHS

class PPGPreprocessor:
    """Complete PPG signal preprocessor with minimal output"""
    
    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
        self.filtered_signals = {}
        self.normalized_signals = {}
        self.quality_scores = {}
        self.peaks_data = {}
        self.processed_metadata = {}
        
    def apply_bandpass_filter(self, ppg_signal: np.ndarray, 
                             lowcut: float = 0.5, highcut: float = 8.0) -> np.ndarray:
        """Apply bandpass filter to remove noise and baseline drift"""
        if len(ppg_signal) < 100:
            return ppg_signal
            
        # Check for NaN or infinite values first
        if not np.all(np.isfinite(ppg_signal)):
            finite_mask = np.isfinite(ppg_signal)
            if not np.any(finite_mask):
                return np.zeros_like(ppg_signal)
            
            # Simple interpolation for missing values
            ppg_signal = np.interp(
                np.arange(len(ppg_signal)),
                np.where(finite_mask)[0],
                ppg_signal[finite_mask]
            )
            
        try:
            nyquist = 0.5 * self.sampling_rate
            low = lowcut / nyquist
            high = highcut / nyquist
            
            if low >= 1.0 or high >= 1.0 or low <= 0 or high <= low:
                return ppg_signal
            
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, ppg_signal)
            
            if not np.all(np.isfinite(filtered_signal)):
                return ppg_signal
            
            return filtered_signal
            
        except:
            return ppg_signal
    
    def remove_artifacts(self, ppg_signal: np.ndarray, 
                        threshold_factor: float = 3.0) -> np.ndarray:
        """Remove motion artifacts and outliers from PPG signal"""
        if len(ppg_signal) < 10:
            return ppg_signal
        
        if np.all(ppg_signal == ppg_signal[0]):
            return ppg_signal
            
        mean_val = np.mean(ppg_signal)
        std_val = np.std(ppg_signal)
        
        if std_val > 0:
            z_scores = np.abs((ppg_signal - mean_val) / std_val)
            artifact_mask = z_scores > threshold_factor
            
            clean_signal = ppg_signal.copy()
            if np.any(artifact_mask):
                artifact_indices = np.where(artifact_mask)[0]
                for idx in artifact_indices:
                    left_idx = max(0, idx - 1)
                    right_idx = min(len(ppg_signal) - 1, idx + 1)
                    clean_signal[idx] = (clean_signal[left_idx] + clean_signal[right_idx]) / 2
            
            return clean_signal
        
        return ppg_signal
    
    def normalize_signal(self, ppg_signal: np.ndarray, 
                        method: str = 'minmax') -> np.ndarray:
        """Fixed normalization with NaN prevention"""
        if len(ppg_signal) == 0:
            return ppg_signal
        
        if not np.all(np.isfinite(ppg_signal)):
            finite_mask = np.isfinite(ppg_signal)
            if not np.any(finite_mask):
                return np.zeros_like(ppg_signal)
        
        if method == 'minmax':
            min_val = np.min(ppg_signal)
            max_val = np.max(ppg_signal)
            
            # Check for zero range
            if max_val == min_val:
                if min_val == 0:
                    return np.zeros_like(ppg_signal)
                else:
                    return np.full_like(ppg_signal, 0.5)
            
            # Safe normalization
            range_val = max_val - min_val
            if range_val > 1e-10:
                normalized = (ppg_signal - min_val) / range_val
                
                if not np.all(np.isfinite(normalized)):
                    return np.full_like(ppg_signal, 0.5)
                
                return normalized
            else:
                return np.full_like(ppg_signal, 0.5)
                
        elif method == 'zscore':
            mean_val = np.mean(ppg_signal)
            std_val = np.std(ppg_signal)
            
            if std_val > 1e-10:
                normalized = (ppg_signal - mean_val) / std_val
                if np.all(np.isfinite(normalized)):
                    return normalized
            
            return ppg_signal - mean_val
            
        else:
            return ppg_signal
    
    def assess_signal_quality(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Assess PPG signal quality with NaN handling"""
        if len(ppg_signal) < 50 or not np.all(np.isfinite(ppg_signal)):
            return {'overall_quality': 0.0, 'snr': 0.0, 'perfusion': 0.0, 'consistency': 0.0}
        
        if np.all(ppg_signal == ppg_signal[0]):
            return {'overall_quality': 0.0, 'snr': 0.0, 'perfusion': 0.0, 'consistency': 0.0}
        
        quality_metrics = {}
        
        try:
            # SNR calculation
            signal_power = np.var(ppg_signal)
            if signal_power > 1e-10:
                try:
                    high_freq = self.apply_bandpass_filter(ppg_signal, lowcut=10.0, highcut=20.0)
                    noise_power = np.var(high_freq)
                    
                    if noise_power > 1e-12:
                        snr_db = 10 * np.log10(signal_power / noise_power)
                    else:
                        snr_db = 50.0
                except:
                    snr_db = 20.0
            else:
                snr_db = 0.0
            
            quality_metrics['snr'] = max(0, min(snr_db / 20.0, 1.0))
            
            # Perfusion Index
            ac_component = np.std(ppg_signal)
            dc_component = np.abs(np.mean(ppg_signal))
            
            if dc_component > 1e-10:
                perfusion = ac_component / dc_component
            else:
                perfusion = ac_component
                
            quality_metrics['perfusion'] = min(perfusion * 10, 1.0)
            
            # Signal consistency
            if len(ppg_signal) > 250:
                try:
                    autocorr = np.correlate(ppg_signal, ppg_signal, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    expected_lag = int(self.sampling_rate * 0.8)
                    if len(autocorr) > expected_lag:
                        consistency = np.max(autocorr[expected_lag:expected_lag+50]) / np.max(autocorr[:10])
                    else:
                        consistency = 0.5
                except:
                    consistency = 0.3
            else:
                consistency = 0.3
                
            quality_metrics['consistency'] = max(0, min(consistency, 1.0))
            
            # Overall quality score
            overall_quality = (
                0.4 * quality_metrics['snr'] + 
                0.3 * quality_metrics['perfusion'] + 
                0.3 * quality_metrics['consistency']
            )
            
            quality_metrics['overall_quality'] = overall_quality
            
        except:
            quality_metrics = {'overall_quality': 0.0, 'snr': 0.0, 'perfusion': 0.0, 'consistency': 0.0}
        
        return quality_metrics
    
    def detect_peaks(self, ppg_signal: np.ndarray, 
                    min_distance: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Detect systolic peaks with NaN handling"""
        if len(ppg_signal) < 100 or not np.all(np.isfinite(ppg_signal)):
            return {'peak_indices': np.array([]), 'peak_values': np.array([]), 
                   'rr_intervals': np.array([]), 'heart_rate': 0.0}
        
        if np.all(ppg_signal == ppg_signal[0]):
            return {'peak_indices': np.array([]), 'peak_values': np.array([]), 
                   'rr_intervals': np.array([]), 'heart_rate': 0.0}
            
        if min_distance is None:
            min_distance = int(self.sampling_rate * 0.4)
        
        try:
            signal_std = np.std(ppg_signal)
            signal_mean = np.mean(ppg_signal)
            
            if signal_std < 1e-10:
                return {'peak_indices': np.array([]), 'peak_values': np.array([]), 
                       'rr_intervals': np.array([]), 'heart_rate': 0.0}
            
            height_threshold = signal_mean + 0.3 * signal_std
            
            peak_indices, _ = find_peaks(
                ppg_signal,
                height=height_threshold,
                distance=min_distance,
                prominence=signal_std * 0.1
            )
            
            if len(peak_indices) > 0:
                peak_values = ppg_signal[peak_indices]
                
                if len(peak_indices) > 1:
                    rr_intervals = np.diff(peak_indices) / self.sampling_rate
                    heart_rate = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0
                else:
                    rr_intervals = np.array([])
                    heart_rate = 0.0
            else:
                peak_values = np.array([])
                rr_intervals = np.array([])
                heart_rate = 0.0
            
            return {
                'peak_indices': peak_indices,
                'peak_values': peak_values,
                'rr_intervals': rr_intervals,
                'heart_rate': heart_rate
            }
            
        except:
            return {'peak_indices': np.array([]), 'peak_values': np.array([]), 
                   'rr_intervals': np.array([]), 'heart_rate': 0.0}
    
    def preprocess_single_signal(self, ppg_signal: np.ndarray, 
                            record_name: str) -> Dict[str, any]:
        """Complete preprocessing pipeline with minimal logging"""
        results = {'record_name': record_name, 'processing_success': False}
        
        try:
            # Complete preprocessing pipeline
            filtered_signal = self.apply_bandpass_filter(ppg_signal)
            clean_signal = self.remove_artifacts(filtered_signal)
            normalized_signal = self.normalize_signal(clean_signal, method='minmax')
            
            # Verify normalization success
            if not np.all(np.isfinite(normalized_signal)):
                raise ValueError("Normalization failed - contains NaN/inf")
            
            quality_metrics = self.assess_signal_quality(normalized_signal)
            peaks_data = self.detect_peaks(normalized_signal)
            
            # Store results
            results.update({
                'original_signal': ppg_signal,
                'filtered_signal': filtered_signal,
                'clean_signal': clean_signal,
                'normalized_signal': normalized_signal,
                'quality_metrics': quality_metrics,
                'peaks_data': peaks_data,
                'processing_success': True
            })
            
        except Exception as e:
            print(f"{record_name} ✗ ERROR: {str(e)}")
            results['processing_success'] = False
        
        return results

    
    def preprocess_all_signals(self, signals_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Preprocess all PPG signals showing each signal name + errors only"""
        print(f"Processing {len(signals_dict)} PPG signals...")
        print("="*50)
        
        preprocessed_data = {}
        successful_count = 0
        error_count = 0
        
        for i, (record_name, signal) in enumerate(signals_dict.items()):
            try:
                result = self.preprocess_single_signal(signal, record_name)
                
                if result['processing_success']:
                    preprocessed_data[record_name] = result
                    successful_count += 1
                    
                    # Store in class attributes
                    self.filtered_signals[record_name] = result['filtered_signal']
                    self.normalized_signals[record_name] = result['normalized_signal']
                    self.quality_scores[record_name] = result['quality_metrics']['overall_quality']
                    self.peaks_data[record_name] = result['peaks_data']
                    
                    # Show signal name with success indicator
                    print(f"{record_name} ✓")
                else:
                    error_count += 1
                    # Error already printed in preprocess_single_signal
                    
            except Exception as e:
                print(f"{record_name} ✗ CRITICAL ERROR: {str(e)}")
                error_count += 1
                continue
        
        print("="*50)
        print(f"COMPLETED: {successful_count} success, {error_count} errors")
        return preprocessed_data

    
    def get_quality_summary(self) -> pd.DataFrame:
        """Get summary of signal quality metrics"""
        if not self.quality_scores:
            return pd.DataFrame()
        
        quality_data = []
        
        for record_name, quality_score in self.quality_scores.items():
            peaks_info = self.peaks_data.get(record_name, {})
            
            quality_data.append({
                'Record': record_name,
                'Quality Score': f"{quality_score:.3f}",
                'Quality Grade': 'Good' if quality_score > 0.7 else 'Fair' if quality_score > 0.4 else 'Poor',
                'Heart Rate (BPM)': f"{peaks_info.get('heart_rate', 0):.1f}",
                'Peaks Detected': len(peaks_info.get('peak_indices', []))
            })
        
        return pd.DataFrame(quality_data)
    
    def plot_preprocessing_example(self, record_name: str = None):
        """Plot preprocessing results for a sample signal"""
        if not self.filtered_signals:
            return
        
        if record_name is None:
            record_name = list(self.filtered_signals.keys())[0]
        
        if record_name not in self.filtered_signals:
            return
        
        # Get processing stages
        original = self.filtered_signals[record_name]
        normalized = self.normalized_signals[record_name]
        peaks_data = self.peaks_data[record_name]
        
        # Create time axis
        time = np.arange(len(normalized)) / self.sampling_rate
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Filtered signal
        axes[0].plot(time, original, 'b-', alpha=0.7, label='Filtered Signal', linewidth=1)
        axes[0].set_title(f'PPG Signal Processing: {record_name}')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Normalized with peaks
        axes[1].plot(time, normalized, 'g-', label='Normalized Signal', linewidth=1)
        
        if len(peaks_data.get('peak_indices', [])) > 0:
            peak_times = peaks_data['peak_indices'] / self.sampling_rate
            peak_values = peaks_data['peak_values']
            axes[1].plot(peak_times, peak_values, 'ro', markersize=4, 
                        label=f'Peaks (HR: {peaks_data.get("heart_rate", 0):.1f} BPM)')
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized Amplitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Limit to first 30 seconds
        max_time = min(30, time[-1])
        for ax in axes:
            ax.set_xlim(0, max_time)
        
        plt.tight_layout()
        plt.show()
    
    def save_preprocessed_data(self):
        """Save preprocessed data with verification"""
        if not self.filtered_signals:
            print("No preprocessed data to save.")
            return
        
        output_path = PATHS.PROCESSED_DATA_DIR / "preprocessed_ppg_signals.npz"
        
        # Prepare data for saving with NaN checks
        save_dict = {}
        saved_count = 0
        
        for record_name, signal in self.normalized_signals.items():
            # Verify signal is clean before saving
            if np.all(np.isfinite(signal)) and len(signal) > 0:
                save_dict[f"normalized_{record_name}"] = signal
                saved_count += 1
        
        # Save metadata
        valid_records = [name for name in self.quality_scores.keys() 
                        if f"normalized_{name}" in save_dict]
        
        save_dict['record_names'] = np.array(valid_records)
        save_dict['quality_scores'] = np.array([self.quality_scores[name] for name in valid_records])
        save_dict['heart_rates'] = np.array([self.peaks_data[name].get('heart_rate', 0) for name in valid_records])
        
        # Save
        np.savez_compressed(output_path, **save_dict)
        
        print(f"Saved {saved_count} clean signals to preprocessed_ppg_signals.npz")

# Main execution function
def run_preprocessing():
    """Main function to run PPG preprocessing pipeline"""
    print("=== PPG Signal Preprocessing Pipeline ===")
    
    # Load the raw PPG data
    loader = PhysioNetLoader()
    ppg_records = loader.find_ppg_records()
    
    if not ppg_records:
        print("ERROR: No PPG records found")
        return None
    
    signals, metadata = loader.load_all_ppg_signals()
    
    # Initialize preprocessor
    preprocessor = PPGPreprocessor(sampling_rate=250)
    
    # Preprocess all signals
    preprocessed_data = preprocessor.preprocess_all_signals(signals)
    
    # Show quality summary
    quality_summary = preprocessor.get_quality_summary()
    print("\nQuality Summary:")
    good_signals = sum(1 for score in preprocessor.quality_scores.values() if score > 0.7)
    fair_signals = sum(1 for score in preprocessor.quality_scores.values() if 0.4 < score <= 0.7)
    poor_signals = sum(1 for score in preprocessor.quality_scores.values() if score <= 0.4)
    
    print(f"High quality (>0.7): {good_signals}")
    print(f"Fair quality (0.4-0.7): {fair_signals}")
    print(f"Poor quality (<0.4): {poor_signals}")
    
    # Save results
    preprocessor.save_preprocessed_data()
    
    # Generate example plot
    if len(preprocessor.normalized_signals) > 0:
        print("Generating example plot...")
        preprocessor.plot_preprocessing_example()
    
    print("=== Preprocessing Complete ===")
    
    return preprocessor

if __name__ == "__main__":
    run_preprocessing()
