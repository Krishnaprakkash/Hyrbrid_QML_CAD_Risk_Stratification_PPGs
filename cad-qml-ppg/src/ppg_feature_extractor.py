# src/ppg_feature_extractor.py
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from physionet_loader import PATHS

class PPGFeatureExtractor:
    """Complete feature extraction for PPG signals - time, frequency, and statistical domains"""
    
    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
        self.feature_matrix = None
        self.feature_names = []
        self.record_names = []
        
    def clean_signal(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Clean signal by removing NaN, inf values and ensuring finite data"""
        if len(ppg_signal) == 0:
            return ppg_signal
            
        # Remove NaN and infinite values
        clean_signal = ppg_signal.copy()
        
        # Replace NaN and inf with interpolated values
        if np.any(~np.isfinite(clean_signal)):
            # Find finite values
            finite_mask = np.isfinite(clean_signal)
            
            if np.sum(finite_mask) == 0:
                # All values are invalid, return zeros
                return np.zeros_like(clean_signal)
            
            if np.sum(finite_mask) < len(clean_signal):
                # Interpolate missing values
                finite_indices = np.where(finite_mask)[0]
                finite_values = clean_signal[finite_mask]
                
                # Simple linear interpolation
                clean_signal = np.interp(
                    np.arange(len(clean_signal)), 
                    finite_indices, 
                    finite_values
                )
        
        return clean_signal
        
    def load_preprocessed_data(self) -> Dict[str, Dict]:
        """Load already preprocessed PPG data from saved file"""
        preprocessed_path = PATHS.PROCESSED_DATA_DIR / "preprocessed_ppg_signals.npz"
        
        if not preprocessed_path.exists():
            print(f"ERROR: Preprocessed data not found at: {preprocessed_path}")
            print("Please run ppg_preprocessor.py first to generate preprocessed data.")
            return {}
        
        print(f"Loading preprocessed data from: {preprocessed_path}")
        
        # Load the preprocessed data
        data = np.load(preprocessed_path, allow_pickle=True)
        
        # Extract data components
        record_names = data['record_names']
        quality_scores = data['quality_scores']
        heart_rates = data['heart_rates']
        
        # Reconstruct preprocessed data structure
        preprocessed_data = {}
        
        for i, record_name in enumerate(record_names):
            try:
                # Load normalized signal
                normalized_signal = data[f'normalized_{record_name}']
                
                # Clean signal to remove NaN/inf values
                normalized_signal = self.clean_signal(normalized_signal)
                
                # Skip if signal is empty or all zeros after cleaning
                if len(normalized_signal) == 0 or np.all(normalized_signal == 0):
                    continue
                
                # Reconstruct peaks data (simplified from heart rate)
                heart_rate = heart_rates[i]
                
                # Estimate peaks from heart rate (approximate)
                if heart_rate > 0 and len(normalized_signal) > 100:
                    # Simple peak detection for this record
                    try:
                        peaks, _ = find_peaks(normalized_signal, 
                                            distance=int(self.sampling_rate * 60 / heart_rate * 0.8),
                                            height=np.mean(normalized_signal) + 0.3 * np.std(normalized_signal))
                        
                        if len(peaks) > 1:
                            rr_intervals = np.diff(peaks) / self.sampling_rate
                            peak_values = normalized_signal[peaks]
                        else:
                            peaks = np.array([])
                            rr_intervals = np.array([])
                            peak_values = np.array([])
                    except:
                        peaks = np.array([])
                        rr_intervals = np.array([])
                        peak_values = np.array([])
                else:
                    peaks = np.array([])
                    rr_intervals = np.array([])
                    peak_values = np.array([])
                
                # Reconstruct data structure
                preprocessed_data[record_name] = {
                    'normalized_signal': normalized_signal,
                    'quality_metrics': {'overall_quality': quality_scores[i]},
                    'peaks_data': {
                        'peak_indices': peaks,
                        'peak_values': peak_values,
                        'rr_intervals': rr_intervals,
                        'heart_rate': heart_rate
                    }
                }
                
            except Exception as e:
                print(f"ERROR loading {record_name}: {e}")
                continue
        
        print(f"Loaded {len(preprocessed_data)} preprocessed PPG signals")
        return preprocessed_data
    
    def extract_time_domain_features(self, ppg_signal: np.ndarray, peaks_data: Dict) -> Dict[str, float]:
        """Extract time-domain features from PPG signal and peaks"""
        features = {}
        
        # Clean signal first
        ppg_signal = self.clean_signal(ppg_signal)
        
        # Heart Rate Variability (HRV) Features
        rr_intervals = peaks_data.get('rr_intervals', np.array([]))
        
        if len(rr_intervals) > 1 and np.all(np.isfinite(rr_intervals)):
            # Basic HRV metrics
            features['mean_rr'] = np.mean(rr_intervals)
            features['std_rr'] = np.std(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            features['cv_rr'] = features['std_rr'] / features['mean_rr'] if features['mean_rr'] > 0 else 0
            
            # pNN50: percentage of RR intervals differing by more than 50ms
            rr_diff = np.abs(np.diff(rr_intervals * 1000))  # Convert to ms
            features['pnn50'] = np.sum(rr_diff > 50) / len(rr_diff) * 100 if len(rr_diff) > 0 else 0
            
            # Triangular index approximation
            try:
                hist_counts, _ = np.histogram(rr_intervals, bins=50)
                max_count = np.max(hist_counts)
                features['tri_index'] = len(rr_intervals) / max_count if max_count > 0 else 0
            except:
                features['tri_index'] = 0
            
        else:
            # Default values when insufficient peaks
            features.update({
                'mean_rr': 0, 'std_rr': 0, 'rmssd': 0, 
                'cv_rr': 0, 'pnn50': 0, 'tri_index': 0
            })
        
        # Heart Rate
        features['heart_rate'] = peaks_data.get('heart_rate', 0)
        
        # Peak-based features
        peak_indices = peaks_data.get('peak_indices', np.array([]))
        peak_values = peaks_data.get('peak_values', np.array([]))
        
        if len(peak_values) > 0 and np.all(np.isfinite(peak_values)):
            features['peak_amplitude_mean'] = np.mean(peak_values)
            features['peak_amplitude_std'] = np.std(peak_values)
            features['peak_amplitude_cv'] = features['peak_amplitude_std'] / features['peak_amplitude_mean'] if features['peak_amplitude_mean'] > 0 else 0
        else:
            features.update({
                'peak_amplitude_mean': 0, 'peak_amplitude_std': 0, 'peak_amplitude_cv': 0
            })
        
        # Signal morphology features
        if len(ppg_signal) > 100 and np.all(np.isfinite(ppg_signal)):
            # Rise time and fall time analysis
            if len(peak_indices) > 2:
                # Calculate average rise time (valley to peak)
                rise_times = []
                fall_times = []
                
                for i in range(1, len(peak_indices)-1):
                    peak_idx = peak_indices[i]
                    prev_peak_idx = peak_indices[i-1]
                    next_peak_idx = peak_indices[i+1]
                    
                    # Find valley before peak (minimum between previous and current peak)
                    valley_start = max(0, prev_peak_idx)
                    valley_end = min(len(ppg_signal), peak_idx)
                    
                    if valley_end > valley_start:
                        valley_segment = ppg_signal[valley_start:valley_end]
                        if len(valley_segment) > 0 and np.all(np.isfinite(valley_segment)):
                            valley_idx = valley_start + np.argmin(valley_segment)
                            
                            # Rise time (valley to peak)
                            rise_time = (peak_idx - valley_idx) / self.sampling_rate
                            if np.isfinite(rise_time) and rise_time > 0:
                                rise_times.append(rise_time)
                            
                            # Fall time (peak to next valley)
                            valley_next_start = peak_idx
                            valley_next_end = min(len(ppg_signal), next_peak_idx)
                            
                            if valley_next_end > valley_next_start:
                                valley_next_segment = ppg_signal[valley_next_start:valley_next_end]
                                if len(valley_next_segment) > 0 and np.all(np.isfinite(valley_next_segment)):
                                    valley_next_idx = valley_next_start + np.argmin(valley_next_segment)
                                    
                                    fall_time = (valley_next_idx - peak_idx) / self.sampling_rate
                                    if np.isfinite(fall_time) and fall_time > 0:
                                        fall_times.append(fall_time)
                
                features['mean_rise_time'] = np.mean(rise_times) if len(rise_times) > 0 else 0
                features['mean_fall_time'] = np.mean(fall_times) if len(fall_times) > 0 else 0
                features['rise_fall_ratio'] = features['mean_rise_time'] / features['mean_fall_time'] if features['mean_fall_time'] > 0 else 0
            else:
                features.update({
                    'mean_rise_time': 0, 'mean_fall_time': 0, 'rise_fall_ratio': 0
                })
        else:
            features.update({
                'mean_rise_time': 0, 'mean_fall_time': 0, 'rise_fall_ratio': 0
            })
        
        return features
    
    def extract_frequency_domain_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features using power spectral density"""
        features = {}
        
        # Clean signal first
        ppg_signal = self.clean_signal(ppg_signal)
        
        if len(ppg_signal) < 256 or not np.all(np.isfinite(ppg_signal)):
            return {
                'total_power': 0, 'vlf_power': 0, 'lf_power': 0, 'hf_power': 0,
                'lf_hf_ratio': 0, 'peak_frequency': 0, 'spectral_entropy': 0
            }
        
        try:
            # Compute power spectral density using Welch's method
            freqs, psd = welch(ppg_signal, fs=self.sampling_rate, nperseg=min(256, len(ppg_signal)//4))
            
            # Check if PSD is valid
            if not np.all(np.isfinite(psd)) or len(psd) == 0:
                return {
                    'total_power': 0, 'vlf_power': 0, 'lf_power': 0, 'hf_power': 0,
                    'lf_hf_ratio': 0, 'peak_frequency': 0, 'spectral_entropy': 0
                }
            
            # Define frequency bands (standard HRV analysis)
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)  # Very Low Frequency
            lf_band = (freqs >= 0.04) & (freqs < 0.15)    # Low Frequency  
            hf_band = (freqs >= 0.15) & (freqs < 0.4)     # High Frequency
            
            # Calculate power in each band
            features['total_power'] = np.trapz(psd, freqs) if len(psd) > 1 else 0
            features['vlf_power'] = np.trapz(psd[vlf_band], freqs[vlf_band]) if np.any(vlf_band) else 0
            features['lf_power'] = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
            features['hf_power'] = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
            
            # LF/HF ratio (important for autonomic balance)
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
            
            # Peak frequency (dominant frequency)
            if len(psd) > 0:
                peak_idx = np.argmax(psd)
                features['peak_frequency'] = freqs[peak_idx]
            else:
                features['peak_frequency'] = 0
            
            # Spectral entropy (measure of frequency distribution)
            if np.sum(psd) > 0:
                psd_norm = psd / np.sum(psd)
                psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
                if len(psd_norm) > 0:
                    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # Add small epsilon
                else:
                    features['spectral_entropy'] = 0
            else:
                features['spectral_entropy'] = 0
                
        except Exception as e:
            # Return default values if frequency analysis fails
            features = {
                'total_power': 0, 'vlf_power': 0, 'lf_power': 0, 'hf_power': 0,
                'lf_hf_ratio': 0, 'peak_frequency': 0, 'spectral_entropy': 0
            }
        
        return features
    
    def extract_statistical_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from PPG signal"""
        features = {}
        
        # Clean signal first
        ppg_signal = self.clean_signal(ppg_signal)
        
        if len(ppg_signal) == 0 or not np.all(np.isfinite(ppg_signal)):
            return {
                'mean': 0, 'std': 0, 'variance': 0, 'skewness': 0, 'kurtosis': 0,
                'min_val': 0, 'max_val': 0, 'range_val': 0, 'iqr': 0,
                'energy': 0, 'zero_crossing_rate': 0, 'signal_entropy': 0
            }
        
        try:
            # Basic statistical moments
            features['mean'] = np.mean(ppg_signal)
            features['std'] = np.std(ppg_signal)
            features['variance'] = np.var(ppg_signal)
            features['skewness'] = stats.skew(ppg_signal)
            features['kurtosis'] = stats.kurtosis(ppg_signal)
            
            # Range-based features
            features['min_val'] = np.min(ppg_signal)
            features['max_val'] = np.max(ppg_signal)
            features['range_val'] = features['max_val'] - features['min_val']
            
            # Quartile-based features
            q25, q75 = np.percentile(ppg_signal, [25, 75])
            features['iqr'] = q75 - q25
            
            # Energy and complexity measures
            features['energy'] = np.sum(ppg_signal ** 2)
            
            # Zero crossing rate (signal complexity)
            zero_crossings = np.sum(np.diff(np.sign(ppg_signal - np.mean(ppg_signal))) != 0)
            features['zero_crossing_rate'] = zero_crossings / len(ppg_signal) if len(ppg_signal) > 1 else 0
            
            # Signal entropy (measure of randomness) - FIXED VERSION
            if len(ppg_signal) > 10:
                try:
                    # Use finite range for histogram
                    signal_range = features['max_val'] - features['min_val']
                    if signal_range > 0 and np.isfinite(signal_range):
                        hist, _ = np.histogram(ppg_signal, bins=50, range=(features['min_val'], features['max_val']), density=True)
                        hist = hist[hist > 0]  # Remove zeros
                        if len(hist) > 0:
                            features['signal_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))  # Add small epsilon
                        else:
                            features['signal_entropy'] = 0
                    else:
                        features['signal_entropy'] = 0
                except:
                    features['signal_entropy'] = 0
            else:
                features['signal_entropy'] = 0
                
        except Exception as e:
            # Return default values if statistical analysis fails
            features = {
                'mean': 0, 'std': 0, 'variance': 0, 'skewness': 0, 'kurtosis': 0,
                'min_val': 0, 'max_val': 0, 'range_val': 0, 'iqr': 0,
                'energy': 0, 'zero_crossing_rate': 0, 'signal_entropy': 0
            }
        
        return features
    
    def extract_morphological_features(self, ppg_signal: np.ndarray, peaks_data: Dict) -> Dict[str, float]:
        """Extract morphological features from PPG waveform"""
        features = {}
        
        # Clean signal first
        ppg_signal = self.clean_signal(ppg_signal)
        
        peak_indices = peaks_data.get('peak_indices', np.array([]))
        
        if len(peak_indices) < 2 or len(ppg_signal) < 100 or not np.all(np.isfinite(ppg_signal)):
            return {
                'pulse_width_mean': 0, 'pulse_width_std': 0,
                'systolic_area': 0, 'diastolic_area': 0, 'area_ratio': 0,
                'amplitude_ratio': 0, 'slope_mean': 0, 'slope_std': 0
            }
        
        try:
            pulse_widths = []
            systolic_areas = []
            diastolic_areas = []
            amplitude_ratios = []
            slopes = []
            
            for i in range(1, len(peak_indices)-1):
                current_peak = peak_indices[i]
                next_peak = peak_indices[i+1]
                
                # Pulse width (peak to peak)
                pulse_width = (next_peak - current_peak) / self.sampling_rate
                if np.isfinite(pulse_width) and pulse_width > 0:
                    pulse_widths.append(pulse_width)
                
                # Extract single pulse segment
                pulse_segment = ppg_signal[current_peak:next_peak]
                
                if len(pulse_segment) > 10 and np.all(np.isfinite(pulse_segment)):
                    # Find minimum (diastolic point) in pulse
                    min_idx = np.argmin(pulse_segment)
                    diastolic_point = current_peak + min_idx
                    
                    # Systolic area (peak to diastolic minimum)
                    systolic_segment = ppg_signal[current_peak:diastolic_point]
                    diastolic_segment = ppg_signal[diastolic_point:next_peak]
                    
                    if len(systolic_segment) > 1 and np.all(np.isfinite(systolic_segment)):
                        systolic_area = np.trapz(systolic_segment)
                        if np.isfinite(systolic_area):
                            systolic_areas.append(abs(systolic_area))
                    
                    if len(diastolic_segment) > 1 and np.all(np.isfinite(diastolic_segment)):
                        diastolic_area = np.trapz(diastolic_segment)
                        if np.isfinite(diastolic_area):
                            diastolic_areas.append(abs(diastolic_area))
                    
                    # Amplitude ratio (systolic peak / diastolic minimum)
                    if len(pulse_segment) > 0:
                        peak_val = ppg_signal[current_peak]
                        valley_val = ppg_signal[diastolic_point]
                        if np.isfinite(peak_val) and np.isfinite(valley_val) and valley_val != 0:
                            amp_ratio = abs(peak_val / (valley_val + 1e-10))
                            if np.isfinite(amp_ratio):
                                amplitude_ratios.append(amp_ratio)
                    
                    # Slope analysis (derivative)
                    if len(pulse_segment) > 3:
                        pulse_derivative = np.gradient(pulse_segment)
                        if np.all(np.isfinite(pulse_derivative)):
                            slopes.extend(pulse_derivative)
            
            # Calculate morphological features
            features['pulse_width_mean'] = np.mean(pulse_widths) if len(pulse_widths) > 0 else 0
            features['pulse_width_std'] = np.std(pulse_widths) if len(pulse_widths) > 0 else 0
            
            features['systolic_area'] = np.mean(systolic_areas) if len(systolic_areas) > 0 else 0
            features['diastolic_area'] = np.mean(diastolic_areas) if len(diastolic_areas) > 0 else 0
            features['area_ratio'] = features['systolic_area'] / (features['diastolic_area'] + 1e-10) if features['diastolic_area'] > 0 else 0
            
            features['amplitude_ratio'] = np.mean(amplitude_ratios) if len(amplitude_ratios) > 0 else 0
            
            features['slope_mean'] = np.mean(slopes) if len(slopes) > 0 else 0
            features['slope_std'] = np.std(slopes) if len(slopes) > 0 else 0
            
        except Exception as e:
            # Return default values if morphological analysis fails
            features = {
                'pulse_width_mean': 0, 'pulse_width_std': 0,
                'systolic_area': 0, 'diastolic_area': 0, 'area_ratio': 0,
                'amplitude_ratio': 0, 'slope_mean': 0, 'slope_std': 0
            }
        
        return features
    
    def extract_all_features(self, preprocessed_data: Dict[str, Dict]) -> pd.DataFrame:
        """Extract all features from preprocessed PPG data with minimal logging"""
        print(f"Extracting features from {len(preprocessed_data)} PPG signals...")
        print("="*50)
        
        all_features = []
        self.record_names = []
        successful_count = 0
        error_count = 0
        
        for i, (record_name, data) in enumerate(preprocessed_data.items()):
            try:
                # Get preprocessed signal and peaks data
                ppg_signal = data['normalized_signal']
                peaks_data = data['peaks_data']
                quality_score = data['quality_metrics']['overall_quality']
                
                # Skip very poor quality signals or invalid signals
                if quality_score < 0.1 or len(ppg_signal) == 0:
                    print(f"{record_name} ✗ SKIPPED: Low quality ({quality_score:.3f})")
                    continue
                
                # Extract features from different domains
                time_features = self.extract_time_domain_features(ppg_signal, peaks_data)
                freq_features = self.extract_frequency_domain_features(ppg_signal)
                stat_features = self.extract_statistical_features(ppg_signal)
                morph_features = self.extract_morphological_features(ppg_signal, peaks_data)
                
                # Combine all features
                combined_features = {
                    'record_name': record_name,
                    'quality_score': quality_score,
                    **time_features,
                    **freq_features,
                    **stat_features,
                    **morph_features
                }
                
                all_features.append(combined_features)
                self.record_names.append(record_name)
                successful_count += 1
                
                # Show success
                print(f"{record_name} ✓")
                
            except Exception as e:
                print(f"{record_name} ✗ ERROR: {str(e)}")
                error_count += 1
                continue
        
        # Create feature matrix
        self.feature_matrix = pd.DataFrame(all_features)
        
        # Store feature names (excluding metadata columns)
        self.feature_names = [col for col in self.feature_matrix.columns 
                             if col not in ['record_name', 'quality_score']]
        
        print("="*50)
        print(f"Feature extraction complete!")
        print(f"Features extracted: {len(self.feature_names)}")
        print(f"Signals processed: {successful_count} success, {error_count} errors")
        
        return self.feature_matrix
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary statistics of extracted features"""
        if self.feature_matrix is None:
            return pd.DataFrame()
        
        # Calculate summary statistics for numeric features only
        numeric_features = self.feature_matrix.select_dtypes(include=[np.number])
        summary = numeric_features.describe().round(4)
        
        return summary
    
    def plot_feature_distributions(self, n_features: int = 8):
        """Plot distributions of top features"""
        if self.feature_matrix is None:
            return
        
        # Select features with highest variance for plotting
        numeric_features = self.feature_matrix.select_dtypes(include=[np.number])
        feature_vars = numeric_features.var().sort_values(ascending=False)
        top_features = feature_vars.head(n_features).index.tolist()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i >= len(axes):
                break
                
            data = self.feature_matrix[feature].dropna()
            
            if len(data) > 0:
                axes[i].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{feature}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(top_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle('PPG Feature Distributions (Top 8 by Variance)', y=1.02, fontsize=14)
        plt.show()
    
    def save_features(self):
        """Save extracted features to file"""
        if self.feature_matrix is None:
            return
        
        # Save feature matrix
        features_path = PATHS.PROCESSED_DATA_DIR / "ppg_features.csv"
        self.feature_matrix.to_csv(features_path, index=False)
        
        # Save feature names separately
        feature_names_path = PATHS.PROCESSED_DATA_DIR / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print(f"Features saved to: ppg_features.csv")
        print(f"Feature names saved to: feature_names.txt")

# Main execution function
def run_feature_extraction():
    """Main function to run complete feature extraction pipeline"""
    print("=== PPG Feature Extraction Pipeline ===")
    
    # Initialize feature extractor
    feature_extractor = PPGFeatureExtractor(sampling_rate=250)
    
    # Load preprocessed data (no re-preprocessing!)
    preprocessed_data = feature_extractor.load_preprocessed_data()
    
    if not preprocessed_data:
        print("ERROR: No preprocessed data available. Exiting.")
        return None, None
    
    # Extract all features
    feature_matrix = feature_extractor.extract_all_features(preprocessed_data)
    
    # Show feature summary
    print("\nFeature Summary Statistics:")
    summary = feature_extractor.get_feature_summary()
    print(f"Total features: {len(feature_extractor.feature_names)}")
    print(f"Feature categories: Time-domain, Frequency-domain, Statistical, Morphological")
    
    # Plot feature distributions
    print("\nGenerating feature distribution plots...")
    feature_extractor.plot_feature_distributions()
    
    # Save features
    feature_extractor.save_features()
    
    print("\n=== Feature Extraction Complete ===")
    
    return feature_extractor, feature_matrix

if __name__ == "__main__":
    run_feature_extraction()
