import numpy as np
import neurokit2 as nk
from typing import Dict, Tuple, List
import pandas as pd

class PPGDataGenerator:
    """Generate synthetic PPG signals for CAD vs healthy classification"""
    
    def __init__(self, sampling_rate: int = 200):
        self.sampling_rate = sampling_rate
        self.duration = 30  # 30 seconds per signal
        
    def generate_healthy_ppg(self, n_signals: int = 100) -> np.ndarray:
        """Generate healthy PPG signals with normal characteristics"""
        signals = []
        
        for i in range(n_signals):
            # Healthy parameters: normal HR, good variability
            heart_rate = np.random.normal(75, 8)  # 75 ± 8 BPM
            heart_rate_std = np.random.normal(0.05, 0.02)  # Good HRV
            
            ppg = nk.ppg_simulate(
                duration=self.duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                heart_rate_std=heart_rate_std
            )
            
            signals.append(ppg)
        
        return np.array(signals)
    
        def generate_cad_ppg(self, n_signals: int = 100) -> np.ndarray:
            """Generate CAD-affected PPG signals with pathological characteristics"""
            signals = []
        
            for i in range(n_signals):
                # CAD parameters: irregular HR, reduced variability, artifacts
                heart_rate = np.random.normal(85, 12)  # Higher, more variable HR
                heart_rate_std = np.random.normal(0.02, 0.01)  # Reduced HRV
            
                # Generate base PPG
                ppg = nk.ppg_simulate(
                    duration=self.duration,
                    sampling_rate=self.sampling_rate,
                    heart_rate=heart_rate,
                    heart_rate_std=heart_rate_std
                )
            
            # Add CAD-specific artifacts
            ppg = self._add_cad_artifacts(ppg)
            
            signals.append(ppg)
        
        return np.array(signals)
        
    def _add_cad_artifacts(self, ppg: np.ndarray) -> np.ndarray:
        """Add CAD-specific artifacts to PPG signal"""
        modified_ppg = ppg.copy()
        
        # 1. Reduced pulse amplitude variability (stiff arteries)
        amplitude_reduction = np.random.uniform(0.7, 0.9)
        modified_ppg *= amplitude_reduction
        
        # 2. Add baseline drift (poor circulation)
        drift = np.linspace(0, np.random.uniform(-0.1, 0.1), len(ppg))
        modified_ppg += drift
        
        # 3. Add occasional irregular beats (arrhythmia)
        if np.random.random() < 0.3:  # 30% chance
            irregular_indices = np.random.choice(len(ppg), size=3, replace=False)
            for idx in irregular_indices:
                if idx < len(ppg) - 100:
                    modified_ppg[idx:idx+50] *= np.random.uniform(0.5, 1.5)
        
        # 4. Add noise (measurement artifacts)
        noise = np.random.normal(0, 0.02, len(ppg))
        modified_ppg += noise
        
        return modified_ppg

