# src/quantum_feature_encoding.py

import pennylane as qml
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class QuantumFeatureEncoder:
    """
    Quantum feature encoding methods for classical PPG features.
    Converts classical data into quantum states.
    """
    
    def __init__(self, n_features, n_qubits=None):
        """
        Initialize quantum feature encoder.
        
        Args:
            n_features (int): Number of classical features
            n_qubits (int): Number of qubits (default: ceil(log2(n_features)))
        """
        self.n_features = n_features
        
        if n_qubits is None:
            # Minimum qubits needed for amplitude encoding
            self.n_qubits = int(np.ceil(np.log2(n_features)))
        else:
            self.n_qubits = n_qubits
        
        print(f"Initialized Quantum Encoder:")
        print(f"  Features: {self.n_features}")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Hilbert space dimension: {2**self.n_qubits}")
    
    def amplitude_encoding(self, features, wires):
        """
        Amplitude encoding: Encode features into quantum state amplitudes.
        |ψ⟩ = Σ f_i |i⟩ where f_i are normalized features
        
        Args:
            features (array): Classical features to encode
            wires (list): Qubit wires to use
        """
        # Normalize features to unit vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features_normalized = features / norm
        else:
            features_normalized = features
        
        # Pad to power of 2 if needed
        target_dim = 2 ** len(wires)
        if len(features_normalized) < target_dim:
            features_padded = np.pad(features_normalized, 
                                     (0, target_dim - len(features_normalized)))
        else:
            features_padded = features_normalized[:target_dim]
        
        # Ensure normalization after padding
        features_padded = features_padded / np.linalg.norm(features_padded)
        
        # Use PennyLane's AmplitudeEmbedding
        qml.AmplitudeEmbedding(features_padded, wires=wires, normalize=True)
    
    def angle_encoding(self, features, wires):
        """
        Angle encoding: Encode features as rotation angles.
        Apply RY(f_i) to each qubit.
        
        Args:
            features (array): Classical features to encode
            wires (list): Qubit wires to use
        """
        n_wires = len(wires)
        
        # Use features cyclically if more qubits than features
        for i, wire in enumerate(wires):
            feature_idx = i % len(features)
            # Scale feature to [0, 2π] range
            angle = features[feature_idx] * np.pi
            qml.RY(angle, wires=wire)
    
    def iqp_encoding(self, features, wires, n_repeats=1):
        """
        IQP (Instantaneous Quantum Polynomial) encoding.
        Diagonal gates followed by entanglement.
        
        Args:
            features (array): Classical features to encode
            wires (list): Qubit wires to use
            n_repeats (int): Number of encoding layers
        """
        n_wires = len(wires)
        
        for _ in range(n_repeats):
            # Layer 1: Hadamard gates
            for wire in wires:
                qml.Hadamard(wires=wire)
            
            # Layer 2: Feature-dependent rotations
            for i, wire in enumerate(wires):
                feature_idx = i % len(features)
                qml.RZ(features[feature_idx] * np.pi, wires=wire)
            
            # Layer 3: Entanglement
            for i in range(n_wires - 1):
                feature_idx = i % len(features)
                qml.CRZ(features[feature_idx] * np.pi, 
                       wires=[wires[i], wires[i+1]])
    
    def basis_encoding(self, features, wires):
        """
        Basis encoding: Binary encoding of features.
        Each feature is thresholded and encoded as |0⟩ or |1⟩.
        
        Args:
            features (array): Classical features to encode
            wires (list): Qubit wires to use
        """
        # Threshold features at median
        threshold = np.median(features)
        binary_features = (features > threshold).astype(int)
        
        # Encode on qubits
        for i, wire in enumerate(wires):
            if i < len(binary_features):
                if binary_features[i] == 1:
                    qml.PauliX(wires=wire)
    
    def dense_angle_encoding(self, features, wires):
        """
        Dense angle encoding: Uses all three rotation gates (RX, RY, RZ).
        More expressive than simple angle encoding.
        
        Args:
            features (array): Classical features to encode
            wires (list): Qubit wires to use
        """
        n_wires = len(wires)
        
        # First layer: RY rotations
        for i, wire in enumerate(wires):
            feature_idx = i % len(features)
            qml.RY(features[feature_idx] * np.pi, wires=wire)
        
        # Second layer: RZ rotations
        for i, wire in enumerate(wires):
            feature_idx = (i + 1) % len(features)
            qml.RZ(features[feature_idx] * np.pi, wires=wire)
        
        # Third layer: RX rotations with entanglement
        for i, wire in enumerate(wires):
            feature_idx = (i + 2) % len(features)
            qml.RX(features[feature_idx] * np.pi, wires=wire)
    
    def get_encoding_circuit(self, encoding_type='amplitude', n_qubits=None):
        """
        Create a quantum device and circuit for specified encoding.
        
        Args:
            encoding_type (str): Type of encoding ('amplitude', 'angle', 'iqp', 'basis', 'dense_angle')
            n_qubits (int): Number of qubits (default: self.n_qubits)
            
        Returns:
            tuple: (device, qnode function)
        """
        if n_qubits is None:
            n_qubits = self.n_qubits
        
        # Create quantum device
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def encoding_circuit(features):
            """Quantum circuit for feature encoding."""
            wires = list(range(n_qubits))
            
            if encoding_type == 'amplitude':
                self.amplitude_encoding(features, wires)
            elif encoding_type == 'angle':
                self.angle_encoding(features, wires)
            elif encoding_type == 'iqp':
                self.iqp_encoding(features, wires, n_repeats=1)
            elif encoding_type == 'basis':
                self.basis_encoding(features, wires)
            elif encoding_type == 'dense_angle':
                self.dense_angle_encoding(features, wires)
            else:
                raise ValueError(f"Unknown encoding type: {encoding_type}")
            
            # Return state vector (for testing)
            return qml.state()
        
        return dev, encoding_circuit
    
    def visualize_encoding(self, features, encoding_type='amplitude', n_qubits=None):
        """
        Visualize the quantum encoding circuit.
        
        Args:
            features (array): Sample features to encode
            encoding_type (str): Type of encoding
            n_qubits (int): Number of qubits
        """
        if n_qubits is None:
            n_qubits = self.n_qubits
        
        dev, circuit = self.get_encoding_circuit(encoding_type, n_qubits)
        
        print(f"\n{'='*60}")
        print(f"Quantum Encoding Circuit: {encoding_type.upper()}")
        print(f"{'='*60}")
        print(f"Qubits: {n_qubits}")
        print(f"Features encoded: {len(features[:2**n_qubits])}")
        print(f"\nCircuit drawer:")
        print(qml.draw(circuit)(features))
        
        return circuit
    
    def test_encoding(self, sample_features, encoding_type='amplitude'):
        """
        Test encoding with sample features and return quantum state.
        
        Args:
            sample_features (array): Sample features to test
            encoding_type (str): Type of encoding
            
        Returns:
            array: Quantum state vector
        """
        dev, circuit = self.get_encoding_circuit(encoding_type)
        
        # Get quantum state
        state = circuit(sample_features)
        
        print(f"\n{'='*60}")
        print(f"Testing {encoding_type.upper()} Encoding")
        print(f"{'='*60}")
        print(f"Input features shape: {sample_features.shape}")
        print(f"Output quantum state shape: {state.shape}")
        print(f"State norm (should be 1.0): {np.linalg.norm(state):.6f}")
        print(f"First 5 amplitudes: {state[:5]}")
        
        return state
    
    def compare_encodings(self, sample_features, encodings=['amplitude', 'angle', 'iqp', 'dense_angle']):
        """
        Compare different encoding methods.
        
        Args:
            sample_features (array): Sample features
            encodings (list): List of encoding types to compare
            
        Returns:
            dict: Dictionary of encoding results
        """
        print(f"\n{'='*60}")
        print("COMPARING QUANTUM ENCODING METHODS")
        print(f"{'='*60}")
        
        results = {}
        
        for encoding in encodings:
            print(f"\n{encoding.upper()} Encoding:")
            print("-" * 40)
            
            try:
                state = self.test_encoding(sample_features, encoding)
                
                # Calculate some metrics
                results[encoding] = {
                    'state': state,
                    'norm': np.linalg.norm(state),
                    'entropy': -np.sum(np.abs(state)**2 * np.log2(np.abs(state)**2 + 1e-10))
                }
                
                print(f"Encoding successful!")
                print(f"Von Neumann entropy: {results[encoding]['entropy']:.4f}")
                
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                results[encoding] = None
        
        return results


# Example usage and testing
def main():
    """
    Main function to test quantum feature encoding.
    """
    print("="*60)
    print("QUANTUM FEATURE ENCODING - PHASE 4 STEP 1")
    print("="*60)
    
    # Load sample data
    from data_preparation import DataPreparator
    
    print("\nLoading sample PPG features...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='robust',
        feature_selection=True,
        selection_method='f_classif',
        k_features=16  # Use 16 features for 4-qubit encoding
    )
    
    # Get first sample
    sample_features = prepared_data['X_train'][0]
    print(f"Sample features: {sample_features.shape}")
    print(f"Feature values (first 5): {sample_features[:5]}")
    
    # Initialize encoder
    n_features = len(sample_features)
    encoder = QuantumFeatureEncoder(n_features=n_features, n_qubits=4)
    
    # Test different encodings
    print("\n" + "="*60)
    print("TESTING ENCODING METHODS")
    print("="*60)
    
    # 1. Amplitude encoding
    print("\n1. AMPLITUDE ENCODING")
    state_amp = encoder.test_encoding(sample_features, 'amplitude')
    circuit_amp = encoder.visualize_encoding(sample_features, 'amplitude', n_qubits=4)
    
    # 2. Angle encoding
    print("\n2. ANGLE ENCODING")
    state_angle = encoder.test_encoding(sample_features, 'angle')
    circuit_angle = encoder.visualize_encoding(sample_features, 'angle', n_qubits=4)
    
    # 3. IQP encoding
    print("\n3. IQP ENCODING")
    state_iqp = encoder.test_encoding(sample_features, 'iqp')
    circuit_iqp = encoder.visualize_encoding(sample_features, 'iqp', n_qubits=4)
    
    # 4. Dense angle encoding
    print("\n4. DENSE ANGLE ENCODING")
    state_dense = encoder.test_encoding(sample_features, 'dense_angle')
    circuit_dense = encoder.visualize_encoding(sample_features, 'dense_angle', n_qubits=4)
    
    # Compare all encodings
    results = encoder.compare_encodings(sample_features)
    
    # Summary
    print("\n" + "="*60)
    print("ENCODING COMPARISON SUMMARY")
    print("="*60)
    
    summary_data = []
    for enc_name, enc_result in results.items():
        if enc_result is not None:
            summary_data.append({
                'Encoding': enc_name,
                'State_Norm': enc_result['norm'],
                'Entropy': enc_result['entropy']
            })
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("PHASE 4 STEP 1 COMPLETED!")
    print("="*60)
    print("\nRecommendation: Use 'amplitude' or 'dense_angle' encoding")
    print("for maximum quantum expressiveness with your PPG features.")
    
    return encoder, results, sample_features

if __name__ == "__main__":
    encoder, results, features = main()