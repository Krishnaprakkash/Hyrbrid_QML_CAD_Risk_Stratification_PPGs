# src/variational_quantum_circuit.py

import pennylane as qml
import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for classification.
    Implements trainable quantum layers on top of encoded features.
    """
    
    def __init__(self, n_qubits, n_layers=2, encoding_type='amplitude'):
        """
        Initialize Variational Quantum Circuit.
        
        Args:
            n_qubits (int): Number of qubits
            n_layers (int): Number of variational layers
            encoding_type (str): Feature encoding method
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        
        # Calculate number of parameters
        # Each layer: n_qubits * 3 rotations + entangling gates
        self.n_params_per_layer = n_qubits * 3
        self.n_params = n_layers * self.n_params_per_layer
        
        print(f"Initialized Variational Quantum Circuit:")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Encoding: {self.encoding_type}")
        print(f"  Total trainable parameters: {self.n_params}")
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
    
    def feature_encoding_layer(self, features):
        """
        Encode classical features into quantum state.
        
        Args:
            features (array): Classical features
        """
        wires = list(range(self.n_qubits))
        
        if self.encoding_type == 'amplitude':
            # Normalize and pad features
            target_dim = 2 ** self.n_qubits
            if len(features) < target_dim:
                features_padded = np.pad(features, (0, target_dim - len(features)))
            else:
                features_padded = features[:target_dim]
            
            # Normalize
            features_padded = features_padded / (np.linalg.norm(features_padded) + 1e-10)
            qml.AmplitudeEmbedding(features_padded, wires=wires, normalize=True)
            
        elif self.encoding_type == 'angle':
            for i, wire in enumerate(wires):
                feature_idx = i % len(features)
                qml.RY(features[feature_idx] * np.pi, wires=wire)
                
        elif self.encoding_type == 'dense_angle':
            # RY rotations
            for i, wire in enumerate(wires):
                feature_idx = i % len(features)
                qml.RY(features[feature_idx] * np.pi, wires=wire)
            # RZ rotations
            for i, wire in enumerate(wires):
                feature_idx = (i + 1) % len(features)
                qml.RZ(features[feature_idx] * np.pi, wires=wire)
    
    def variational_layer(self, params):
        """
        Single variational layer with rotations and entanglement.
        
        Args:
            params (array): Parameters for this layer (shape: n_qubits * 3)
        """
        # Reshape parameters
        params_reshaped = params.reshape(self.n_qubits, 3)
        
        # Apply parameterized rotations to each qubit
        for i in range(self.n_qubits):
            qml.Rot(params_reshaped[i, 0], 
                   params_reshaped[i, 1], 
                   params_reshaped[i, 2], 
                   wires=i)
        
        # Entangling layer - circular CNOT chain
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def strongly_entangling_layer(self, params):
        """
        Strongly entangling variational layer.
        Uses PennyLane's built-in template.
        
        Args:
            params (array): Parameters for this layer
        """
        params_reshaped = params.reshape(self.n_qubits, 3)
        qml.StronglyEntanglingLayers(weights=[params_reshaped], 
                                     wires=range(self.n_qubits))
    
    def basic_entangling_layer(self, params):
        """
        Basic entangling layer with simple architecture.
        
        Args:
            params (array): Parameters for this layer
        """
        params_reshaped = params.reshape(self.n_qubits, 3)
        
        # Single-qubit rotations
        for i in range(self.n_qubits):
            qml.RX(params_reshaped[i, 0], wires=i)
            qml.RY(params_reshaped[i, 1], wires=i)
            qml.RZ(params_reshaped[i, 2], wires=i)
        
        # Entanglement - nearest neighbor
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def create_quantum_classifier(self, layer_type='basic'):
        """
        Create quantum circuit for binary classification.
        
        Args:
            layer_type (str): Type of variational layer ('basic', 'strong', 'custom')
            
        Returns:
            qnode: Quantum circuit function
        """
        @qml.qnode(self.dev, interface='numpy')
        def circuit(features, params):
            """
            Quantum circuit for classification.
            
            Args:
                features (array): Input features
                params (array): Variational parameters
                
            Returns:
                float: Expectation value for classification
            """
            # Feature encoding
            self.feature_encoding_layer(features)
            
            # Variational layers
            for layer_idx in range(self.n_layers):
                layer_params = params[layer_idx * self.n_params_per_layer:
                                     (layer_idx + 1) * self.n_params_per_layer]
                
                if layer_type == 'basic':
                    self.basic_entangling_layer(layer_params)
                elif layer_type == 'strong':
                    self.strongly_entangling_layer(layer_params)
                else:
                    self.variational_layer(layer_params)
            
            # Measurement - expectation value of Pauli-Z on first qubit
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def create_multi_measurement_classifier(self):
        """
        Create quantum circuit with multiple measurements.
        Measures all qubits and combines results.
        
        Returns:
            qnode: Quantum circuit function
        """
        @qml.qnode(self.dev, interface='numpy')
        def circuit(features, params):
            """Quantum circuit with multi-qubit measurement."""
            
            # Feature encoding
            self.feature_encoding_layer(features)
            
            # Variational layers
            for layer_idx in range(self.n_layers):
                layer_params = params[layer_idx * self.n_params_per_layer:
                                     (layer_idx + 1) * self.n_params_per_layer]
                self.basic_entangling_layer(layer_params)
            
            # Multiple measurements
            measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            return measurements
        
        return circuit
    
    def initialize_parameters(self, seed=42):
        """
        Initialize random parameters for the quantum circuit.
        
        Args:
            seed (int): Random seed
            
        Returns:
            array: Initial parameters
        """
        np.random.seed(seed)
        # Initialize parameters uniformly in [-π, π]
        params = np.random.uniform(-np.pi, np.pi, size=self.n_params)
        
        print(f"\nInitialized {self.n_params} random parameters")
        print(f"Parameter range: [{params.min():.4f}, {params.max():.4f}]")
        
        return params
    
    def visualize_circuit(self, sample_features, sample_params=None, layer_type='basic'):
        """
        Visualize the quantum circuit structure.
        
        Args:
            sample_features (array): Sample features for visualization
            sample_params (array): Sample parameters (optional)
            layer_type (str): Type of variational layer
        """
        if sample_params is None:
            sample_params = self.initialize_parameters()
        
        circuit = self.create_quantum_classifier(layer_type)
        
        print(f"\n{'='*60}")
        print(f"VARIATIONAL QUANTUM CIRCUIT - {layer_type.upper()}")
        print(f"{'='*60}")
        print(f"Encoding: {self.encoding_type}")
        print(f"Qubits: {self.n_qubits}")
        print(f"Layers: {self.n_layers}")
        print(f"Parameters: {self.n_params}")
        print(f"\nCircuit Structure:")
        
        # Fixed: Remove expansion_strategy argument
        print(qml.draw(circuit)(sample_features, sample_params))
        
        return circuit

    
    def test_circuit_output(self, sample_features, sample_params=None):
        """
        Test circuit with sample inputs.
        
        Args:
            sample_features (array): Sample features
            sample_params (array): Sample parameters
            
        Returns:
            float: Circuit output
        """
        if sample_params is None:
            sample_params = self.initialize_parameters()
        
        circuit = self.create_quantum_classifier()
        output = circuit(sample_features, sample_params)
        
        print(f"\n{'='*60}")
        print("CIRCUIT OUTPUT TEST")
        print(f"{'='*60}")
        print(f"Input features shape: {sample_features.shape}")
        print(f"Parameters shape: {sample_params.shape}")
        print(f"Output (expectation value): {output:.6f}")
        print(f"Output range: [-1, 1]")
        print(f"Prediction (0 if <0, 1 if >=0): {1 if output >= 0 else 0}")
        
        return output
    
    def test_gradient(self, sample_features, sample_params=None):
        """
        Test gradient computation using parameter-shift rule.
        
        Args:
            sample_features (array): Sample features
            sample_params (array): Sample parameters
            
        Returns:
            array: Gradient values
        """
        if sample_params is None:
            sample_params = self.initialize_parameters()
        
        circuit = self.create_quantum_classifier()
        
        # Compute gradient
        grad_fn = qml.grad(circuit, argnum=1)
        gradients = grad_fn(sample_features, sample_params)
        
        print(f"\n{'='*60}")
        print("GRADIENT COMPUTATION TEST")
        print(f"{'='*60}")
        print(f"Gradient shape: {gradients.shape}")
        print(f"Gradient norm: {np.linalg.norm(gradients):.6f}")
        print(f"Non-zero gradients: {np.count_nonzero(gradients)}/{len(gradients)}")
        print(f"First 5 gradient values: {gradients[:5]}")
        
        # Check for vanishing gradients
        if np.linalg.norm(gradients) < 1e-6:
            print("⚠️ WARNING: Vanishing gradients detected!")
        else:
            print("✓ Gradients are healthy")
        
        return gradients
    
    def compare_layer_types(self, sample_features):
        """
        Compare different variational layer architectures.
        
        Args:
            sample_features (array): Sample features
            
        Returns:
            dict: Comparison results
        """
        print(f"\n{'='*60}")
        print("COMPARING VARIATIONAL LAYER TYPES")
        print(f"{'='*60}")
        
        layer_types = ['basic', 'strong', 'custom']
        results = {}
        
        for layer_type in layer_types:
            print(f"\n{layer_type.upper()} Layer:")
            print("-" * 40)
            
            try:
                params = self.initialize_parameters()
                circuit = self.create_quantum_classifier(layer_type)
                
                # Test output
                output = circuit(sample_features, params)
                
                # Test gradient
                grad_fn = qml.grad(circuit, argnum=1)
                gradients = grad_fn(sample_features, params)
                
                results[layer_type] = {
                    'output': output,
                    'gradient_norm': np.linalg.norm(gradients),
                    'non_zero_grads': np.count_nonzero(gradients)
                }
                
                print(f"Output: {output:.6f}")
                print(f"Gradient norm: {results[layer_type]['gradient_norm']:.6f}")
                print(f"Non-zero gradients: {results[layer_type]['non_zero_grads']}")
                
            except Exception as e:
                print(f"Error: {e}")
                results[layer_type] = None
        
        return results


# Main function
def main():
    """
    Main function to test Variational Quantum Circuits.
    """
    print("="*60)
    print("VARIATIONAL QUANTUM CIRCUITS - PHASE 4 STEP 2")
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
        k_features=16  # 16 features for 4 qubits
    )
    
    sample_features = prepared_data['X_train'][0]
    print(f"Sample features shape: {sample_features.shape}")
    
    # Initialize VQC
    print("\n" + "="*60)
    print("INITIALIZING VARIATIONAL QUANTUM CIRCUIT")
    print("="*60)
    
    vqc = VariationalQuantumCircuit(
        n_qubits=4,
        n_layers=2,
        encoding_type='amplitude'  # Using recommended encoding
    )
    
    # Initialize parameters
    params = vqc.initialize_parameters()
    
    # Visualize circuit
    print("\n" + "="*60)
    print("CIRCUIT VISUALIZATION")
    print("="*60)
    circuit = vqc.visualize_circuit(sample_features, params)
    
    # Test circuit output
    print("\n" + "="*60)
    print("TESTING CIRCUIT OUTPUT")
    print("="*60)
    output = vqc.test_circuit_output(sample_features, params)
    
    # Test gradients
    print("\n" + "="*60)
    print("TESTING GRADIENT COMPUTATION")
    print("="*60)
    gradients = vqc.test_gradient(sample_features, params)
    
    # Compare layer types
    print("\n" + "="*60)
    print("COMPARING LAYER ARCHITECTURES")
    print("="*60)
    layer_comparison = vqc.compare_layer_types(sample_features)
    
    # Summary
    print("\n" + "="*60)
    print("VQC ARCHITECTURE SUMMARY")
    print("="*60)
    
    import pandas as pd
    summary_data = []
    for layer_type, result in layer_comparison.items():
        if result is not None:
            summary_data.append({
                'Layer_Type': layer_type,
                'Output': result['output'],
                'Gradient_Norm': result['gradient_norm'],
                'Non_Zero_Grads': result['non_zero_grads']
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("PHASE 4 STEP 2 COMPLETED!")
    print("="*60)
    print("\nRecommendation: Use 'basic' layer architecture")
    print("for stable gradients and efficient training.")
    
    return vqc, params, circuit, gradients

if __name__ == "__main__":
    vqc, params, circuit, gradients = main()
