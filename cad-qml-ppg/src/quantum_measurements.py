# src/quantum_measurements.py

import pennylane as qml
import numpy as np
from typing import Callable, Tuple
import time

class QuantumMeasurementLayer:
    """
    Quantum measurement strategies and hybrid quantum-classical integration.
    Converts quantum circuit outputs into classical predictions.
    """
    
    def __init__(self, n_qubits):
        """
        Initialize quantum measurement layer.
        
        Args:
            n_qubits (int): Number of qubits
        """
        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        print(f"Initialized Quantum Measurement Layer:")
        print(f"  Qubits: {self.n_qubits}")
    
    def single_qubit_measurement(self, circuit_output):
        """
        Single qubit measurement - simplest approach.
        Measures expectation value of first qubit.
        
        Args:
            circuit_output (float): Expectation value from quantum circuit
            
        Returns:
            int: Binary prediction (0 or 1)
        """
        # Map [-1, 1] to [0, 1]
        probability = (circuit_output + 1) / 2
        prediction = 1 if probability > 0.5 else 0
        return prediction
    
    def multi_qubit_measurement(self, circuit_outputs):
        """
        Multi-qubit measurement - combines all qubit measurements.
        
        Args:
            circuit_outputs (array): Expectation values from all qubits
            
        Returns:
            int: Binary prediction
        """
        # Average all qubit measurements
        avg_expectation = np.mean(circuit_outputs)
        probability = (avg_expectation + 1) / 2
        prediction = 1 if probability > 0.5 else 0
        return prediction
    
    def weighted_measurement(self, circuit_outputs, weights=None):
        """
        Weighted measurement - learned weights for each qubit.
        
        Args:
            circuit_outputs (array): Expectation values from all qubits
            weights (array): Weights for each qubit (optional)
            
        Returns:
            int: Binary prediction
        """
        if weights is None:
            weights = np.ones(len(circuit_outputs)) / len(circuit_outputs)
        
        weighted_expectation = np.dot(circuit_outputs, weights)
        probability = (weighted_expectation + 1) / 2
        prediction = 1 if probability > 0.5 else 0
        return prediction
    
    def parity_measurement(self, circuit_outputs):
        """
        Parity measurement - count majority voting.
        
        Args:
            circuit_outputs (array): Expectation values from all qubits
            
        Returns:
            int: Binary prediction based on parity
        """
        # Convert to binary predictions
        binary_outputs = [(exp > 0).astype(int) for exp in circuit_outputs]
        # Majority vote
        prediction = 1 if sum(binary_outputs) > len(binary_outputs) / 2 else 0
        return prediction


class HybridQuantumClassicalModel:
    """
    Complete hybrid quantum-classical model for CAD detection.
    Integrates quantum circuits with classical post-processing.
    """
    
    def __init__(self, n_qubits, n_layers, encoding_type='amplitude', 
                 measurement_type='single', layer_type='basic'):
        """
        Initialize hybrid model.
        
        Args:
            n_qubits (int): Number of qubits
            n_layers (int): Number of variational layers
            encoding_type (str): Feature encoding method
            measurement_type (str): Measurement strategy
            layer_type (str): Variational layer architecture
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.measurement_type = measurement_type
        self.layer_type = layer_type
        
        # Import VQC components
        from quantum_feature_encoding import QuantumFeatureEncoder
        from variational_quantum_circuit import VariationalQuantumCircuit
        
        # Initialize components
        self.encoder = QuantumFeatureEncoder(n_features=2**n_qubits, n_qubits=n_qubits)
        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers, encoding_type)
        self.measurement_layer = QuantumMeasurementLayer(n_qubits)
        
        # Initialize parameters
        self.params = self.vqc.initialize_parameters()
        
        # Create quantum circuit
        if measurement_type == 'single':
            self.circuit = self.vqc.create_quantum_classifier(layer_type)
        else:
            self.circuit = self.vqc.create_multi_measurement_classifier()
        
        print(f"\nInitialized Hybrid Quantum-Classical Model:")
        print(f"  Architecture: {encoding_type} → VQC({n_layers} layers) → {measurement_type} measurement")
        print(f"  Total parameters: {len(self.params)}")
    
    def predict_single(self, features, params=None):
        """
        Make prediction for a single sample.
        
        Args:
            features (array): Input features
            params (array): Circuit parameters (optional)
            
        Returns:
            tuple: (prediction, probability)
        """
        if params is None:
            params = self.params
        
        # Get quantum circuit output
        circuit_output = self.circuit(features, params)
        
        # Apply measurement
        if self.measurement_type == 'single':
            # Single qubit measurement
            probability = (circuit_output + 1) / 2
            prediction = 1 if probability > 0.5 else 0
        else:
            # Multi-qubit measurement
            avg_output = np.mean(circuit_output)
            probability = (avg_output + 1) / 2
            prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def predict(self, X, params=None):
        """
        Make predictions for multiple samples.
        
        Args:
            X (array): Input features (n_samples, n_features)
            params (array): Circuit parameters (optional)
            
        Returns:
            array: Predictions
        """
        if params is None:
            params = self.params
        
        predictions = []
        
        for i, features in enumerate(X):
            pred, _ = self.predict_single(features, params)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X, params=None):
        """
        Predict probabilities for multiple samples.
        
        Args:
            X (array): Input features
            params (array): Circuit parameters (optional)
            
        Returns:
            array: Probabilities (n_samples, 2)
        """
        if params is None:
            params = self.params
        
        probabilities = []
        
        for features in X:
            _, prob = self.predict_single(features, params)
            probabilities.append([1 - prob, prob])  # [P(class=0), P(class=1)]
        
        return np.array(probabilities)
    
    def evaluate(self, X, y, params=None):
        """
        Evaluate model performance.
        
        Args:
            X (array): Input features
            y (array): True labels
            params (array): Circuit parameters (optional)
            
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        if params is None:
            params = self.params
        
        # Make predictions
        y_pred = self.predict(X, params)
        y_proba = self.predict_proba(X, params)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0
        }
        
        return metrics
    
    def test_inference_speed(self, X_sample, n_runs=10):
        """
        Test inference speed of the hybrid model.
        
        Args:
            X_sample (array): Sample data
            n_runs (int): Number of runs for averaging
            
        Returns:
            dict: Speed metrics
        """
        print(f"\n{'='*60}")
        print("TESTING INFERENCE SPEED")
        print(f"{'='*60}")
        
        times = []
        
        for i in range(n_runs):
            start = time.time()
            _ = self.predict(X_sample, self.params)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        time_per_sample = avg_time / len(X_sample)
        
        print(f"Samples: {len(X_sample)}")
        print(f"Runs: {n_runs}")
        print(f"Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
        print(f"Time per sample: {time_per_sample*1000:.2f} ms")
        print(f"Throughput: {len(X_sample)/avg_time:.2f} samples/second")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'time_per_sample': time_per_sample,
            'throughput': len(X_sample) / avg_time
        }


# Main function
def main():
    """
    Main function to test quantum measurements and hybrid integration.
    """
    print("="*60)
    print("QUANTUM MEASUREMENTS & HYBRID INTEGRATION - PHASE 4 STEP 3")
    print("="*60)
    
    # Load sample data
    from data_preparation import DataPreparator
    
    print("\nLoading PPG features...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='robust',
        feature_selection=True,
        selection_method='f_classif',
        k_features=16
    )
    
    X_train = prepared_data['X_train'][:100]  # Use subset for testing
    y_train = prepared_data['y_train'][:100]
    X_test = prepared_data['X_test'][:20]
    y_test = prepared_data['y_test'][:20]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize hybrid model
    print("\n" + "="*60)
    print("INITIALIZING HYBRID QUANTUM-CLASSICAL MODEL")
    print("="*60)
    
    model = HybridQuantumClassicalModel(
        n_qubits=4,
        n_layers=2,
        encoding_type='amplitude',
        measurement_type='single',
        layer_type='basic'
    )
    
    # Test single prediction
    print("\n" + "="*60)
    print("TESTING SINGLE SAMPLE PREDICTION")
    print("="*60)
    
    sample = X_test[0]
    true_label = y_test[0]
    
    prediction, probability = model.predict_single(sample)
    
    print(f"Sample features shape: {sample.shape}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {prediction}")
    print(f"Predicted probability (CAD): {probability:.4f}")
    print(f"Prediction: {'Correct ✓' if prediction == true_label else 'Incorrect ✗'}")
    
    # Test batch prediction
    print("\n" + "="*60)
    print("TESTING BATCH PREDICTION")
    print("="*60)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
    print(f"First 5 true labels: {y_test[:5]}")
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION (Untrained)")
    print("="*60)
    
    metrics = model.evaluate(X_test, y_test)
    
    print("Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nNote: Performance is random (untrained parameters)")
    print("Expected metrics after training: 0.45-0.65 range")
    
    # Test inference speed
    speed_metrics = model.test_inference_speed(X_test, n_runs=5)
    
    # Summary
    print("\n" + "="*60)
    print("HYBRID MODEL SUMMARY")
    print("="*60)
    print(f"Architecture:")
    print(f"  Input: {X_train.shape[1]} features")
    print(f"  Encoding: {model.encoding_type}")
    print(f"  Qubits: {model.n_qubits}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Parameters: {len(model.params)}")
    print(f"  Measurement: {model.measurement_type}")
    print(f"\nPerformance:")
    print(f"  Current Accuracy: {metrics['accuracy']:.4f} (untrained)")
    print(f"  Inference Speed: {speed_metrics['time_per_sample']*1000:.2f} ms/sample")
    
    print("\n" + "="*60)
    print("PHASE 4 STEP 3 COMPLETED!")
    print("="*60)
    print("\nNext: Phase 5 - Train the hybrid quantum-classical model")
    print("to improve accuracy from random to 0.50+ range!")
    
    return model, metrics, speed_metrics

if __name__ == "__main__":
    model, metrics, speed_metrics = main()
