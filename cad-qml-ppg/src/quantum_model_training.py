# src/quantum_model_training.py

import pennylane as qml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import ADASYN
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class QuantumModelTrainer:
    """
    Training pipeline for hybrid quantum-classical models.
    """
    
    def __init__(self, n_qubits=4, n_layers=2, encoding_type='amplitude', 
                 layer_type='basic'):
        """
        Initialize quantum model trainer.
        
        Args:
            n_qubits (int): Number of qubits
            n_layers (int): Number of variational layers
            encoding_type (str): Feature encoding method
            layer_type (str): Variational layer type
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.layer_type = layer_type
        
        # Import quantum components
        from variational_quantum_circuit import VariationalQuantumCircuit
        
        # Initialize VQC
        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers, encoding_type)
        self.circuit = self.vqc.create_quantum_classifier(layer_type)
        
        # Initialize parameters
        self.params = self.vqc.initialize_parameters()
        self.best_params = self.params.copy()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        print(f"Initialized Quantum Model Trainer:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Layers: {n_layers}")
        print(f"  Parameters: {len(self.params)}")
        print(f"  Encoding: {encoding_type}")
        print(f"  Architecture: {layer_type}")
    
    def binary_cross_entropy_loss(self, predictions, labels):
        """
        Binary cross-entropy loss function.
        
        Args:
            predictions (array): Model predictions (probabilities)
            labels (array): True labels
            
        Returns:
            float: Loss value
        """
        epsilon = 1e-10
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(labels * np.log(predictions) + 
                       (1 - labels) * np.log(1 - predictions))
        return loss
    
    def predict_batch(self, X, params):
        """
        Make predictions for a batch of samples.
        
        Args:
            X (array): Input features
            params (array): Circuit parameters
            
        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = []
        probabilities = []
        
        for features in X:
            # Get quantum circuit output
            output = self.circuit(features, params)
            
            # Convert to probability
            prob = (output + 1) / 2
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
        
        return np.array(predictions), np.array(probabilities)
    
    def cost_function(self, params, X_batch, y_batch):
        """
        Cost function for computing loss.
        
        Args:
            params (array): Circuit parameters
            X_batch (array): Batch of features
            y_batch (array): Batch of labels
            
        Returns:
            float: Loss value
        """
        probabilities = []
        for features in X_batch:
            output = self.circuit(features, params)
            prob = (output + 1) / 2
            probabilities.append(prob)
        
        probabilities = np.array(probabilities)
        loss = self.binary_cross_entropy_loss(probabilities, y_batch)
        
        return loss

    def compute_loss_and_gradients(self, X_batch, y_batch, params):
        """
        Compute loss and gradients for a batch using numerical gradients.
        
        Args:
            X_batch (array): Batch of input features
            y_batch (array): Batch of labels
            params (array): Current parameters
            
        Returns:
            tuple: (loss, gradients)
        """
        # Compute loss
        loss = self.cost_function(params, X_batch, y_batch)
        
        # Compute gradients using parameter-shift rule
        # Manual gradient computation for each parameter
        gradients = np.zeros_like(params)
        shift = np.pi / 2  # Parameter shift value
        
        for i in range(len(params)):
            # Shift parameter up
            params_shifted_up = params.copy()
            params_shifted_up[i] += shift
            loss_up = self.cost_function(params_shifted_up, X_batch, y_batch)
            
            # Shift parameter down
            params_shifted_down = params.copy()
            params_shifted_down[i] -= shift
            loss_down = self.cost_function(params_shifted_down, X_batch, y_batch)
            
            # Compute gradient using parameter-shift rule
            gradients[i] = (loss_up - loss_down) / 2
        
        return loss, gradients

    
    def train_epoch(self, X_train, y_train, params, learning_rate, batch_size=32):
        """
        Train for one epoch.
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            params (array): Current parameters
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            
        Returns:
            tuple: (updated_params, epoch_loss, epoch_accuracy)
        """
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        epoch_losses = []
        all_predictions = []
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Compute loss and gradients
            loss, gradients = self.compute_loss_and_gradients(X_batch, y_batch, params)
            
            # Update parameters using gradient descent
            params = params - learning_rate * gradients
            
            # Track metrics
            epoch_losses.append(loss)
            
            # Get predictions for accuracy
            preds, _ = self.predict_batch(X_batch, params)
            all_predictions.extend(preds)
        
        # Calculate epoch metrics
        epoch_loss = np.mean(epoch_losses)
        
        # Calculate accuracy on full training set
        train_preds, _ = self.predict_batch(X_train, params)
        epoch_accuracy = accuracy_score(y_train, train_preds)
        
        return params, epoch_loss, epoch_accuracy
    
    def evaluate(self, X, y, params):
        """
        Evaluate model on a dataset.
        
        Args:
            X (array): Input features
            y (array): True labels
            params (array): Model parameters
            
        Returns:
            dict: Evaluation metrics
        """
        predictions, probabilities = self.predict_batch(X, params)
        
        loss = self.binary_cross_entropy_loss(probabilities, y)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0
        }
        
        return metrics
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, learning_rate=0.01, batch_size=32,
              early_stopping_patience=10, verbose=True):
        """
        Train the quantum model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            X_val (array): Validation features
            y_val (array): Validation labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            early_stopping_patience (int): Patience for early stopping
            verbose (bool): Print training progress
            
        Returns:
            dict: Training history
        """
        print(f"\n{'='*60}")
        print("STARTING QUANTUM MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        
        best_val_f1 = 0
        patience_counter = 0
        
        params = self.params.copy()
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train one epoch
            params, train_loss, train_acc = self.train_epoch(
                X_train, y_train, params, learning_rate, batch_size
            )
            
            # Evaluate on validation set
            val_metrics = self.evaluate(X_val, y_val, params)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(learning_rate)
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1_score']:.4f}")
            
            # Early stopping based on F1-score
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                self.best_params = params.copy()
                patience_counter = 0
                if verbose:
                    print(f"  ✓ Best F1-score: {best_val_f1:.4f} (saved)")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        
        # Update params to best
        self.params = self.best_params
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Best validation F1-score: {best_val_f1:.4f}")
        
        return self.history
    
    def plot_training_history(self, save_path='visualizations/quantum_training_history.png'):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Learning Rate', fontweight='bold')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined metrics
        axes[1, 1].plot(epochs, self.history['train_loss'], 'b--', label='Train Loss', alpha=0.7)
        axes[1, 1].plot(epochs, self.history['val_loss'], 'r--', label='Val Loss', alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(epochs, self.history['train_accuracy'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, self.history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Loss', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        axes[1, 1].set_title('Loss and Accuracy Combined', fontweight='bold')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Quantum Model Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved: {save_path}")
        plt.close()
    
    def save_model(self, filepath='models/quantum_model.pkl'):
        """
        Save trained model parameters.
        
        Args:
            filepath (str): Path to save the model
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'params': self.best_params,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'encoding_type': self.encoding_type,
            'layer_type': self.layer_type,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved: {filepath}")


# Main function
def main():
    """
    Main function to train quantum model.
    """
    print("="*60)
    print("PHASE 5: HYBRID QUANTUM-CLASSICAL MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    from data_preparation import DataPreparator
    
    print("\nLoading and preparing data...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='robust',
        feature_selection=True,
        selection_method='f_classif',
        k_features=16  # 16 features for 4 qubits
    )
    
    # Apply ADASYN to balance training data
    print("\nApplying ADASYN resampling...")
    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(
        prepared_data['X_train'],
        prepared_data['y_train']
    )
    
    print(f"Original training size: {len(prepared_data['y_train'])}")
    print(f"Resampled training size: {len(y_train_resampled)}")
    print(f"Class distribution: {np.bincount(y_train_resampled)}")
    
    # Use subset for faster training (remove this for full training)
    n_train = min(200, len(X_train_resampled))
    n_val = min(50, len(prepared_data['X_val']))
    
    X_train_subset = X_train_resampled[:n_train]
    y_train_subset = y_train_resampled[:n_train]
    X_val_subset = prepared_data['X_val'][:n_val]
    y_val_subset = prepared_data['y_val'][:n_val]
    
    print(f"\nUsing training subset: {n_train} samples")
    print(f"Using validation subset: {n_val} samples")
    
    # Initialize trainer
    print("\n" + "="*60)
    print("INITIALIZING QUANTUM MODEL")
    print("="*60)
    
    trainer = QuantumModelTrainer(
        n_qubits=4,
        n_layers=2,
        encoding_type='amplitude',
        layer_type='basic'
    )
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING QUANTUM MODEL")
    print("="*60)
    
    history = trainer.train(
        X_train=X_train_subset,
        y_train=y_train_subset,
        X_val=X_val_subset,
        y_val=y_val_subset,
        epochs=30,
        learning_rate=0.01,
        batch_size=16,
        early_stopping_patience=5,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_metrics = trainer.evaluate(
        prepared_data['X_test'],
        prepared_data['y_test'],
        trainer.best_params
    )
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model()
    
    # Compare with classical baseline
    print("\n" + "="*60)
    print("COMPARISON WITH CLASSICAL BASELINE")
    print("="*60)
    
    classical_baseline = {
        'Model': 'Logistic Regression (ADASYN)',
        'F1_Score': 0.4500,
        'Recall': 0.9643,
        'Precision': 0.2935,
        'ROC_AUC': 0.4643
    }
    
    comparison_df = pd.DataFrame([
        {
            'Model': 'Quantum Hybrid Model',
            'F1_Score': test_metrics['f1_score'],
            'Recall': test_metrics['recall'],
            'Precision': test_metrics['precision'],
            'ROC_AUC': test_metrics['roc_auc']
        },
        classical_baseline
    ])
    
    print("\nPerformance Comparison:")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("PHASE 5 COMPLETED!")
    print("="*60)
    
    return trainer, history, test_metrics, comparison_df

if __name__ == "__main__":
    trainer, history, test_metrics, comparison = main()
