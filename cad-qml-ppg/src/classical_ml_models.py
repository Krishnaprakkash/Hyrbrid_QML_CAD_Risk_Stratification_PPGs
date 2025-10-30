# src/classical_ml_models.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

class ClassicalMLModels:
    """
    Implementation of classical machine learning models for CAD detection.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize classical ML models.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.trained_models = {}
        
    def initialize_models(self):
        """
        Initialize all classical ML models with optimized parameters.
        
        Returns:
            dict: Dictionary of initialized models
        """
        self.models = {
            'SVM_RBF': SVC(
                kernel='rbf',
                C=10.0,  # Increased
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'SVM_Linear': SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=200,  # Increased
                max_depth=15,  # Increased
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # Deeper network
                activation='relu',
                solver='adam',
                alpha=0.001,  # Regularization
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=1000,  # More iterations
                shuffle=True,
                early_stopping=True,  # Stop when no improvement
                validation_fraction=0.15,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=20,
                tol=1e-4,
                random_state=self.random_state,
                verbose=False
            ),
            'Logistic_Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            )
        }
        
        print(f"Initialized {len(self.models)} optimized classical ML models")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        
        return self.models

    
    def train_model(self, model_name, model, X_train, y_train):
        """
        Train a single model and record training time.
        
        Args:
            model_name (str): Name of the model
            model: Scikit-learn model instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            tuple: (trained_model, training_time)
        """
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"{model_name} training completed in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate a trained model on test data.
        
        Args:
            model_name (str): Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
        metrics['sensitivity'] = metrics['recall']  # Sensitivity = Recall
        
        print(f"\n{model_name} Test Results:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model_name, model, X_train, y_train, cv_folds=10):
        """
        Perform cross-validation on a model.
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation for {model_name}...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validate on multiple metrics
        cv_scores = {
            'accuracy': cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(model, X_train, y_train, cv=cv, scoring='precision'),
            'recall': cross_val_score(model, X_train, y_train, cv=cv, scoring='recall'),
            'f1': cross_val_score(model, X_train, y_train, cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        }
        
        # Calculate mean and std for each metric
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        print(f"{model_name} Cross-Validation Results (mean ± std):")
        print(f"  Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"  Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
        print(f"  Recall:    {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
        print(f"  F1-Score:  {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        print(f"  ROC-AUC:   {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        
        return cv_results
    
    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, 
                               X_val=None, y_val=None, perform_cv=True, cv_folds=10):
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            perform_cv (bool): Whether to perform cross-validation
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Complete results for all models
        """
        print("="*60)
        print("TRAINING AND EVALUATING CLASSICAL ML MODELS")
        print("="*60)
        
        # Initialize models if not already done
        if not self.models:
            self.initialize_models()
        
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Processing: {model_name}")
            print(f"{'='*60}")
            
            # Train model
            trained_model, training_time = self.train_model(model_name, model, X_train, y_train)
            self.trained_models[model_name] = trained_model
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model_name, trained_model, X_test, y_test)
            
            # Evaluate on validation set if provided
            val_metrics = None
            if X_val is not None and y_val is not None:
                print(f"\n{model_name} Validation Results:")
                val_metrics = self.evaluate_model(f"{model_name} (Validation)", 
                                                  trained_model, X_val, y_val)
            
            # Perform cross-validation
            cv_results = None
            if perform_cv:
                cv_results = self.cross_validate_model(model_name, model, X_train, y_train, cv_folds)
            
            # Store all results
            all_results[model_name] = {
                'training_time': training_time,
                'test_metrics': test_metrics,
                'validation_metrics': val_metrics,
                'cv_results': cv_results
            }
        
        self.results = all_results
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED AND EVALUATED SUCCESSFULLY")
        print("="*60)
        
        return all_results
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            tuple: (best_model_name, best_score, best_model)
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.results.items():
            score = results['test_metrics'][metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        best_model = self.trained_models[best_model_name]
        
        print(f"\nBest model based on {metric}: {best_model_name}")
        print(f"Score: {best_score:.4f}")
        
        return best_model_name, best_score, best_model
    
    def get_results_summary(self):
        """
        Generate a summary DataFrame of all results.
        
        Returns:
            pd.DataFrame: Summary of model performances
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        summary_data = []
        
        for model_name, results in self.results.items():
            test_metrics = results['test_metrics']
            cv_results = results['cv_results']
            
            row = {
                'Model': model_name,
                'Training_Time': f"{results['training_time']:.2f}s",
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_Precision': test_metrics['precision'],
                'Test_Recall': test_metrics['recall'],
                'Test_Specificity': test_metrics['specificity'],
                'Test_F1': test_metrics['f1_score'],
                'Test_ROC_AUC': test_metrics['roc_auc']
            }
            
            if cv_results:
                row['CV_Accuracy'] = cv_results['accuracy_mean']
                row['CV_ROC_AUC'] = cv_results['roc_auc_mean']
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        
        return df_summary

# Example usage function
def main():
    """
    Example usage of ClassicalMLModels class.
    """
    # Import data preparation
    from data_preparation import DataPreparator
    
    # Prepare data
    print("Preparing data...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data()
    
    # Initialize and train models
    ml_models = ClassicalMLModels()
    ml_models.initialize_models()
    
    # Train and evaluate all models
    results = ml_models.train_and_evaluate_all(
        X_train=prepared_data['X_train'],
        y_train=prepared_data['y_train'],
        X_test=prepared_data['X_test'],
        y_test=prepared_data['y_test'],
        X_val=prepared_data['X_val'],
        y_val=prepared_data['y_val'],
        perform_cv=True,
        cv_folds=10
    )
    
    # Get best model
    best_model_name, best_score, best_model = ml_models.get_best_model(metric='roc_auc')
    
    # Get results summary
    summary_df = ml_models.get_results_summary()
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    return ml_models, results, summary_df

if __name__ == "__main__":
    ml_models, results, summary_df = main()
