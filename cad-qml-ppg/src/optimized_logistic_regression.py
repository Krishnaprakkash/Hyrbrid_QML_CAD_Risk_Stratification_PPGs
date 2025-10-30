# src/optimized_logistic_regression.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import time

class OptimizedLogisticRegression:
    """
    Optimized Logistic Regression for imbalanced CAD detection with multiple improvement techniques.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the optimized logistic regression system.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.results = {}
        self.scaler = StandardScaler()
        
    def create_baseline_model(self, class_weight='balanced'):
        """
        Create baseline logistic regression model.
        
        Args:
            class_weight (str or dict): Class weights strategy
            
        Returns:
            LogisticRegression: Configured model
        """
        model = LogisticRegression(
            class_weight=class_weight,
            solver='saga',
            max_iter=2000,
            random_state=self.random_state,
            n_jobs=-1
        )
        return model
    
    def apply_smote(self, X_train, y_train, method='smote'):
        """
        Apply SMOTE or its variants to balance the training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method (str): 'smote', 'adasyn', 'borderline', 'smote_tomek', 'smote_enn'
            
        Returns:
            tuple: Resampled (X_train, y_train)
        """
        print(f"\nApplying {method.upper()} resampling...")
        print(f"Original distribution: {np.bincount(y_train)}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state, k_neighbors=5)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif method == 'borderline':
            sampler = BorderlineSMOTE(random_state=self.random_state, kind='borderline-1')
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        print(f"New training size: {len(y_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5):
        """
        Perform comprehensive hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters and best score
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING WITH GRID SEARCH")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['saga'],  # saga supports all penalties
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],  # For elasticnet
            'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
        }
        
        # Create base model
        base_model = LogisticRegression(
            max_iter=2000,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Setup GridSearchCV with stratified k-fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',  # Optimize for F1-score (best for imbalanced data)
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        print(f"Testing {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['class_weight'])} parameter combinations...")
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"\nGrid search completed in {training_time:.2f} seconds")
        print(f"Best F1-Score: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")
        
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'training_time': training_time
        }
    
    def train_model(self, X_train, y_train, params=None):
        """
        Train logistic regression model with specified parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            params (dict): Model parameters
            
        Returns:
            tuple: (trained_model, training_time)
        """
        if params is None:
            params = self.best_params if self.best_params else {}
        
        print("\nTraining optimized logistic regression model...")
        
        model = LogisticRegression(
            **params,
            max_iter=2000,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_model(self, model, X_test, y_test, set_name="Test"):
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            set_name (str): Name of the dataset being evaluated
            
        Returns:
            dict: Comprehensive metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        
        # Print results
        print(f"\n{set_name} Set Evaluation:")
        print("="*50)
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"Precision (PPV):  {metrics['precision']:.4f}")
        print(f"Recall (Sens):    {metrics['recall']:.4f}")
        print(f"Specificity:      {metrics['specificity']:.4f}")
        print(f"F1-Score:         {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
        print(f"NPV:              {metrics['npv']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn}  FP: {fp}")
        print(f"  FN: {fn}  TP: {tp}")
        
        return metrics
    
    def train_multiple_strategies(self, X_train, y_train, X_test, y_test):
        """
        Train and compare multiple strategies for handling imbalanced data.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Results for all strategies
        """
        print("\n" + "="*60)
        print("TRAINING MULTIPLE IMBALANCE HANDLING STRATEGIES")
        print("="*60)
        
        strategies = {
            'Baseline_Balanced': None,
            'SMOTE': 'smote',
            'ADASYN': 'adasyn',
            'BorderlineSMOTE': 'borderline',
            'SMOTE_Tomek': 'smote_tomek'
        }
        
        results = {}
        
        for strategy_name, method in strategies.items():
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_name}")
            print(f"{'='*60}")
            
            # Prepare training data
            if method is None:
                # Baseline with balanced class weights
                X_train_processed = X_train
                y_train_processed = y_train
                model_params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'saga',
                    'class_weight': 'balanced'
                }
            else:
                # Apply resampling technique
                X_train_processed, y_train_processed = self.apply_smote(X_train, y_train, method)
                model_params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'saga',
                    'class_weight': None  # No need for class_weight after SMOTE
                }
            
            # Train model
            model, training_time = self.train_model(X_train_processed, y_train_processed, model_params)
            
            # Evaluate model
            test_metrics = self.evaluate_model(model, X_test, y_test, f"{strategy_name} Test")
            
            # Store results
            results[strategy_name] = {
                'model': model,
                'training_time': training_time,
                'test_metrics': test_metrics,
                'params': model_params
            }
        
        self.results = results
        return results
    
    def get_best_strategy(self, metric='f1_score'):
        """
        Get the best performing strategy.
        
        Args:
            metric (str): Metric to compare
            
        Returns:
            tuple: (best_strategy_name, best_score, best_model)
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        best_score = -1
        best_strategy = None
        
        for strategy_name, results in self.results.items():
            score = results['test_metrics'][metric]
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        print(f"\n{'='*60}")
        print(f"BEST STRATEGY: {best_strategy}")
        print(f"Best {metric}: {best_score:.4f}")
        print(f"{'='*60}")
        
        return best_strategy, best_score, self.results[best_strategy]['model']
    
    def get_results_summary(self):
        """
        Generate summary DataFrame of all strategies.
        
        Returns:
            pd.DataFrame: Summary of performances
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        summary_data = []
        
        for strategy_name, results in self.results.items():
            metrics = results['test_metrics']
            
            row = {
                'Strategy': strategy_name,
                'Training_Time': f"{results['training_time']:.2f}s",
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Specificity': metrics['specificity'],
                'F1_Score': metrics['f1_score'],
                'ROC_AUC': metrics['roc_auc']
            }
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('F1_Score', ascending=False)
        
        return df_summary


# Main execution function
def main():
    """
    Main execution function with hyperparameter tuning for best performance.
    """
    print("="*60)
    print("OPTIMIZED LOGISTIC REGRESSION WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    # Import data preparation
    from data_preparation import DataPreparator
    
    # Prepare data with more features
    print("\nPreparing data...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='robust',  # Changed to robust scaling
        feature_selection=False,   # Use ALL features
        selection_method='f_classif',
        k_features=40
    )
    
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    
    # Initialize optimizer
    optimizer = OptimizedLogisticRegression()
    
    # Strategy 1: Apply ADASYN (your best performer)
    print("\n" + "="*60)
    print("STEP 1: APPLYING ADASYN RESAMPLING")
    print("="*60)
    X_train_resampled, y_train_resampled = optimizer.apply_smote(X_train, y_train, method='adasyn')
    
    # Strategy 2: Hyperparameter tuning on resampled data
    print("\n" + "="*60)
    print("STEP 2: COMPREHENSIVE HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define more aggressive parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 50, 100],
        'penalty': ['l2', 'l1'],
        'solver': ['saga'],
        'class_weight': [None, 'balanced', {0: 1, 1: 1.5}, {0: 1, 1: 2}]
    }
    
    base_model = LogisticRegression(
        max_iter=3000,
        random_state=optimizer.random_state,
        n_jobs=-1,
        warm_start=False
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=optimizer.random_state)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    print(f"Testing {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['class_weight'])} combinations...")
    
    start_time = time.time()
    grid_search.fit(X_train_resampled, y_train_resampled)
    tuning_time = time.time() - start_time
    
    print(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_tuned_model = grid_search.best_estimator_
    
    # Strategy 3: Evaluate on test set
    print("\n" + "="*60)
    print("STEP 3: FINAL MODEL EVALUATION")
    print("="*60)
    
    test_metrics = optimizer.evaluate_model(best_tuned_model, X_test, y_test, "Final Tuned Model")
    
    # Strategy 4: Compare with baseline strategies
    print("\n" + "="*60)
    print("STEP 4: COMPARING WITH OTHER STRATEGIES")
    print("="*60)
    
    comparison_results = {}
    
    # Test other resampling methods with best params
    resampling_methods = {
        'ADASYN_Tuned': 'adasyn',
        'SMOTE_Tuned': 'smote',
        'BorderlineSMOTE_Tuned': 'borderline',
        'Baseline_Tuned': None
    }
    
    for strategy_name, method in resampling_methods.items():
        print(f"\nTesting: {strategy_name}")
        
        if method is not None:
            X_temp, y_temp = optimizer.apply_smote(X_train, y_train, method)
        else:
            X_temp, y_temp = X_train, y_train
        
        # Train with best parameters from grid search
        model = LogisticRegression(
            **grid_search.best_params_,
            max_iter=3000,
            random_state=optimizer.random_state,
            n_jobs=-1
        )
        
        start = time.time()
        model.fit(X_temp, y_temp)
        train_time = time.time() - start
        
        metrics = optimizer.evaluate_model(model, X_test, y_test, strategy_name)
        
        comparison_results[strategy_name] = {
            'model': model,
            'training_time': train_time,
            'test_metrics': metrics
        }
    
    # Create comprehensive summary
    summary_data = []
    for strategy_name, results in comparison_results.items():
        metrics = results['test_metrics']
        row = {
            'Strategy': strategy_name,
            'Training_Time': f"{results['training_time']:.2f}s",
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'F1_Score': metrics['f1_score'],
            'ROC_AUC': metrics['roc_auc']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1_Score', ascending=False)
    
    print("\n" + "="*60)
    print("FINAL PERFORMANCE COMPARISON (Sorted by F1-Score)")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Identify best model
    best_idx = summary_df['F1_Score'].idxmax()
    best_strategy_name = summary_df.loc[best_idx, 'Strategy']
    best_final_model = comparison_results[best_strategy_name]['model']
    
    print(f"\n{'='*60}")
    print(f"BEST FINAL MODEL: {best_strategy_name}")
    print(f"F1-Score: {summary_df.loc[best_idx, 'F1_Score']:.4f}")
    print(f"Recall: {summary_df.loc[best_idx, 'Recall']:.4f}")
    print(f"Precision: {summary_df.loc[best_idx, 'Precision']:.4f}")
    print(f"ROC-AUC: {summary_df.loc[best_idx, 'ROC_AUC']:.4f}")
    print(f"{'='*60}")
    
    # Save results
    print("\nSaving results...")
    summary_df.to_csv('data/processed/logistic_regression_optimized_results.csv', index=False)
    print("Results saved to: data/processed/logistic_regression_optimized_results.csv")
    
    # Save best model parameters
    params_df = pd.DataFrame([grid_search.best_params_])
    params_df.to_csv('data/processed/best_logistic_regression_params.csv', index=False)
    print("Best parameters saved to: data/processed/best_logistic_regression_params.csv")
    
    return optimizer, comparison_results, summary_df, best_final_model, grid_search.best_params_

if __name__ == "__main__":
    optimizer, results, summary_df, best_model, best_params = main()

