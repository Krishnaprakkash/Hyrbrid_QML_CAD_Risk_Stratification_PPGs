# src/model_visualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os

class ModelVisualizer:
    """
    Comprehensive visualization for classical ML model evaluation.
    """
    
    def __init__(self, save_dir='visualizations'):
        """
        Initialize visualizer.
        
        Args:
            save_dir (str): Directory to save visualization files
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name, normalize=False):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            normalize (bool): Whether to normalize values
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=['Normal', 'CAD'],
                    yticklabels=['Normal', 'CAD'],
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
    def plot_roc_curves(self, models_dict, X_test, y_test):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_dict (dict): Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test labels
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Could not plot ROC for {model_name}: {e}")
                continue
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/roc_curves_all_models.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
    def plot_precision_recall_curves(self, models_dict, X_test, y_test):
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_dict (dict): Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test labels
        """
        plt.figure(figsize=(10, 8))
        
        baseline_precision = np.sum(y_test) / len(y_test)
        
        for model_name, model in models_dict.items():
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, lw=2,
                        label=f'{model_name} (AUC = {pr_auc:.3f})')
            except Exception as e:
                print(f"Could not plot PR curve for {model_name}: {e}")
                continue
        
        # Plot baseline
        plt.plot([0, 1], [baseline_precision, baseline_precision], 
                'k--', lw=2, label=f'Baseline (Prevalence = {baseline_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/precision_recall_curves.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
    def plot_metrics_comparison(self, results_df):
        """
        Plot bar chart comparing multiple metrics across models.
        
        Args:
            results_df (pd.DataFrame): DataFrame with model results
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'ROC_AUC']
        
        # Filter only numeric columns that exist
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            data = results_df.sort_values(metric, ascending=False)
            
            bars = ax.barh(data['Strategy'], data[metric], color='skyblue', edgecolor='navy')
            
            # Color the best bar
            bars[0].set_color('lightcoral')
            
            ax.set_xlabel('Score', fontsize=11)
            ax.set_title(f'{metric.replace("_", " ")}', fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1.0])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, data[metric])):
                ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Model Performance Comparison - All Metrics', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/metrics_comparison_all.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plot feature importance for models that support it.
        
        Args:
            model: Trained model with coef_ attribute
            feature_names (list): List of feature names
            top_n (int): Number of top features to display
        """
        if not hasattr(model, 'coef_'):
            print("Model does not support feature importance visualization")
            return
        
        # Get coefficients
        coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coef)
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='teal', edgecolor='black')
        
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/feature_importance_top{top_n}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
    def plot_training_time_comparison(self, results_df):
        """
        Plot training time comparison.
        
        Args:
            results_df (pd.DataFrame): DataFrame with training times
        """
        if 'Training_Time' not in results_df.columns:
            print("Training_Time column not found")
            return
        
        # Extract numeric values from Training_Time (remove 's')
        training_times = results_df['Training_Time'].str.replace('s', '').astype(float)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results_df['Strategy'], training_times, 
                      color='lightgreen', edgecolor='darkgreen', width=0.6)
        
        # Highlight fastest
        min_idx = training_times.idxmin()
        bars[min_idx].set_color('gold')
        
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.xlabel('Model Strategy', fontsize=12)
        plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, time in zip(bars, training_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        filename = f"{self.save_dir}/training_time_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
    def plot_recall_precision_tradeoff(self, results_df):
        """
        Scatter plot showing recall vs precision trade-off.
        
        Args:
            results_df (pd.DataFrame): DataFrame with results
        """
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(results_df['Recall'], results_df['Precision'],
                            s=results_df['F1_Score']*500, 
                            c=results_df['ROC_AUC'], cmap='viridis',
                            alpha=0.6, edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for idx, row in results_df.iterrows():
            plt.annotate(row['Strategy'], 
                        (row['Recall'], row['Precision']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (PPV)', fontsize=12)
        plt.title('Recall vs Precision Trade-off\n(Bubble size = F1-Score, Color = ROC-AUC)', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='ROC-AUC Score')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        filename = f"{self.save_dir}/recall_precision_tradeoff.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


# Main function to generate all visualizations
def main():
    """
    Generate all visualizations for classical ML models.
    """
    print("="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    results_df = pd.read_csv('data/processed/logistic_regression_optimized_results.csv')
    print(f"Loaded {len(results_df)} model results")
    
    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir='visualizations')
    
    # Load data and models (you'll need to retrain or load models)
    from data_preparation import DataPreparator
    from optimized_logistic_regression import OptimizedLogisticRegression
    
    print("\nPreparing data...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='robust',
        feature_selection=False,
        k_features=40
    )
    
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    
    # Retrain models for visualization
    print("\nRetraining models for visualization...")
    optimizer = OptimizedLogisticRegression()
    
    models_dict = {}
    
    # Train each strategy
    for strategy in results_df['Strategy']:
        print(f"Training {strategy}...")
        
        if 'ADASYN' in strategy:
            X_temp, y_temp = optimizer.apply_smote(X_train, y_train, 'adasyn')
        elif 'SMOTE' in strategy and 'Borderline' in strategy:
            X_temp, y_temp = optimizer.apply_smote(X_train, y_train, 'borderline')
        elif 'SMOTE' in strategy:
            X_temp, y_temp = optimizer.apply_smote(X_train, y_train, 'smote')
        else:
            X_temp, y_temp = X_train, y_train
        
        model = optimizer.create_baseline_model(class_weight='balanced')
        model.fit(X_temp, y_temp)
        models_dict[strategy] = model
    
    # Generate visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Confusion matrices for each model
    print("\n1. Generating confusion matrices...")
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        visualizer.plot_confusion_matrix(y_test, y_pred, model_name)
    
    # 2. ROC curves
    print("\n2. Generating ROC curves...")
    visualizer.plot_roc_curves(models_dict, X_test, y_test)
    
    # 3. Precision-Recall curves
    print("\n3. Generating Precision-Recall curves...")
    visualizer.plot_precision_recall_curves(models_dict, X_test, y_test)
    
    # 4. Metrics comparison
    print("\n4. Generating metrics comparison...")
    visualizer.plot_metrics_comparison(results_df)
    
    # 5. Training time comparison
    print("\n5. Generating training time comparison...")
    visualizer.plot_training_time_comparison(results_df)
    
    # 6. Recall-Precision tradeoff
    print("\n6. Generating recall-precision tradeoff plot...")
    visualizer.plot_recall_precision_tradeoff(results_df)
    
    # 7. Feature importance for best model
    print("\n7. Generating feature importance...")
    best_model = models_dict[results_df.iloc[0]['Strategy']]
    visualizer.plot_feature_importance(best_model, prepared_data['feature_names'], top_n=20)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED")
    print(f"Visualizations saved in: visualizations/")
    print("="*60)
    
    return visualizer, models_dict

if __name__ == "__main__":
    visualizer, models = main()
