# src/visualize_all_classical_models.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import os

class ClassicalModelVisualizer:
    """
    Comprehensive visualization for all classical ML models.
    """
    
    def __init__(self, save_dir='visualizations/classical_models'):
        """
        Initialize visualizer.
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print(f"Visualizer initialized. Saving to: {save_dir}")
    
    def plot_all_confusion_matrices(self, models_dict, X_test, y_test):
        """
        Plot confusion matrices for all models in a grid.
        
        Args:
            models_dict (dict): Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test labels
        """
        n_models = len(models_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel() if n_models > 1 else [axes]
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            ax = axes[idx]
            
            # Get predictions
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'CAD'],
                       yticklabels=['Normal', 'CAD'],
                       cbar_kws={'label': 'Count'})
            
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11)
            ax.set_xlabel('Predicted Label', fontsize=11)
            
            # Add accuracy to title
            accuracy = np.trace(cm) / np.sum(cm)
            ax.text(0.5, 1.08, f'Accuracy: {accuracy:.3f}', 
                   ha='center', transform=ax.transAxes, fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - All Classical ML Models', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/all_confusion_matrices.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_roc_curves_all(self, models_dict, X_test, y_test):
        """
        Plot ROC curves for all models on one chart.
        
        Args:
            models_dict (dict): Dictionary of models
            X_test: Test features
            y_test: Test labels
        """
        plt.figure(figsize=(12, 9))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2.5, color=colors[idx],
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Could not plot ROC for {model_name}: {e}")
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('ROC Curves - All Classical ML Models', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/roc_curves_all_models.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_precision_recall_all(self, models_dict, X_test, y_test):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            models_dict (dict): Dictionary of models
            X_test: Test features
            y_test: Test labels
        """
        plt.figure(figsize=(12, 9))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
        baseline = np.sum(y_test) / len(y_test)
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, lw=2.5, color=colors[idx],
                        label=f'{model_name} (AUC = {pr_auc:.3f})')
            except Exception as e:
                print(f"Could not plot PR curve for {model_name}: {e}")
        
        # Baseline
        plt.plot([0, 1], [baseline, baseline], 'k--', lw=2,
                label=f'Baseline (Prevalence = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
        plt.ylabel('Precision (PPV)', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curves - All Classical ML Models', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="best", fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/precision_recall_all_models.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_metrics_comparison_detailed(self, results_dict):
        """
        Detailed metrics comparison with multiple subplots.
        
        Args:
            results_dict (dict): Dictionary with model results
        """
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes = axes.ravel()
        
        model_names = list(results_dict.keys())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [results_dict[model]['test_metrics'][metric] 
                     for model in model_names]
            
            # Sort by value
            sorted_indices = np.argsort(values)[::-1]
            sorted_names = [model_names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # Create bar chart
            bars = ax.barh(sorted_names, sorted_values, color='steelblue', edgecolor='navy', linewidth=1.5)
            
            # Highlight best
            bars[0].set_color('coral')
            bars[0].set_edgecolor('darkred')
            
            ax.set_xlabel('Score', fontsize=11, fontweight='bold')
            ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.set_xlim([0, 1.0])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, sorted_values)):
                ax.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('Detailed Performance Metrics - All Classical ML Models',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/detailed_metrics_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_training_time_vs_performance(self, results_dict):
        """
        Scatter plot: Training time vs F1-Score.
        
        Args:
            results_dict (dict): Dictionary with model results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        model_names = list(results_dict.keys())
        training_times = [results_dict[model]['training_time'] for model in model_names]
        f1_scores = [results_dict[model]['test_metrics']['f1_score'] for model in model_names]
        accuracies = [results_dict[model]['test_metrics']['accuracy'] for model in model_names]
        
        # Plot 1: Time vs F1-Score
        scatter1 = ax1.scatter(training_times, f1_scores, s=200, c=f1_scores,
                              cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, name in enumerate(model_names):
            ax1.annotate(name, (training_times[i], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax1.set_title('Training Time vs F1-Score', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='F1-Score')
        
        # Plot 2: Time vs Accuracy
        scatter2 = ax2.scatter(training_times, accuracies, s=200, c=accuracies,
                              cmap='plasma', alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, name in enumerate(model_names):
            ax2.annotate(name, (training_times[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Accuracy')
        
        plt.suptitle('Efficiency Analysis - Classical ML Models', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{self.save_dir}/training_efficiency.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_recall_precision_scatter(self, results_dict):
        """
        Scatter plot showing recall vs precision trade-off.
        
        Args:
            results_dict (dict): Dictionary with results
        """
        plt.figure(figsize=(12, 10))
        
        model_names = list(results_dict.keys())
        recalls = [results_dict[model]['test_metrics']['recall'] for model in model_names]
        precisions = [results_dict[model]['test_metrics']['precision'] for model in model_names]
        f1_scores = [results_dict[model]['test_metrics']['f1_score'] for model in model_names]
        roc_aucs = [results_dict[model]['test_metrics']['roc_auc'] for model in model_names]
        
        scatter = plt.scatter(recalls, precisions, s=np.array(f1_scores)*800,
                            c=roc_aucs, cmap='coolwarm', alpha=0.7,
                            edgecolors='black', linewidth=2)
        
        for i, name in enumerate(model_names):
            plt.annotate(name, (recalls[i], precisions[i]),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
        plt.ylabel('Precision (PPV)', fontsize=13, fontweight='bold')
        plt.title('Recall vs Precision Trade-off\n(Bubble size = F1-Score, Color = ROC-AUC)',
                 fontsize=15, fontweight='bold')
        plt.colorbar(scatter, label='ROC-AUC Score')
        plt.grid(True, alpha=0.3)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add diagonal F1 contours
        x = np.linspace(0, 1, 100)
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            y = (f1 * x) / (2 * x - f1)
            y = np.where((y >= 0) & (y <= 1), y, np.nan)
            plt.plot(x, y, 'k--', alpha=0.2, linewidth=1)
            plt.text(0.9, (f1 * 0.9) / (2 * 0.9 - f1), f'F1={f1}',
                    fontsize=9, alpha=0.5)
        
        plt.tight_layout()
        
        filename = f"{self.save_dir}/recall_precision_tradeoff.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_summary_table(self, results_dict):
        """
        Create and save a comprehensive summary table.
        
        Args:
            results_dict (dict): Dictionary with results
            
        Returns:
            pd.DataFrame: Summary dataframe
        """
        summary_data = []
        
        for model_name, results in results_dict.items():
            metrics = results['test_metrics']
            
            row = {
                'Model': model_name,
                'Training_Time_s': f"{results['training_time']:.3f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}",
                'ROC_AUC': f"{metrics['roc_auc']:.4f}"
            }
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('F1_Score', ascending=False)
        
        # Save as CSV
        csv_filename = f"{self.save_dir}/summary_table.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(14, len(df)*0.6 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colColours=['lightblue']*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Highlight best values
        for i in range(1, len(df.columns)):
            table[(0, i)].set_facecolor('steelblue')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Summary - All Classical ML Models',
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filename = f"{self.save_dir}/summary_table.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
        return df


def main():
    """
    Generate all visualizations for classical ML models.
    """
    print("="*60)
    print("VISUALIZING ALL CLASSICAL ML MODELS")
    print("="*60)
    
    # Load data and retrain models
    from data_preparation import DataPreparator
    from classical_ml_models import ClassicalMLModels
    
    print("\nPreparing data...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data()
    
    print("\nRetraining all classical ML models...")
    ml_models = ClassicalMLModels()
    ml_models.initialize_models()
    
    results = ml_models.train_and_evaluate_all(
        X_train=prepared_data['X_train'],
        y_train=prepared_data['y_train'],
        X_test=prepared_data['X_test'],
        y_test=prepared_data['y_test'],
        perform_cv=False  # Skip CV for faster visualization
    )
    
    # Initialize visualizer
    visualizer = ClassicalModelVisualizer()
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Confusion matrices
    print("\n1. Generating confusion matrices...")
    visualizer.plot_all_confusion_matrices(
        ml_models.trained_models,
        prepared_data['X_test'],
        prepared_data['y_test']
    )
    
    # 2. ROC curves
    print("\n2. Generating ROC curves...")
    visualizer.plot_roc_curves_all(
        ml_models.trained_models,
        prepared_data['X_test'],
        prepared_data['y_test']
    )
    
    # 3. Precision-Recall curves
    print("\n3. Generating Precision-Recall curves...")
    visualizer.plot_precision_recall_all(
        ml_models.trained_models,
        prepared_data['X_test'],
        prepared_data['y_test']
    )
    
    # 4. Detailed metrics comparison
    print("\n4. Generating detailed metrics comparison...")
    visualizer.plot_metrics_comparison_detailed(results)
    
    # 5. Training efficiency
    print("\n5. Generating training efficiency analysis...")
    visualizer.plot_training_time_vs_performance(results)
    
    # 6. Recall-Precision scatter
    print("\n6. Generating recall-precision trade-off...")
    visualizer.plot_recall_precision_scatter(results)
    
    # 7. Summary table
    print("\n7. Creating summary table...")
    summary_df = visualizer.create_summary_table(results)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("="*60)
    print(f"\nVisualizations saved in: {visualizer.save_dir}/")
    print(f"Total files generated: 7")
    
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    return visualizer, summary_df

if __name__ == "__main__":
    visualizer, summary = main()
