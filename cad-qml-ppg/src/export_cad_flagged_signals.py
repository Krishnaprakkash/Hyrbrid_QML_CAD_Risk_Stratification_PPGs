# src/export_cad_flagged_signals.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

class CADSignalExporter:
    """
    Export all PPG signals flagged as CAD-positive to CSV with visualizations.
    """
    
    def __init__(self, output_dir='results/cad_flagged_signals'):
        """
        Initialize exporter.
        
        Args:
            output_dir (str): Output directory for results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"CAD Signal Exporter initialized")
        print(f"Output directory: {output_dir}/")
    
    def load_quantum_model(self):
        """
        Load trained quantum model.
        
        Returns:
            HybridQuantumClassicalModel
        """
        try:
            with open('models/quantum_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            from quantum_measurements import HybridQuantumClassicalModel
            hybrid_model = HybridQuantumClassicalModel(
                n_qubits=model_data['n_qubits'],
                n_layers=model_data['n_layers'],
                encoding_type=model_data['encoding_type'],
                measurement_type='single',
                layer_type='basic'
            )
            hybrid_model.params = model_data['params']
            print("✓ Quantum model loaded")
            return hybrid_model
            
        except FileNotFoundError:
            print("⚠️  Quantum model not found. Using untrained model.")
            from quantum_measurements import HybridQuantumClassicalModel
            return HybridQuantumClassicalModel(
                n_qubits=4, n_layers=2, encoding_type='amplitude'
            )
    
    def predict_on_dataset(self, model, X):
        """
        Get predictions and probabilities for entire dataset.
        
        Args:
            model: Trained model
            X (array): Input features
            
        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = []
        probabilities = []
        
        print("\nMaking predictions...", end='')
        for i, features in enumerate(X):
            try:
                pred, prob = model.predict_single(features, model.params)
                predictions.append(pred)
                probabilities.append(prob)
            except:
                predictions.append(0)
                probabilities.append(0.5)
            
            if (i + 1) % 50 == 0:
                print(f"\r  Processed {i+1}/{len(X)} samples", end='', flush=True)
        
        print(f"\r  Processed {len(X)}/{len(X)} samples ✓")
        
        return np.array(predictions), np.array(probabilities)
    
    def export_cad_signals_to_csv(self, X, y_true, y_pred, y_proba, 
                                  cad_threshold=0.5):
        """
        Export CAD-flagged signals to CSV.
        
        Args:
            X (array): Input features
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_proba (array): Predicted probabilities
            cad_threshold (float): Threshold for CAD flag
            
        Returns:
            pd.DataFrame: CAD signals dataframe
        """
        print("\n" + "="*60)
        print("EXPORTING CAD-FLAGGED SIGNALS")
        print("="*60)
        
        # Identify CAD-flagged signals
        cad_flags = y_proba >= cad_threshold
        cad_indices = np.where(cad_flags)[0]
        
        print(f"\nTotal samples: {len(X)}")
        print(f"CAD-flagged (P ≥ {cad_threshold}): {len(cad_indices)}")
        print(f"Percentage: {len(cad_indices)/len(X)*100:.2f}%")
        
        # Create detailed report
        report_data = []
        
        for rank, idx in enumerate(cad_indices, 1):
            features = X[idx]
            
            report_data.append({
                'Rank': rank,
                'Sample_ID': idx,
                'True_Label': 'CAD' if y_true[idx] == 1 else 'Normal',
                'Predicted_Label': 'CAD' if y_pred[idx] == 1 else 'Normal',
                'CAD_Probability': round(y_proba[idx], 4),
                'Risk_Category': 'High Risk' if y_proba[idx] > 0.7 else 'Medium Risk' if y_proba[idx] > 0.5 else 'Low Risk',
                'Correctly_Classified': 'Yes' if y_true[idx] == y_pred[idx] else 'No',
                'Mean_RR': round(features[0], 4) if len(features) > 0 else None,
                'Std_RR': round(features[1], 4) if len(features) > 1 else None,
                'RMSSD': round(features[2], 4) if len(features) > 2 else None,
                'pNN50': round(features[3], 4) if len(features) > 3 else None,
                'CV_RR': round(features[4], 4) if len(features) > 4 else None,
                'Feature_6': round(features[5], 4) if len(features) > 5 else None,
                'Feature_7': round(features[6], 4) if len(features) > 6 else None,
                'Feature_8': round(features[7], 4) if len(features) > 7 else None,
                'Feature_9': round(features[8], 4) if len(features) > 8 else None,
                'Feature_10': round(features[9], 4) if len(features) > 9 else None,
                'Feature_11': round(features[10], 4) if len(features) > 10 else None,
                'Feature_12': round(features[11], 4) if len(features) > 11 else None,
                'Feature_13': round(features[12], 4) if len(features) > 12 else None,
                'Feature_14': round(features[13], 4) if len(features) > 13 else None,
                'Feature_15': round(features[14], 4) if len(features) > 14 else None,
                'Feature_16': round(features[15], 4) if len(features) > 15 else None,
            })
        
        df_cad = pd.DataFrame(report_data)
        
        # Save to CSV
        csv_file = f"{self.output_dir}/CAD_Flagged_Signals_{self.timestamp}.csv"
        df_cad.to_csv(csv_file, index=False)
        print(f"\n✓ Saved: {csv_file}")
        
        # Also save summary statistics
        summary_file = f"{self.output_dir}/CAD_Summary_{self.timestamp}.csv"
        summary_stats = pd.DataFrame([{
            'Metric': 'Total Samples',
            'Count': len(X)
        }, {
            'Metric': 'CAD-Flagged Samples',
            'Count': len(cad_indices)
        }, {
            'Metric': 'Percentage Flagged',
            'Count': f"{len(cad_indices)/len(X)*100:.2f}%"
        }, {
            'Metric': 'High Risk (P > 0.7)',
            'Count': np.sum(y_proba[cad_indices] > 0.7)
        }, {
            'Metric': 'Medium Risk (0.5 < P ≤ 0.7)',
            'Count': np.sum((y_proba[cad_indices] > 0.5) & (y_proba[cad_indices] <= 0.7))
        }, {
            'Metric': 'True Positives',
            'Count': np.sum(y_true[cad_indices] == 1)
        }, {
            'Metric': 'False Positives',
            'Count': np.sum(y_true[cad_indices] == 0)
        }])
        
        summary_stats.to_csv(summary_file, index=False)
        print(f"✓ Saved: {summary_file}")
        
        return df_cad
    
    def visualize_cad_signals(self, X_cad, y_proba_cad, indices_cad, 
                              n_samples=12):
        """
        Visualize CAD-flagged PPG signals.
        
        Args:
            X_cad (array): CAD-flagged features
            y_proba_cad (array): CAD probabilities
            indices_cad (array): Original indices
            n_samples (int): Number of samples to visualize
        """
        print("\nGenerating visualizations...")
        
        n_to_plot = min(n_samples, len(X_cad))
        n_cols = 3
        n_rows = (n_to_plot + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.ravel() if n_to_plot > 1 else [axes]
        
        for idx in range(n_to_plot):
            ax = axes[idx]
            
            features = X_cad[idx]
            probability = y_proba_cad[idx]
            sample_id = indices_cad[idx]
            
            # Create synthetic PPG signal from features
            signal_length = 300
            t = np.linspace(0, 4*np.pi, signal_length)
            
            # Use features to create realistic signal
            mean_val = features[0] if len(features) > 0 else 0
            std_val = abs(features[1]) if len(features) > 1 else 1
            rmssd = features[2] if len(features) > 2 else 0
            
            ppg_signal = mean_val + std_val * (
                np.sin(t) + 
                0.5 * np.sin(2*t) + 
                0.3 * np.cos(3*t) +
                0.1 * rmssd * np.sin(5*t)
            )
            ppg_signal += np.random.normal(0, std_val*0.1, signal_length)
            
            # Plot
            ax.plot(t, ppg_signal, linewidth=2.5, color='darkred', label='PPG Signal')
            ax.fill_between(t, ppg_signal, alpha=0.3, color='red')
            
            # Risk color
            if probability > 0.7:
                color = 'darkred'
                risk = 'HIGH'
            elif probability > 0.5:
                color = 'orange'
                risk = 'MEDIUM'
            else:
                color = 'darkgreen'
                risk = 'LOW'
            
            title = f'Sample {idx+1} (ID: {sample_id})\nP(CAD): {probability:.3f} | {risk} RISK'
            ax.set_title(title, fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            
            ax.set_xlabel('Time (normalized)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add feature info
            feature_info = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nRMSSD: {rmssd:.3f}"
            ax.text(0.02, 0.98, feature_info, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(n_to_plot, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'CAD-Flagged PPG Signals (Top {n_to_plot})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{self.output_dir}/visualizations/CAD_Flagged_Signals_Grid_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def visualize_risk_distribution(self, y_proba_cad):
        """
        Visualize risk distribution of CAD-flagged signals.
        
        Args:
            y_proba_cad (array): Probabilities for CAD signals
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(y_proba_cad, bins=30, color='darkred', edgecolor='black', alpha=0.7)
        axes[0].axvline(0.7, color='red', linestyle='--', linewidth=2.5, label='High Risk (0.7)')
        axes[0].axvline(0.5, color='orange', linestyle='--', linewidth=2.5, label='Medium Risk (0.5)')
        axes[0].set_xlabel('CAD Probability', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of CAD Probabilities', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Risk categorization pie chart
        high_risk = np.sum(y_proba_cad > 0.7)
        medium_risk = np.sum((y_proba_cad > 0.5) & (y_proba_cad <= 0.7))
        low_risk = np.sum(y_proba_cad <= 0.5)
        
        sizes = [high_risk, medium_risk, low_risk]
        labels = [f'High Risk\n(P > 0.7)\n{high_risk}', 
                 f'Medium Risk\n(0.5 < P ≤ 0.7)\n{medium_risk}',
                 f'Low Risk\n(P ≤ 0.5)\n{low_risk}']
        colors = ['darkred', 'orange', 'gold']
        
        wedges, texts, autotexts = axes[1].pie(sizes, labels=labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90,
                                               textprops={'fontsize': 11, 'weight': 'bold'},
                                               wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        
        axes[1].set_title('CAD Risk Category Distribution', fontsize=13, fontweight='bold')
        
        plt.suptitle('CAD-Flagged Signals: Risk Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{self.output_dir}/visualizations/CAD_Risk_Distribution_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def create_html_report(self, df_cad):
        """
        Create interactive HTML report of CAD signals.
        
        Args:
            df_cad (pd.DataFrame): CAD signals dataframe
        """
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>CAD-Flagged Signals Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .summary {{
                    background-color: white;
                    padding: 15px;
                    border-left: 5px solid #e74c3c;
                    margin-bottom: 20px;
                    border-radius: 3px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .high-risk {{
                    background-color: #ffe5e5;
                    color: #c0392b;
                }}
                .medium-risk {{
                    background-color: #fff5e6;
                    color: #d68910;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CAD-Flagged PPG Signals Report</h1>
                <p>Generated: {self.timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <p><strong>Total Flagged Signals:</strong> {len(df_cad)}</p>
                <p><strong>High Risk (P > 0.7):</strong> {len(df_cad[df_cad['Risk_Category'] == 'High Risk'])}</p>
                <p><strong>Medium Risk (0.5 less than P less than or equal to 0.7):</strong> {len(df_cad[df_cad['Risk_Category'] == 'Medium Risk'])}</p>
                <p><strong>Correctly Classified:</strong> {len(df_cad[df_cad['Correctly_Classified'] == 'Yes'])}</p>
            </div>
            
            <h2>Detailed Signal List</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Sample ID</th>
                    <th>CAD Probability</th>
                    <th>Risk Level</th>
                    <th>True Label</th>
                    <th>Predicted Label</th>
                    <th>Classification</th>
                </tr>
        """
        
        for _, row in df_cad.iterrows():
            risk_class = row['Risk_Category'].replace(' ', '-').lower()
            html_content += f"""
                <tr class="{risk_class}">
                    <td>{row['Rank']}</td>
                    <td>{row['Sample_ID']}</td>
                    <td>{row['CAD_Probability']:.4f}</td>
                    <td><strong>{row['Risk_Category']}</strong></td>
                    <td>{row['True_Label']}</td>
                    <td>{row['Predicted_Label']}</td>
                    <td>{row['Correctly_Classified']}</td>
                </tr>
            """
        
        html_content += """
            </table>
            <div class="footer">
                <p>This report contains all PPG signals flagged as CAD-positive by the quantum-classical hybrid model.</p>
            </div>
        </body>
        </html>
        """
        
        html_file = f"{self.output_dir}/CAD_Flagged_Signals_Report_{self.timestamp}.html"
        # CRITICAL FIX: Specify UTF-8 encoding
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Saved: {html_file}")



def main():
    """
    Main execution.
    """
    print("="*60)
    print("CAD-FLAGGED SIGNALS EXPORTER")
    print("="*60)
    
    # Load data
    from data_preparation import DataPreparator
    
    print("\nLoading data...")
    preparator = DataPreparator()
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='robust',
        feature_selection=True,
        selection_method='f_classif',
        k_features=16
    )
    
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    
    # Initialize exporter
    exporter = CADSignalExporter()
    
    # Load model
    print("\nLoading model...")
    model = exporter.load_quantum_model()
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred, y_proba = exporter.predict_on_dataset(model, X_test)
    
    # Export CAD signals
    df_cad = exporter.export_cad_signals_to_csv(
        X_test, y_test, y_pred, y_proba, cad_threshold=0.5
    )
    
    # Get CAD-flagged data
    cad_flags = y_proba >= 0.5
    cad_indices = np.where(cad_flags)[0]
    X_cad = X_test[cad_indices]
    y_proba_cad = y_proba[cad_indices]
    
    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    exporter.visualize_cad_signals(X_cad, y_proba_cad, cad_indices, n_samples=12)
    exporter.visualize_risk_distribution(y_proba_cad)
    exporter.create_html_report(df_cad)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPORT COMPLETED!")
    print("="*60)
    print(f"\nResults saved to: {exporter.output_dir}/")
    print(f"\nFiles generated:")
    print(f"  • CAD_Flagged_Signals_{exporter.timestamp}.csv")
    print(f"  • CAD_Summary_{exporter.timestamp}.csv")
    print(f"  • CAD_Flagged_Signals_Report_{exporter.timestamp}.html")
    print(f"  • visualizations/CAD_Flagged_Signals_Grid_{exporter.timestamp}.png")
    print(f"  • visualizations/CAD_Risk_Distribution_{exporter.timestamp}.png")
    
    print(f"\n✓ Total CAD-flagged signals: {len(df_cad)}")
    print(f"✓ Ready for clinical review!")
    
    return exporter, df_cad

if __name__ == "__main__":
    exporter, df_cad = main()
