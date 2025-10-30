import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import os

class DataPreparator:
    """
    Handles data loading, label generation, and preprocessing for classical ML models.
    """
    
    def __init__(self, features_path="data/processed/ppg_features.csv", random_state=42):
        """
        Initialize the data preparator.
        
        Args:
            features_path (str): Path to the processed features CSV file
            random_state (int): Random state for reproducibility
        """
        self.features_path = features_path
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        
    def load_features(self):
        """
        Load the processed PPG features from CSV file.
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found at {self.features_path}")
            
        features_df = pd.read_csv(self.features_path)
        
        # Check if first column is non-numeric (patient ID or record name)
        first_col = features_df.columns[0]
        if features_df[first_col].dtype == 'object' or features_df[first_col].dtype == 'string':
            print(f"Detected non-numeric column '{first_col}' - removing from features")
            # Remove the first column (assumed to be ID/name column)
            features_df = features_df.drop(columns=[first_col])
        
        # Ensure all remaining columns are numeric
        non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"Removing non-numeric columns: {non_numeric_cols}")
            features_df = features_df.drop(columns=non_numeric_cols)
        
        print(f"Loaded features: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        # Store feature names for later use
        self.feature_names = features_df.columns.tolist()
        
        return features_df

    def generate_synthetic_labels(self, n_samples, cad_prevalence=0.3):
        """
        Generate synthetic CAD labels for binary classification.
        
        Args:
            n_samples (int): Number of samples
            cad_prevalence (float): Proportion of positive CAD cases
            
        Returns:
            np.ndarray: Binary labels (0: No CAD, 1: CAD)
        """
        np.random.seed(self.random_state)
        
        # Generate labels with specified prevalence
        n_positive = int(n_samples * cad_prevalence)
        labels = np.concatenate([
            np.ones(n_positive),
            np.zeros(n_samples - n_positive)
        ])
        
        # Shuffle the labels
        np.random.shuffle(labels)
        
        print(f"Generated labels: {n_positive} CAD cases ({cad_prevalence:.1%}), "
              f"{n_samples - n_positive} normal cases ({1-cad_prevalence:.1%})")
        
        return labels.astype(int)
    
    def create_train_val_test_split(self, X, y, test_size=0.15, val_size=0.15):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            stratify=y_temp, random_state=self.random_state
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {X_train.shape[0]} samples ({len(X_train)/len(X):.1%})")
        print(f"  Validation set: {X_val.shape[0]} samples ({len(X_val)/len(X):.1%})")
        print(f"  Test set: {X_test.shape[0]} samples ({len(X_test)/len(X):.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test, method='standard'):
        """
        Scale features using specified method.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            method (str): 'standard' or 'robust' scaling
            
        Returns:
            tuple: Scaled feature matrices
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Features scaled using {method} scaling")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def select_features(self, X_train, y_train, X_val, X_test, 
                       method='f_classif', k=30):
        """
        Perform feature selection.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train: Training labels
            method (str): 'f_classif' or 'mutual_info'
            k (int): Number of features to select
            
        Returns:
            tuple: Feature-selected matrices and selected feature indices
        """
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Method must be 'f_classif' or 'mutual_info'")
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        
        # Fit selector on training data
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature indices and names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Feature selection completed using {method}")
        print(f"Selected {k} features out of {len(self.feature_names)}")
        print(f"Selected features: {selected_features[:5]}..." if len(selected_features) > 5 
              else f"Selected features: {selected_features}")
        
        return X_train_selected, X_val_selected, X_test_selected, selected_indices
    
    def prepare_data(self, cad_prevalence=0.3, scaling_method='standard', 
                    feature_selection=True, selection_method='f_classif', k_features=30):
        """
        Complete data preparation pipeline.
        
        Args:
            cad_prevalence (float): Proportion of CAD cases
            scaling_method (str): Feature scaling method
            feature_selection (bool): Whether to perform feature selection
            selection_method (str): Feature selection method
            k_features (int): Number of features to select
            
        Returns:
            dict: Dictionary containing all prepared datasets and metadata
        """
        print("="*60)
        print("STARTING DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Load features
        features_df = self.load_features()
        X = features_df.values
        
        # Generate synthetic labels
        y = self.generate_synthetic_labels(len(features_df), cad_prevalence)
        
        # Create train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_split(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test, method=scaling_method
        )
        
        # Feature selection (optional)
        if feature_selection:
            X_train_final, X_val_final, X_test_final, selected_indices = self.select_features(
                X_train_scaled, y_train, X_val_scaled, X_test_scaled,
                method=selection_method, k=k_features
            )
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
        else:
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
            X_test_final = X_test_scaled
            selected_indices = None
            selected_feature_names = self.feature_names
        
        print("="*60)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return {
            'X_train': X_train_final,
            'X_val': X_val_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': selected_feature_names,
            'selected_indices': selected_indices,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'data_info': {
                'n_samples': len(features_df),
                'n_features_original': len(self.feature_names),
                'n_features_selected': len(selected_feature_names),
                'cad_prevalence': cad_prevalence,
                'scaling_method': scaling_method,
                'feature_selection': feature_selection
            }
        }

# Example usage function
def main():
    """
    Example usage of the DataPreparator class.
    """
    # Initialize data preparator
    preparator = DataPreparator()
    
    # Prepare data with default settings
    prepared_data = preparator.prepare_data(
        cad_prevalence=0.3,
        scaling_method='standard',
        feature_selection=True,
        selection_method='f_classif',
        k_features=30
    )
    
    # Print summary
    info = prepared_data['data_info']
    print(f"\nData preparation summary:")
    print(f"Total samples: {info['n_samples']}")
    print(f"Original features: {info['n_features_original']}")
    print(f"Selected features: {info['n_features_selected']}")
    print(f"CAD prevalence: {info['cad_prevalence']:.1%}")
    
    return prepared_data

if __name__ == "__main__":
    prepared_data = main()
