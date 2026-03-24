
"""
Heart Disease Prediction - Data Preprocessing Module
==================================================
This module handles data loading, cleaning, and preprocessing for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePreprocessor:
    """
    A comprehensive preprocessor for heart disease prediction data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = 'target'

    def load_data(self, file_path):
        """Load heart disease dataset from CSV file."""
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def explore_data(self, data):
        """Perform basic exploratory data analysis."""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\nColumn names: {list(data.columns)}")
        print(f"\nData types:\n{data.dtypes}")
        print(f"\nMissing values:\n{data.isnull().sum()}")
        print(f"\nTarget distribution:\n{data[self.target_column].value_counts()}")

        # Basic statistics
        print(f"\nNumerical features statistics:")
        print(data.describe())

        return data

    def clean_data(self, data):
        """Clean the dataset by handling missing values and outliers."""
        print("\n=== DATA CLEANING ===")

        # Handle missing values
        if data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # For numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                imputer_num = SimpleImputer(strategy='median')
                data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

            # For categorical columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

        # Remove duplicates
        initial_shape = data.shape[0]
        data = data.drop_duplicates()
        print(f"Removed {initial_shape - data.shape[0]} duplicate rows")

        return data

    def engineer_features(self, data):
        """Create new features and transform existing ones."""
        print("\n=== FEATURE ENGINEERING ===")

        # Create age groups
        if 'age' in data.columns:
            data['age_group'] = pd.cut(data['age'], 
                                     bins=[0, 40, 55, 70, 100], 
                                     labels=['Young', 'Middle', 'Senior', 'Elderly'])
            print("Created age groups")

        # Create BMI if height and weight available (example)
        # This would be dataset specific

        # Create risk score combinations
        if all(col in data.columns for col in ['chol', 'trestbps']):
            data['chol_bp_risk'] = (data['chol'] > 240) & (data['trestbps'] > 140)
            print("Created cholesterol-BP risk indicator")

        return data

    def encode_categorical_features(self, data):
        """Encode categorical features."""
        print("\n=== CATEGORICAL ENCODING ===")

        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]

        for col in categorical_cols:
            if data[col].nunique() <= 2:
                # Binary encoding
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                print(f"Label encoded: {col}")
            else:
                # One-hot encoding
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)
                print(f"One-hot encoded: {col}")

        return data

    def scale_features(self, X_train, X_test=None):
        """Scale numerical features."""
        print("\n=== FEATURE SCALING ===")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def split_data(self, data, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print("\n=== DATA SPLITTING ===")

        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Training target distribution:\n{y_train.value_counts()}")

        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(self, file_path, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline."""
        print("Starting Heart Disease Data Preprocessing Pipeline...")
        print("=" * 60)

        # Load data
        data = self.load_data(file_path)
        if data is None:
            return None

        # Explore data
        data = self.explore_data(data)

        # Clean data
        data = self.clean_data(data)

        # Engineer features
        data = self.engineer_features(data)

        # Encode categorical features
        data = self.encode_categorical_features(data)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(data, test_size, random_state)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train_scaled.columns.tolist(),
            'preprocessor': self
        }

# Usage example
if __name__ == "__main__":
    preprocessor = HeartDiseasePreprocessor()
    # result = preprocessor.preprocess_pipeline('data/raw/heart_disease.csv')
