
"""
Heart Disease Prediction - Model Training Module
===============================================
This module contains various machine learning models for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseModelTrainer:
    """
    Comprehensive model trainer for heart disease prediction.
    """

    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.results = {}

    def initialize_models(self):
        """Initialize all machine learning models with default parameters."""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'svm': SVC(random_state=42, probability=True),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'naive_bayes': GaussianNB(),
            'neural_network': MLPClassifier(random_state=42, max_iter=500),
            'adaboost': AdaBoostClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        print(f"Initialized {len(self.models)} machine learning models")

    def get_hyperparameter_grids(self):
        """Define hyperparameter grids for each model."""
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'adaboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        return param_grids

    def train_single_model(self, model_name, model, X_train, y_train, cv_folds=5):
        """Train a single model with cross-validation."""
        print(f"\nTraining {model_name}...")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                  scoring='accuracy')

        # Fit the model
        model.fit(X_train, y_train)

        # Store results
        self.results[model_name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        print(f"{model_name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return model

    def train_all_models(self, X_train, y_train, cv_folds=5):
        """Train all models with cross-validation."""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)

        self.initialize_models()

        for model_name, model in self.models.items():
            self.train_single_model(model_name, model, X_train, y_train, cv_folds)

        # Display results summary
        self.display_cv_results()

    def hyperparameter_tuning(self, X_train, y_train, model_names=None, cv_folds=3):
        """Perform hyperparameter tuning for specified models."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)

        if model_names is None:
            # Focus on best performing models for tuning
            top_models = sorted(self.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:5]
            model_names = [model[0] for model in top_models]

        param_grids = self.get_hyperparameter_grids()

        for model_name in model_names:
            if model_name in param_grids:
                print(f"\nTuning {model_name}...")

                model = self.models[model_name]
                param_grid = param_grids[model_name]

                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv_folds, scoring='accuracy',
                    n_jobs=-1, verbose=0
                )

                grid_search.fit(X_train, y_train)

                # Store best model
                self.best_models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'grid_search': grid_search
                }

                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test data."""
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST DATA")
        print("="*60)

        evaluation_results = {}

        # Evaluate base models
        for model_name, result in self.results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

        # Evaluate tuned models
        for model_name, result in self.best_models.items():
            model = result['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            evaluation_results[f"{model_name}_tuned"] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

        self.evaluation_results = evaluation_results
        self.display_evaluation_results()

        return evaluation_results

    def display_cv_results(self):
        """Display cross-validation results in a formatted table."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)

        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV_Mean': [result['cv_mean'] for result in self.results.values()],
            'CV_Std': [result['cv_std'] for result in self.results.values()]
        })

        results_df = results_df.sort_values('CV_Mean', ascending=False)
        print(results_df.to_string(index=False))

    def display_evaluation_results(self):
        """Display evaluation results in a formatted table."""
        print("\n" + "="*60)
        print("TEST SET EVALUATION RESULTS")
        print("="*60)

        results_list = []
        for model_name, result in self.evaluation_results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'AUC_Score': result['auc_score'] if result['auc_score'] is not None else 'N/A'
            })

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print(results_df.to_string(index=False))

    def get_best_model(self):
        """Get the best performing model based on test accuracy."""
        if not hasattr(self, 'evaluation_results'):
            print("Please run evaluate_models() first")
            return None

        best_model_name = max(self.evaluation_results.items(), 
                            key=lambda x: x[1]['accuracy'])[0]

        if '_tuned' in best_model_name:
            base_name = best_model_name.replace('_tuned', '')
            best_model = self.best_models[base_name]['model']
        else:
            best_model = self.results[best_model_name]['model']

        print(f"\nBest performing model: {best_model_name}")
        print(f"Test accuracy: {self.evaluation_results[best_model_name]['accuracy']:.4f}")

        return best_model, best_model_name

    def save_model(self, model, model_name, file_path=None):
        """Save trained model to disk."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"models/trained_models/{model_name}_{timestamp}.joblib"

        joblib.dump(model, file_path)
        print(f"Model saved to: {file_path}")

        return file_path

    def load_model(self, file_path):
        """Load trained model from disk."""
        model = joblib.load(file_path)
        print(f"Model loaded from: {file_path}")
        return model

# Usage example
if __name__ == "__main__":
    # This would be used with preprocessed data
    trainer = HeartDiseaseModelTrainer()
    # trainer.train_all_models(X_train, y_train)
    # trainer.hyperparameter_tuning(X_train, y_train)
    # trainer.evaluate_models(X_test, y_test)
    # best_model, best_name = trainer.get_best_model()
