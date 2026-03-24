
"""
Heart Disease Prediction - Model Evaluation Module
=================================================
This module provides comprehensive evaluation and visualization tools for heart disease prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.model_selection import learning_curve, validation_curve
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseEvaluator:
    """
    Comprehensive evaluation class for heart disease prediction models.
    """

    def __init__(self, model, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.feature_names = None

    def set_feature_names(self, feature_names):
        """Set feature names for interpretability."""
        self.feature_names = feature_names

    def evaluate_classification_metrics(self, X_test, y_test, y_pred=None, y_pred_proba=None):
        """Calculate comprehensive classification metrics."""
        if y_pred is None:
            y_pred = self.model.predict(X_test)
        if y_pred_proba is None and hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred)
        }

        if y_pred_proba is not None:
            metrics['auc_score'] = roc_auc_score(y_test, y_pred_proba)

        return metrics

    def plot_confusion_matrix(self, X_test, y_test, y_pred=None, figsize=(8, 6)):
        """Plot confusion matrix with detailed annotations."""
        if y_pred is None:
            y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})', 
                        ha='center', va='center', color='red', fontsize=10)

        plt.tight_layout()
        plt.show()

        return cm

    def plot_roc_curve(self, X_test, y_test, y_pred_proba=None, figsize=(8, 6)):
        """Plot ROC curve with AUC score."""
        if y_pred_proba is None:
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                print("Model does not support probability prediction")
                return None

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return fpr, tpr, auc_score

    def plot_precision_recall_curve(self, X_test, y_test, y_pred_proba=None, figsize=(8, 6)):
        """Plot Precision-Recall curve."""
        if y_pred_proba is None:
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                print("Model does not support probability prediction")
                return None

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = precision.mean()

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=y_test.mean(), color='red', linestyle='--', 
                   label=f'Baseline (AP = {y_test.mean():.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return precision, recall, avg_precision

    def plot_feature_importance(self, feature_names=None, top_n=15, figsize=(10, 8)):
        """Plot feature importance if available."""
        if feature_names is None:
            feature_names = self.feature_names

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            print("Model does not provide feature importance")
            return None

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]

        # Create feature importance dataframe
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(data=feature_imp_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {self.model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

        return feature_imp_df

    def plot_learning_curve(self, X, y, cv=5, figsize=(10, 6)):
        """Plot learning curve to assess model performance vs training size."""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curve - {self.model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return train_sizes, train_scores, val_scores

    def explain_prediction_shap(self, X_train, X_test, sample_idx=0):
        """Use SHAP to explain individual predictions."""
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(self.model, X_train)
            shap_values = explainer(X_test)

            # Plot explanation for a specific sample
            shap.plots.waterfall(shap_values[sample_idx])

            # Summary plot
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names)

            return shap_values
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None

    def explain_prediction_lime(self, X_train, X_test, sample_idx=0):
        """Use LIME to explain individual predictions."""
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=self.feature_names if self.feature_names else X_train.columns,
                class_names=['No Disease', 'Disease'],
                mode='classification'
            )

            # Explain instance
            explanation = explainer.explain_instance(
                X_test.iloc[sample_idx].values,
                self.model.predict_proba,
                num_features=10
            )

            explanation.show_in_notebook(show_table=True)

            return explanation
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None

    def generate_classification_report(self, X_test, y_test, y_pred=None):
        """Generate detailed classification report."""
        if y_pred is None:
            y_pred = self.model.predict(X_test)

        report = classification_report(y_test, y_pred, 
                                     target_names=['No Disease', 'Disease'],
                                     output_dict=True)

        print(f"\n{'='*50}")
        print(f"CLASSIFICATION REPORT - {self.model_name}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, 
                                  target_names=['No Disease', 'Disease']))

        return report

    def comprehensive_evaluation(self, X_train, X_test, y_train, y_test):
        """Run comprehensive evaluation with all visualizations."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION - {self.model_name}")
        print(f"{'='*60}")

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self.evaluate_classification_metrics(X_test, y_test, y_pred, y_pred_proba)

        print("\nPerformance Metrics:")
        print("-" * 20)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")

        # Generate plots
        self.plot_confusion_matrix(X_test, y_test, y_pred)

        if y_pred_proba is not None:
            self.plot_roc_curve(X_test, y_test, y_pred_proba)
            self.plot_precision_recall_curve(X_test, y_test, y_pred_proba)

        self.plot_feature_importance()
        self.plot_learning_curve(X_train, y_train)

        # Classification report
        self.generate_classification_report(X_test, y_test, y_pred)

        return metrics

# Utility functions for comparing multiple models
def compare_models(models_dict, X_test, y_test):
    """Compare multiple models side by side."""
    comparison_results = {}

    for model_name, model in models_dict.items():
        evaluator = HeartDiseaseEvaluator(model, model_name)
        metrics = evaluator.evaluate_classification_metrics(X_test, y_test)
        comparison_results[model_name] = metrics

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)

    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.round(4))

    return comparison_df

def plot_model_comparison(comparison_df, metric='accuracy', figsize=(12, 6)):
    """Plot comparison of multiple models."""
    plt.figure(figsize=figsize)

    models = comparison_df.index
    values = comparison_df[metric]

    bars = plt.bar(models, values, color='skyblue', edgecolor='navy')
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xlabel('Models')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    # This would be used with trained models and test data
    # evaluator = HeartDiseaseEvaluator(model, "Random Forest")
    # evaluator.set_feature_names(feature_names)
    # evaluator.comprehensive_evaluation(X_train, X_test, y_train, y_test)
    pass
