
"""
Heart Disease Prediction - Main Execution Script
===============================================
This script orchestrates the complete heart disease prediction pipeline.
"""

import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import HeartDiseasePreprocessor
from model_training import HeartDiseaseModelTrainer
from model_evaluation import HeartDiseaseEvaluator, compare_models, plot_model_comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictionPipeline:
    """
    Complete pipeline for heart disease prediction project.
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.preprocessor = HeartDiseasePreprocessor()
        self.trainer = HeartDiseaseModelTrainer()
        self.best_model = None
        self.best_model_name = None
        self.processed_data = None

    def create_sample_dataset(self):
        """Create a sample dataset for demonstration (Cleveland Heart Disease Dataset format)."""
        print("Creating sample heart disease dataset...")

        # Sample data based on Cleveland Heart Disease Dataset
        np.random.seed(42)
        n_samples = 500

        data = {
            'age': np.random.randint(25, 80, n_samples),
            'sex': np.random.choice([0, 1], n_samples),  # 0: female, 1: male
            'cp': np.random.choice([0, 1, 2, 3], n_samples),  # chest pain type
            'trestbps': np.random.randint(90, 200, n_samples),  # resting blood pressure
            'chol': np.random.randint(100, 400, n_samples),  # serum cholesterol
            'fbs': np.random.choice([0, 1], n_samples),  # fasting blood sugar > 120 mg/dl
            'restecg': np.random.choice([0, 1, 2], n_samples),  # resting ECG results
            'thalach': np.random.randint(60, 220, n_samples),  # max heart rate achieved
            'exang': np.random.choice([0, 1], n_samples),  # exercise induced angina
            'oldpeak': np.random.uniform(0, 6, n_samples),  # ST depression
            'slope': np.random.choice([0, 1, 2], n_samples),  # slope of peak exercise ST segment
            'ca': np.random.choice([0, 1, 2, 3], n_samples),  # number of major vessels
            'thal': np.random.choice([0, 1, 2], n_samples),  # thalassemia
        }

        # Create synthetic target based on risk factors (simplified logic)
        risk_score = (
            (data['age'] > 60) * 0.3 +
            (data['sex'] == 1) * 0.2 +  # males higher risk
            (data['cp'] == 0) * 0.2 +   # asymptomatic chest pain
            (data['trestbps'] > 140) * 0.2 +
            (data['chol'] > 240) * 0.2 +
            (data['thalach'] < 150) * 0.2 +
            (data['exang'] == 1) * 0.3 +
            (data['oldpeak'] > 2) * 0.2 +
            (data['ca'] > 0) * 0.3 +
            (data['thal'] == 2) * 0.2
        )

        # Add some noise and create binary target
        risk_score += np.random.normal(0, 0.2, n_samples)
        data['target'] = (risk_score > np.percentile(risk_score, 60)).astype(int)

        df = pd.DataFrame(data)

        # Save sample dataset
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/heart_disease_sample.csv', index=False)

        print(f"Sample dataset created with {n_samples} samples")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")

        return df

    def run_complete_pipeline(self, create_sample=True):
        """Run the complete heart disease prediction pipeline."""
        print("\n" + "="*80)
        print("HEART DISEASE PREDICTION - COMPLETE ML PIPELINE")
        print("="*80)
        print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Data Loading/Creation
        if create_sample or self.data_path is None:
            df = self.create_sample_dataset()
            data_path = 'data/raw/heart_disease_sample.csv'
        else:
            data_path = self.data_path

        # Step 2: Data Preprocessing
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)

        self.processed_data = self.preprocessor.preprocess_pipeline(data_path)

        if self.processed_data is None:
            print("Error: Data preprocessing failed!")
            return None

        X_train = self.processed_data['X_train']
        X_test = self.processed_data['X_test']
        y_train = self.processed_data['y_train']
        y_test = self.processed_data['y_test']
        feature_names = self.processed_data['feature_names']

        # Step 3: Model Training
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)

        self.trainer.train_all_models(X_train, y_train)

        # Step 4: Hyperparameter Tuning (for top 3 models)
        print("\n" + "="*60)
        print("STEP 3: HYPERPARAMETER TUNING")
        print("="*60)

        self.trainer.hyperparameter_tuning(X_train, y_train)

        # Step 5: Model Evaluation
        print("\n" + "="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)

        evaluation_results = self.trainer.evaluate_models(X_test, y_test)

        # Step 6: Get Best Model
        self.best_model, self.best_model_name = self.trainer.get_best_model()

        # Step 7: Comprehensive Evaluation of Best Model
        print("\n" + "="*60)
        print("STEP 5: COMPREHENSIVE EVALUATION")
        print("="*60)

        evaluator = HeartDiseaseEvaluator(self.best_model, self.best_model_name)
        evaluator.set_feature_names(feature_names)

        final_metrics = evaluator.comprehensive_evaluation(X_train, X_test, y_train, y_test)

        # Step 8: Save Best Model
        print("\n" + "="*60)
        print("STEP 6: MODEL PERSISTENCE")
        print("="*60)

        os.makedirs('models/trained_models', exist_ok=True)
        model_path = self.trainer.save_model(self.best_model, self.best_model_name)

        # Step 9: Generate Summary Report
        self.generate_summary_report(final_metrics, model_path)

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'best_model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': final_metrics,
            'model_path': model_path,
            'processed_data': self.processed_data
        }

    def generate_summary_report(self, metrics, model_path):
        """Generate a summary report of the pipeline results."""
        report = f"""
HEART DISEASE PREDICTION - PROJECT SUMMARY REPORT
=================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BEST MODEL PERFORMANCE:
----------------------
Model: {self.best_model_name}
Accuracy: {metrics.get('accuracy', 'N/A'):.4f}
Precision: {metrics.get('precision', 'N/A'):.4f}
Recall: {metrics.get('recall', 'N/A'):.4f}
F1-Score: {metrics.get('f1_score', 'N/A'):.4f}
AUC Score: {metrics.get('auc_score', 'N/A'):.4f}
Matthews Correlation Coefficient: {metrics.get('matthews_corrcoef', 'N/A'):.4f}

MODEL DETAILS:
--------------
Model Path: {model_path}
Features Used: {len(self.processed_data['feature_names'])}
Training Samples: {len(self.processed_data['y_train'])}
Testing Samples: {len(self.processed_data['y_test'])}

DATASET INFORMATION:
-------------------
Target Distribution (Training):
{self.processed_data['y_train'].value_counts().to_dict()}

Target Distribution (Testing):
{self.processed_data['y_test'].value_counts().to_dict()}

RECOMMENDATIONS:
---------------
1. The model is ready for deployment in a clinical decision support system
2. Consider collecting more diverse data to improve generalization
3. Implement continuous monitoring and retraining procedures
4. Validate model performance on external datasets before clinical use

"""

        # Save report
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/heart_disease_prediction_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nSummary report saved to: {report_path}")
        print(report)

        return report_path

    def predict_new_patient(self, patient_data):
        """Make prediction for a new patient."""
        if self.best_model is None:
            print("Error: No trained model available. Run the pipeline first.")
            return None

        # Preprocess the patient data (this would need to match training preprocessing)
        # For now, assume data is already preprocessed
        prediction = self.best_model.predict([patient_data])[0]

        if hasattr(self.best_model, 'predict_proba'):
            probability = self.best_model.predict_proba([patient_data])[0]
            return {
                'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                'probability': {
                    'No Heart Disease': probability[0],
                    'Heart Disease': probability[1]
                }
            }
        else:
            return {
                'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease'
            }

def main():
    """Main function to run the heart disease prediction pipeline."""
    # Initialize pipeline
    pipeline = HeartDiseasePredictionPipeline()

    # Run complete pipeline
    results = pipeline.run_complete_pipeline(create_sample=True)

    if results:
        print("\n🎉 Heart Disease Prediction Pipeline completed successfully!")
        print(f"📊 Best Model: {results['model_name']}")
        print(f"🎯 Accuracy: {results['metrics'].get('accuracy', 'N/A'):.4f}")
        print(f"💾 Model saved at: {results['model_path']}")
    else:
        print("❌ Pipeline failed to complete")

if __name__ == "__main__":
    main()
