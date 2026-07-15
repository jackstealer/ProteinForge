import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class StrokePredictionModel:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_data(self):
        print("="*70)
        print("LOADING HEALTHCARE STROKE DATA")
        print("="*70)
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nColumns: {', '.join(self.df.columns.tolist())}")
        print(f"\nTarget Variable: stroke")
        print(f"  - 0: No Stroke")
        print(f"  - 1: Stroke")
        return self.df

    def explore_data(self):
        print("\n" + "="*70)
        print("DATA EXPLORATION")
        print("="*70)

        print("\n📊 First 5 rows:")
        print(self.df.head())

        print("\n📊 Dataset Info:")
        print(self.df.info())

        print("\n📊 Statistical Summary:")
        print(self.df.describe())

        print("\n📊 Target Distribution:")
        target_counts = self.df['stroke'].value_counts()
        print(target_counts)
        print(f"\nClass Balance:")
        print(f"  No Stroke (0): {target_counts[0]} ({target_counts[0]/len(self.df)*100:.1f}%)")
        print(f"  Stroke (1): {target_counts[1]} ({target_counts[1]/len(self.df)*100:.1f}%)")

        print("\n📊 Missing Values:")
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent.round(2)
        })
        print(missing_df[missing_df['Missing Count'] > 0])

    def preprocess_data(self):
        print("\n" + "="*70)
        print("DATA PREPROCESSING")
        print("="*70)

        self.df_processed = self.df.copy()

        print("\n🔧 Dropping ID column...")
        self.df_processed.drop('id', axis=1, inplace=True)
        print("  ✓ Removed ID column")

        print("\n🔧 Handling Missing Values...")
        if self.df_processed['bmi'].isnull().any():
            self.df_processed['bmi'] = pd.to_numeric(self.df_processed['bmi'], errors='coerce')
            missing_bmi = self.df_processed['bmi'].isnull().sum()
            median_bmi = self.df_processed['bmi'].median()
            self.df_processed['bmi'].fillna(median_bmi, inplace=True)
            print(f"  ✓ Filled {missing_bmi} missing BMI values with median: {median_bmi:.2f}")

        print("\n🔧 Encoding Categorical Variables...")

        self.df_processed['gender'] = self.df_processed['gender'].map({
            'Male': 1, 'Female': 0, 'Other': 2
        })
        print("  ✓ Encoded gender")

        self.df_processed['ever_married'] = self.df_processed['ever_married'].map({
            'Yes': 1, 'No': 0
        })
        print("  ✓ Encoded ever_married")

        self.df_processed['Residence_type'] = self.df_processed['Residence_type'].map({
            'Urban': 1, 'Rural': 0
        })
        print("  ✓ Encoded Residence_type")

        work_type_dummies = pd.get_dummies(self.df_processed['work_type'], prefix='work')
        self.df_processed = pd.concat([self.df_processed, work_type_dummies], axis=1)
        self.df_processed.drop('work_type', axis=1, inplace=True)
        print("  ✓ One-hot encoded work_type")

        smoking_dummies = pd.get_dummies(self.df_processed['smoking_status'], prefix='smoking')
        self.df_processed = pd.concat([self.df_processed, smoking_dummies], axis=1)
        self.df_processed.drop('smoking_status', axis=1, inplace=True)
        print("  ✓ One-hot encoded smoking_status")

        if self.df_processed.isnull().any().any():
            self.df_processed.dropna(inplace=True)
            print(f"  ✓ Dropped remaining rows with missing values")

        print(f"\n✅ Preprocessing completed!")
        print(f"   Final dataset shape: {self.df_processed.shape[0]} rows × {self.df_processed.shape[1]} columns")

        return self.df_processed

    def split_data(self, test_size=0.2, random_state=42):
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)

        X = self.df_processed.drop('stroke', axis=1)
        y = self.df_processed['stroke']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n✓ Data split completed!")
        print(f"  Training set: {len(self.X_train)} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Testing set: {len(self.X_test)} samples ({test_size*100:.0f}%)")
        print(f"\n  Features: {self.X_train.shape[1]}")
        print(f"  Feature names: {', '.join(X.columns.tolist()[:10])}...")

        print("\n🔧 Scaling features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("  ✓ Features scaled using StandardScaler")

    def train_models(self):
        print("\n" + "="*70)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*70)

        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }

        for name, model in self.models.items():
            print(f"\n🤖 Training {name}...")
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"  ✓ {name} trained successfully!")
            print(f"    - Accuracy: {accuracy:.4f}")
            print(f"    - Precision: {precision:.4f}")
            print(f"    - Recall: {recall:.4f}")
            print(f"    - F1-Score: {f1:.4f}")
            print(f"    - ROC-AUC: {roc_auc:.4f}")
            print(f"    - Cross-Val Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate_models(self):
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)

        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'CV Mean': metrics['cv_mean']
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 Model Comparison:")
        print(comparison_df.to_string(index=False))

        best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   ROC-AUC Score: {comparison_df['ROC-AUC'].max():.4f}")

        print(f"\n📋 Detailed Classification Report for {best_model_name}:")
        best_y_pred = self.results[best_model_name]['y_pred']
        print(classification_report(self.y_test, best_y_pred,
                                   target_names=['No Stroke', 'Stroke']))

        return comparison_df, best_model_name

    def create_visualizations(self):
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)

        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')

        print("\n📈 Creating visualization 1/6: Model Performance Comparison...")
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        comparison_df = pd.DataFrame(comparison_data)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            ax.bar(comparison_df['Model'], comparison_df[metric], color='steelblue', alpha=0.7)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

        ax = axes[1, 2]
        comparison_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Overall Model Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/1_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Creating visualization 2/6: Confusion Matrices...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()

        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Stroke', 'Stroke'],
                       yticklabels=['No Stroke', 'Stroke'])
            axes[idx].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.4f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')

        plt.tight_layout()
        plt.savefig('visualizations/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Creating visualization 3/6: ROC Curves...")
        plt.figure(figsize=(10, 8))

        for name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.4f})", linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig('visualizations/3_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Creating visualization 4/6: Precision-Recall Curves...")
        plt.figure(figsize=(10, 8))

        for name, metrics in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, metrics['y_pred_proba'])
            plt.plot(recall, precision, label=f"{name}", linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig('visualizations/4_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Creating visualization 5/6: Cross-Validation Scores...")
        plt.figure(figsize=(10, 6))

        cv_data = []
        for name, metrics in self.results.items():
            cv_data.append({
                'Model': name,
                'Mean': metrics['cv_mean'],
                'Std': metrics['cv_std']
            })
        cv_df = pd.DataFrame(cv_data)

        plt.bar(cv_df['Model'], cv_df['Mean'], yerr=cv_df['Std'],
               capsize=5, color='coral', alpha=0.7)
        plt.title('Cross-Validation Scores (5-Fold)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/5_cross_validation_scores.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Creating visualization 6/6: Feature Importance...")
        rf_model = self.results['Random Forest']['model']
        feature_names = self.df_processed.drop('stroke', axis=1).columns

        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], color='mediumseagreen', alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/6_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\n✅ All visualizations created successfully!")
        print("   Saved in: visualizations/")

    def generate_report(self, best_model_name):
        print("\n" + "="*70)
        print("GENERATING ANALYSIS REPORT")
        print("="*70)

        report = []
        report.append("="*70)
        report.append("STROKE PREDICTION - MACHINE LEARNING MODEL REPORT")
        report.append("="*70)
        report.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: healthcare-dataset-stroke-data.csv")
        report.append(f"Total Samples: {len(self.df)}")
        report.append(f"Training Samples: {len(self.X_train)}")
        report.append(f"Testing Samples: {len(self.X_test)}")

        report.append("\n" + "-"*70)
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-"*70)

        for name, metrics in self.results.items():
            report.append(f"\n{name}:")
            report.append(f"  • Accuracy:  {metrics['accuracy']:.4f}")
            report.append(f"  • Precision: {metrics['precision']:.4f}")
            report.append(f"  • Recall:    {metrics['recall']:.4f}")
            report.append(f"  • F1-Score:  {metrics['f1_score']:.4f}")
            report.append(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
            report.append(f"  • Cross-Val: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")

        report.append("\n" + "-"*70)
        report.append("BEST MODEL")
        report.append("-"*70)
        report.append(f"\n🏆 {best_model_name}")
        best_metrics = self.results[best_model_name]
        report.append(f"\nPerformance Metrics:")
        report.append(f"  • Accuracy:  {best_metrics['accuracy']:.4f}")
        report.append(f"  • Precision: {best_metrics['precision']:.4f}")
        report.append(f"  • Recall:    {best_metrics['recall']:.4f}")
        report.append(f"  • F1-Score:  {best_metrics['f1_score']:.4f}")
        report.append(f"  • ROC-AUC:   {best_metrics['roc_auc']:.4f}")

        cm = confusion_matrix(self.y_test, best_metrics['y_pred'])
        report.append(f"\nConfusion Matrix:")
        report.append(f"  True Negatives:  {cm[0][0]}")
        report.append(f"  False Positives: {cm[0][1]}")
        report.append(f"  False Negatives: {cm[1][0]}")
        report.append(f"  True Positives:  {cm[1][1]}")

        report.append("\n" + "-"*70)
        report.append("KEY INSIGHTS")
        report.append("-"*70)

        report.append("\n1. MODEL COMPARISON")
        report.append(f"   • Total models trained: {len(self.models)}")
        report.append(f"   • Best performing model: {best_model_name}")
        report.append(f"   • Best ROC-AUC score: {best_metrics['roc_auc']:.4f}")

        report.append("\n2. DATASET CHARACTERISTICS")
        target_counts = self.df['stroke'].value_counts()
        report.append(f"   • Total samples: {len(self.df)}")
        report.append(f"   • No stroke cases: {target_counts[0]} ({target_counts[0]/len(self.df)*100:.1f}%)")
        report.append(f"   • Stroke cases: {target_counts[1]} ({target_counts[1]/len(self.df)*100:.1f}%)")
        report.append(f"   • Class imbalance detected: {'Yes' if target_counts[0]/target_counts[1] > 3 else 'No'}")

        report.append("\n3. MODEL RECOMMENDATIONS")
        if best_metrics['recall'] < 0.5:
            report.append("   ⚠ Low recall indicates the model misses many stroke cases")
            report.append("   → Consider class balancing techniques (SMOTE, class weights)")
        if best_metrics['precision'] < 0.5:
            report.append("   ⚠ Low precision indicates many false positives")
            report.append("   → Consider adjusting decision threshold")
        report.append(f"   ✓ {best_model_name} recommended for deployment")
        report.append("   ✓ Monitor model performance on new data regularly")

        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)

        report_text = '\n'.join(report)
        with open('model_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print("\n✅ Report saved to: model_report.txt")


def main():
    print("\n" + "="*70)
    print("TASK 2: PREDICTIVE MODELING USING MACHINE LEARNING")
    print("="*70)
    print("\nObjective: Build a model to predict stroke outcomes")
    print("\nKey Features:")
    print("  ✓ Apply algorithms: Logistic Regression, Decision Trees, Random Forest")
    print("  ✓ Train and test models for accuracy")
    print("  ✓ Visualize performance using confusion matrices and ROC curves")
    print("\n" + "="*70)

    model = StrokePredictionModel('healthcare-dataset-stroke-data.csv')

    model.load_data()
    model.explore_data()
    model.preprocess_data()
    model.split_data()
    model.train_models()
    comparison_df, best_model_name = model.evaluate_models()
    model.create_visualizations()
    model.generate_report(best_model_name)

    print("\n" + "="*70)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output Files Generated:")
    print("   1. model_report.txt - Comprehensive model evaluation report")
    print("   2. visualizations/ - 6 visualization files")
    print("\n🎯 Models Trained:")
    print("   • Logistic Regression")
    print("   • Decision Tree")
    print("   • Random Forest")
    print("   • Gradient Boosting")
    print(f"\n🏆 Best Model: {best_model_name}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
