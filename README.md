Wine Classification using Multiple ML Models
a. Problem Statement

The objective of this project is to classify wine samples into different categories based on their chemical properties using multiple machine learning classification models and compare their performance.

b. Dataset Description

The Wine dataset contains chemical analysis results of wines grown in the same region.

Type: Multi-class Classification

Number of Classes: 3

Features: 13 chemical properties

Target: Wine class (0, 1, 2)

c. Models Used

Logistic Regression

Decision Tree

K-Nearest Neighbors

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

Model Comparison Table

(After running train_models.py, paste actual values here)

ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression						
Decision Tree						
KNN						
Naive Bayes						
Random Forest						
XGBoost						
Observations
Model	Observation
Logistic Regression	Performs well due to well-separated classes
Decision Tree	Can overfit if not pruned
KNN	Performs well with proper scaling
Naive Bayes	Assumes feature independence
Random Forest	Improves stability and accuracy
XGBoost	Typically provides best performance
✅ 5️⃣ Final GitHub Structure
