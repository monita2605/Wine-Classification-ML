&nbsp;                                             Wine Classification using Machine Learning



**a. Problem Statement**



The objective of this project is to develop and compare multiple Machine Learning classification models to predict the wine class based on its chemical composition.



The task involves training six different supervised classification algorithms on the same dataset and evaluating their performance using multiple evaluation metrics. The goal is to determine which model performs best and provides the most reliable predictions.



The models are implemented using a Pipeline-based architecture to ensure proper preprocessing, avoid data leakage, and maintain consistency during training and testing.



**b. Dataset Description**



The dataset used in this project is the Wine Dataset, which contains chemical analysis results of wines derived from three different cultivars.



Each sample represents a wine instance described by 13 numerical features.



Features:



Alcohol



Malic Acid



Ash



Alcalinity of Ash



Magnesium



Total Phenols



Flavanoids



Nonflavanoid Phenols



Proanthocyanins



Color Intensity



Hue



OD280/OD315 of Diluted Wines



Proline



**Target Variable:**



target → Wine Class (Multiclass Classification)



Data Split:



80% Training Data



20% Testing Data



Stratified sampling used to preserve class distribution.



**c. Models Used**



The following six classification models were implemented:



Logistic Regression



Decision Tree Classifier



k-Nearest Neighbors (kNN)



Naive Bayes (GaussianNB)



Random Forest (Ensemble Model)



XGBoost (Ensemble Model)



All models were trained using Scikit-learn Pipelines, ensuring preprocessing steps (such as scaling) are properly integrated within the model workflow.



📊 Model Comparison Table

ML Model Name	                Accuracy	   AUC	        Precision	   Recall	F1 Score	MCC

Logistic Regression	        0.6721	           0.7695	0.6761	           0.6721	0.6728	        0.3453

Decision Tree	                0.6776	           0.6763	0.6788	           0.6776	0.6780	        0.3518

kNN	                        0.6721	           0.7110	0.6715	           0.6721	0.6678	        0.3330

Naive Bayes	                0.6721	           0.7372	0.6711	           0.6721	0.6713	        0.3362

Random Forest (Ensemble)	0.7377	           0.7918	0.7403	           0.7377	0.7382	        0.4751

XGBoost (Ensemble)	        0.7377	           0.7886	0.7417	           0.7377	0.7383	        0.4770



**Evaluation Metrics Used:**



Accuracy – Overall correctness of predictions



AUC – Ability to distinguish between classes



Precision – Correct positive predictions ratio



Recall – Sensitivity to actual positive cases



F1 Score – Harmonic mean of Precision and Recall



MCC (Matthews Correlation Coefficient) – Balanced measure even for imbalanced datasets



**Observations on Model Performance**



ML Model Name	                     Observation about Model Performance

Logistic Regression	              Provides stable and consistent results with strong AUC. Performs moderately well but does not outperform ensemble methods.

Decision Tree	                      Slight improvement in Accuracy compared to Logistic Regression but lower AUC. Prone to overfitting due to tree structure.

kNN	                              Performance is comparable to Logistic Regression but slightly lower MCC. Sensitive to feature scaling and distance metric.

Naive Bayes	                      Performs reasonably well despite strong independence assumptions. However, its assumptions limit overall predictive power.

Random Forest (Ensemble)	      Significantly better performance across most metrics. High AUC and MCC indicate strong generalization ability.

XGBoost (Ensemble)	              Achieves the highest MCC and competitive AUC. Demonstrates superior predictive performance and robustness.



Final Conclusion



From the comparative analysis:



Best Accuracy:           Random Forest \& XGBoost (0.7377)



Best AUC:                Random Forest (0.7918)



Best MCC:                XGBoost (0.4770)



The ensemble models — particularly XGBoost and Random Forest — clearly outperform individual classifiers.



Therefore, XGBoost is selected as the best overall performing model for the Wine Classification dataset due to its strong balance across all evaluation metrics.

