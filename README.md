                            Machine Learning Classification Models and Evaluation Metrics

a. Problem Statement

The objective of this project is to implement and compare six different machine learning classification models on a Wine Quality dataset and evaluate their performance using multiple evaluation metrics.

The goal is to determine which classification algorithm performs best in predicting the wine quality class based on physicochemical features.

b. Dataset Description

The dataset used contains physicochemical properties of wine samples and a target variable indicating the wine quality class.

Features include:

Fixed Acidity

Volatile Acidity

Citric Acid

Residual Sugar

Chlorides

Free Sulfur Dioxide

Total Sulfur Dioxide

Density

pH

Sulphates

Alcohol

Dataset Information:

Number of samples: (Fill from your dataset, e.g., 1599 or your split count)

Number of features: 11

Target variable: Wine Quality

Type of problem: Binary / Multiclass Classification

c. Models Used

The following machine learning classification models were implemented:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble Boosting Model)

d. Model Comparison Table

After training and evaluation, the models were compared using the following metrics:

Accuracy

AUC (Area Under ROC Curve)

Precision

Recall

F1 Score

MCC (Matthews Correlation Coefficient)

ML Model	              Accuracy	AUC	Precision	Recall	F1 Score	MCC

Logistic Regression	0.672131148	0.769518072	0.676110062	0.672131148	0.672837761	0.34525274
Decision Tree           0.677595628	0.676325301	0.678772765	0.677595628	0.67800235	0.351761726
KNN                     0.672131148	0.711024096	0.671460071	0.672131148	0.667756606	0.333040984
Naive Bayes             0.672131148	0.737228916	0.67110955	0.672131148	0.671297784	0.336163224
Random Forest           0.737704918	0.791807229	0.740326772	0.737704918	0.738223723	0.475102729
XGBoost	                0.737704918	0.788554217	0.741652503	0.737704918	0.738270209	0.476971818
					

e. Observations

ML Model	                  Observation

Logistic Regression	          Performs well for linearly separable data
Decision Tree	                  Can overfit if tree depth is not controlled
KNN	                          Sensitive to scaling and K value selection
Naive Bayes	                  Works well when probabilistic assumptions hold
Random Forest	                  Reduces overfitting and improves accuracy using ensemble learning
XGBoost	                          Often achieves best performance due to boosting and sequential learning

f. Technologies Used

Python
Scikit-learn
XGBoost
Pandas
NumPy
Streamlit
Matplotlib
Seaborn

g. Project Structure

wine-ml-project/
│
├── train_models.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── wine_train.csv
├── wine_test.csv
└── model/
      ├── logistic.pkl
      ├── decision_tree.pkl
      ├── knn.pkl
      ├── naive_bayes.pkl
      ├── random_forest.pkl
      ├── xgboost.pkl
      ├── scaler.pkl

h. How to Run the Project
1. Train Models and Generate Pickle Files
python train_models.py
2. Run Streamlit App
streamlit run streamlit_app.py
i. Conclusion

This project demonstrates the practical implementation of multiple classification algorithms and highlights the importance of evaluation metrics in selecting the best performing model.

Ensemble models such as Random Forest and XGBoost generally provide better predictive performance compared to single estimators.


