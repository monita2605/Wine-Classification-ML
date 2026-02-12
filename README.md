                                # Wine Classification using Machine Learning Models



\## a. Problem Statement



The objective of this project is to classify wine samples into different categories using multiple machine learning classification models and compare their performance using evaluation metrics.



---



\## b. Dataset Description



The Wine dataset contains chemical analysis of wine samples.



\- Type: Classification Problem

\- Target Column: target

\- Number of Models Implemented: 6

\- Dataset split into training and testing sets



---



\## c. Models Used



1\. Logistic Regression

2\. Decision Tree

3\. K-Nearest Neighbors

4\. Naive Bayes

5\. Random Forest (Ensemble)

6\. XGBoost (Ensemble)



---



\## Evaluation Metrics Used



\- Accuracy

\- AUC Score

\- Precision

\- Recall

\- F1 Score

\- Matthews Correlation Coefficient (MCC)



---



\## Model Comparison Table



| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

|--------|----------|-----|-----------|--------|----------|-----|

| Logistic Regression |  |  |  |  |  |  |

| Decision Tree |  |  |  |  |  |  |

| KNN |  |  |  |  |  |  |

| Naive Bayes |  |  |  |  |  |  |

| Random Forest |  |  |  |  |  |  |

| XGBoost |  |  |  |  |  |  |



(Add values printed from train\_models.py here)



---



\## Observations



\- Logistic Regression performs well for structured numeric data.

\- Decision Tree may overfit if depth is not controlled.

\- KNN performance depends on feature scaling.

\- Naive Bayes assumes feature independence.

\- Random Forest improves stability and reduces overfitting.

\- XGBoost often gives the best performance due to boosting.



---



\## Streamlit App Features



\- Upload test dataset (CSV)

\- Select model from dropdown

\- Display evaluation metrics

\- Display confusion matrix



---



\## How to Run



1\. Train models:



&nbsp;  python train\_models.py



2\. Run Streamlit app:



&nbsp;  streamlit run app.py

