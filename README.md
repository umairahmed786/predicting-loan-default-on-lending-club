# Predicting Loan Default on Lending Club

## Overview
This project uses machine learning to predict whether a borrower will fully repay a loan or default on it. Using historical Lending Club loan data, the project compares different classification models to identify borrowers who are more likely to default.

The main focus is not only on model performance but also on building a realistic evaluation process that avoids data leakage and survivorship bias.

## Objectives
- Analyze historical loan and borrower characteristics.
- Clean data and handle missing values.
- Prevent data leakage by removing information unavailable at loan issuance.
- Address survivorship bias by retaining loans whose full repayment term ended by the data cutoff.
- Train and compare multiple machine learning models.
- Evaluate models using appropriate classification metrics.
- Investigate how dataset selection affects model performance.

## Dataset
The project uses the Lending Club loan dataset from Kaggle, covering loans issued between 2007 and 2018.

**Dataset:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download `accepted_2007_to_2018Q4.csv` and place it in:

`dataset_accepted_2007_to_2018Q4/accepted_2007_to_2018Q4.csv`

The dataset is not included in this repository.

## Project Workflow

### 1. Data Preprocessing
- Load the large CSV file in chunks to manage memory usage.
- Remove identifiers, redundant columns, and information that could reveal loan outcomes.
- Convert loan statuses into binary labels:
  - `0`: Fully Paid
  - `1`: Charged Off or Default
- Process employment length, credit history, and other borrower features.
- Handle missing values through model pipelines and missing-value indicators.

### 2. Preventing Survivorship Bias
Loans that have not completed their full repayment term may not have had enough time to default. Including them can distort the default rate and produce misleading evaluation results.

The project therefore retains loans whose full terms ended by December 2018, allowing their outcomes to be evaluated more fairly.

### 3. Exploratory Data Analysis
Analyze default rates across loan issue years and loan terms to identify patterns in the data and understand how incomplete loan histories can affect the results.

### 4. Model Training
Train and compare several classification models:

- Dummy Classifier (baseline)
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- XGBoost
- LightGBM

The project also investigates class imbalance and compares feature sets and model configurations.

### 5. Model Evaluation
Evaluate model performance using:
- **ROC-AUC:** Measures how well the model ranks defaulters above non-defaulters.
- **PR-AUC:** Evaluates precision and recall, particularly useful for imbalanced datasets.
- **Brier Score:** Measures the accuracy of predicted probabilities.
- **Precision and Recall:** Help assess the trade-off between identifying defaults and incorrectly flagging repaying borrowers.

Chronological data splitting is used to simulate predicting outcomes for loans issued in later periods.

### 6. Experiments and Findings
Additional experiments investigate why earlier model evaluations achieved higher ROC-AUC scores.

The results show that including loans whose repayment terms had not fully elapsed can artificially improve evaluation scores. Removing these loans reduces ROC-AUC to approximately 0.70–0.72, providing a more realistic assessment of model performance.

This demonstrates that dataset selection and evaluation methodology can be just as important as the choice of machine learning algorithm.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Jupyter Notebook

## How to Run
1. Clone or download the repository.
2. Install the required dependencies.
3. Download the Lending Club dataset from Kaggle.
4. Place the CSV file in the expected dataset directory.
5. Open `predicting_loan_default_on_lending_club.ipynb` in Jupyter Notebook or JupyterLab.
6. Run the notebook cells sequentially.

## Conclusion
This project demonstrates an end-to-end credit-risk prediction workflow, from data preprocessing and exploratory analysis to model comparison and evaluation.

Its key finding is that reliable loan-default prediction requires careful handling of incomplete loan histories, data leakage, class imbalance, and chronological changes in borrower populations. A realistic evaluation provides more meaningful insights than a high score obtained from a biased dataset.
