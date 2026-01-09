This folder contains the Logistic Regression implementation for the

UCI Credit Approval dataset



The focus of this code is to evaluate a classical and interpretable

machine learning model using a methodologically correct experimental setup

based on nested cross-validation.



---



\## Files in This Repository



\### `data\_preprocessing.py`

Shared data loading and preprocessing utilities.



Responsibilities:

\- Load the UCI Credit Approval dataset (`crx.data`)

\- Assign feature names (A1–A15) and target label

\- Handle missing values (`? → NaN`)

\- Map class labels (`+ → 1`, `- → 0`)

\- Split data into features `X` and target `Y`



This module is designed to be reusable and model-agnostic to ensure

a fair and consistent preprocessing pipeline.



---



\### `logistic\_regression\_credit.py`

Main script implementing Logistic Regression experiments.



Main functionalities:

\- Build a preprocessing + model pipeline using:

&nbsp; - Median imputation and standardization for numerical features

&nbsp; - Most-frequent imputation and one-hot encoding for categorical features

\- Perform \*\*5-fold nested stratified cross-validation\*\*

&nbsp; - Inner CV: hyperparameter tuning (ROC-AUC)

&nbsp; - Outer CV: unbiased performance evaluation

\- Report evaluation metrics:

&nbsp; - Accuracy

&nbsp; - ROC-AUC

&nbsp; - Precision, Recall, F1-score

\- Train a final Logistic Regression model on the full dataset

&nbsp; for \*\*interpretability analysis only\*\*

\- Generate and save:

&nbsp; - Cross-validation results

&nbsp; - Model coefficients and odds ratios

&nbsp; - Visualization of top influential features





\## Dataset



\- UCI Credit Approval Dataset

\- 690 instances

\- 15 anonymized features (mixed numerical and categorical)

\- Binary target: credit approved (+) or rejected (-)



Dataset file:

\- `crx.data`



Source:

https://doi.org/10.24432/C5FS30



---



\## Requirements



\- Python 3.9+

\- numpy

\- pandas

\- scikit-learn

\- matplotlib



---



\## How to Run



Make sure `crx.data` is available in one of the expected locations

(e.g., `Data/crx.data` or the working directory).



Run the Logistic Regression experiment:



```bash

python logistic\_regression\_credit.py



