# ==============================================================================
# SCRIPT: Feature_Engineering.py (Model Training & Evaluation)
# TOPIC: Logistic Regression on Titanic Dataset
# STYLE: Interviewer Q&A + In-Depth Technical Breakdown for Every Keyword
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: IMPORTING PREPROCESSED DATA
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION]: "Why do we import 'df' from Preprocessing.py rather than writing everything in one single script?"
# [ANSWER]: 
# 1. Modular Architecture: Separates Data Cleaning/ETL pipelines from Model Training and Evaluation pipelines.
# 2. Code Reusability: Cleaned data can be imported into multiple model experiments (e.g., Logistic Regression, Random Forest, SVM) without duplicate code.
# 3. Maintainability: If preprocessing steps change (e.g., new imputation strategy), model scripts don't need to be rewritten.
from Preprocessing import df


# ------------------------------------------------------------------------------
# STEP 2: DATA SEPARATION (FEATURES 'X' vs TARGET 'y')
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "What is the difference between Feature Matrix (X) and Target Vector (y)?"
# [ANSWER]:
# - 'X' (Independent Variables / Features / Predictors / Covariates): The input attributes that the model uses to learn patterns.
# - 'y' (Dependent Variable / Target / Ground Truth / Label): The outcome we want the model to predict ('survived': 0 = Died, 1 = Survived).
#
# [INTERVIEWER QUESTION 2]: "Why do we specify 'axis=1' in df.drop()?"
# [ANSWER]:
# - 'axis=0' refers to row-wise operations (index).
# - 'axis=1' refers to column-wise operations.
# - Therefore, `df.drop('survived', axis=1)` removes the column named 'survived' across all rows and returns the remaining feature columns.
#
# [INTERVIEWER QUESTION 3]: "Why is 'X' conventionally written in uppercase and 'y' in lowercase?"
# [ANSWER]: Linear Algebra convention:
# - 'X' is a 2D Matrix (N samples x D features) -> Capital letter.
# - 'y' is a 1D Vector (N samples x 1 label) -> Small letter.

# Drop target column to isolate all independent features
x = df.drop('survived', axis=1) 

# Extract target column as dependent variable
y = df['survived']


# ------------------------------------------------------------------------------
# STEP 3: TRAIN-TEST SPLIT (DATASET PARTITIONING)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Why do we split data into training and testing sets? Why not train on 100% data?"
# [ANSWER]:
# - To evaluate GENERALIZATION on unseen data and prevent OVERFITTING.
# - If we train and evaluate on the same data, the model can simply memorize the training examples (high training accuracy, poor real-world performance).
# - Test set acts as a proxy for unseen future data.
#
# [INTERVIEWER QUESTION 2]: "What does 'test_size=0.2' signify?"
# [ANSWER]: 20% of the total dataset is set aside for testing, and the remaining 80% is used for training the model.
#
# [INTERVIEWER QUESTION 3]: "What does 'random_state=42' do and why is it used?"
# [ANSWER]:
# - It is the seed for the pseudo-random number generator.
# - It guarantees REPRODUCIBILITY: Every time the script is executed, it splits the exact same rows into train and test sets.
# - (Without random_state, every execution creates a different split, making it impossible to fairly compare model experiments).
#
# [INTERVIEWER QUESTION 4 (ADVANCED)]: "What is 'stratify' and when would you use `stratify=y`?"
# [ANSWER]:
# - In classification (especially imbalanced datasets), `stratify=y` ensures that both the training and test splits retain the exact same percentage of each class (0s and 1s) as the original dataset.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    random_state=42
)


# ------------------------------------------------------------------------------
# STEP 4: MODEL INSTANTIATION (LOGISTIC REGRESSION)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Why is it called Logistic 'Regression' if it solves a Classification problem?"
# [ANSWER]:
# - It uses a Linear Regression equation under the hood: z = w1*x1 + w2*x2 + ... + wn*xn + b (z = w^T * x + b).
# - It passes this continuous value 'z' (linear combination) through a Sigmoid (Logistic) Function:
#       p = 1 / (1 + e^(-z))
# - The Sigmoid function "squashes" any real value from (-inf, +inf) into a probability between [0, 1].
#
# [INTERVIEWER QUESTION 2]: "What are the core assumptions of Logistic Regression?"
# [ANSWER]:
# 1. Target variable is categorical (Binary: 0 or 1 for standard logistic regression).
# 2. Linearity in Log-Odds: Independent features are linearly related to the Log-Odds (Logit): ln(p / (1-p)) = w^T * x + b.
# 3. No extreme Multicollinearity among independent features.
# 4. Independent observations (no autocorrelation between records).
#
# [INTERVIEWER QUESTION 3]: "What are the key hyperparameters of LogisticRegression()?"
# [ANSWER]:
# - 'penalty': Regularization norm ('l2' default, 'l1', 'elasticnet'). L1 creates sparsity for feature selection; L2 shrinks weights.
# - 'C': Inverse regularization strength (C = 1/lambda). Smaller C = stronger regularization (prevents overfitting); Larger C = weaker regularization.
# - 'solver': Optimization algorithm ('lbfgs' default, 'liblinear', 'saga').
# - 'max_iter': Maximum iterations for the optimization algorithm to converge (default 100). We set max_iter=1000 to prevent convergence warnings.
# - 'class_weight': Balances class weights if data is imbalanced (e.g., class_weight='balanced').

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)


# ------------------------------------------------------------------------------
# STEP 5: MODEL TRAINING (FITTING)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "What mathematical process occurs during `model.fit(x_train, y_train)`?"
# [ANSWER]:
# - The model finds the optimal weights (w) and bias (b) that minimize the Binary Cross-Entropy / Log-Loss function:
#       Cost = - (1/N) * sum[ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
# - Unlike Linear Regression which has a closed-form OLS analytical solution, Logistic Regression uses Maximum Likelihood Estimation (MLE) optimized iteratively via Gradient Descent or Quasi-Newton methods (L-BFGS).
#
# [INTERVIEWER QUESTION 2]: "How do you inspect the learned parameters?"
# [ANSWER]:
# - `model.coef_`: The learned feature weights (coefficients). Positive weight means higher feature value increases probability of survival.
# - `model.intercept_`: The base log-odds (bias) when all features are 0.

model.fit(x_train, y_train)


# ------------------------------------------------------------------------------
# STEP 6: MODEL PREDICTION (INFERENCE)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "How does `model.predict()` convert probabilities to class labels?"
# [ANSWER]:
# 1. Calculates predicted probability p = P(y=1|X) using the Sigmoid formula.
# 2. Applies a default Decision Threshold of 0.5:
#    - If p >= 0.5 -> Predicts Class 1 (Survived).
#    - If p < 0.5  -> Predicts Class 0 (Did Not Survive).
#
# [INTERVIEWER QUESTION 2]: "What is the difference between `predict()` and `predict_proba()`?"
# [ANSWER]:
# - `model.predict(x_test)`: Returns discrete class labels [0, 1, 0, 1...].
# - `model.predict_proba(x_test)`: Returns continuous probabilities [P(y=0), P(y=1)] for each sample.
#   Used when adjusting decision thresholds for business needs (e.g., lower threshold to 0.3 to catch more positive cases in medical diagnostics).

y_pred = model.predict(x_test)


# ------------------------------------------------------------------------------
# STEP 7: MODEL EVALUATION (CLASSIFICATION METRICS)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Why don't we use R-squared (R2) or Mean Squared Error (MSE) to evaluate Logistic Regression?"
# [ANSWER]:
# - R2 and MSE are Regression metrics designed for continuous outputs.
# - Logistic Regression outputs discrete classes and probabilities, requiring Classification metrics based on the Confusion Matrix.
#
# [INTERVIEWER QUESTION 2]: "What is the Accuracy Paradox? Why can Accuracy be a misleading metric?"
# [ANSWER]:
# - In an imbalanced dataset (e.g., 95% Class 0 and 5% Class 1), a naive model that predicts 0 every time achieves 95% accuracy while failing 100% of the positive class.
# - That is why we must evaluate Confusion Matrix, Precision, Recall, and F1-Score.

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# [INTERVIEWER QUESTION]: "What is the formula and meaning of Accuracy?"
# [ANSWER]:
# - Formula: Accuracy = (True Positives + True Negatives) / (Total Predictions) = (TP + TN) / (TP + TN + FP + FN)
# - Meaning: Proportion of total predictions that were correct.
# - Note: Standard Scikit-Learn convention is accuracy_score(y_true, y_pred)
h = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {h:.4f}")


# ------------------------------------------------------------------------------
# STEP 8: CONFUSION MATRIX BREAKDOWN
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION]: "Explain the structure of Scikit-Learn's Confusion Matrix:"
# [ANSWER]:
# Scikit-Learn formats the confusion matrix as:
#
#                         Predicted Class 0 (Died)      Predicted Class 1 (Survived)
# Actual Class 0 (Died):          TN                            FP (Type I Error)
# Actual Class 1 (Survived):      FN (Type II Error)            TP
#
# Breakdown for our test set (Total = 178 samples | Actual 0s = 109, Actual 1s = 69):
# - TN (True Negative)  = 90 -> Actually died (0), model correctly predicted died (0).
# - FP (False Positive) = 19 -> Actually died (0), but model predicted survived (1) [False Alarm / Type I Error].
# - FN (False Negative) = 16 -> Actually survived (1), but model predicted died (0) [Missed Detection / Type II Error].
# - TP (True Positive)  = 53 -> Actually survived (1), model correctly predicted survived (1).

i = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(i)


# ------------------------------------------------------------------------------
# STEP 9: CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE, SUPPORT)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Define Precision. When is Precision prioritized over Recall?"
# [ANSWER]:
# - Formula: Precision = TP / (TP + FP)
# - Meaning: Out of all instances the model predicted as Positive, how many were actually Positive?
# - Priority: When FALSE POSITIVES (FP) are costly.
#   Example: Spam Email Filter (You don't want an important email incorrectly flagged as Spam and sent to spam folder).
#
# [INTERVIEWER QUESTION 2]: "Define Recall (Sensitivity / True Positive Rate). When is Recall prioritized?"
# [ANSWER]:
# - Formula: Recall = TP / (TP + FN)
# - Meaning: Out of all actual Positive instances, how many did the model successfully identify?
# - Priority: When FALSE NEGATIVES (FN) are critical/dangerous.
#   Example: Cancer Detection or Titanic Rescue (You cannot afford to miss an infected patient or a survivor).
#
# [INTERVIEWER QUESTION 3]: "What is F1-Score and why do we use Harmonic Mean instead of Arithmetic Mean?"
# [ANSWER]:
# - Formula: F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
# - Harmonic Mean penalizes extreme imbalances between Precision and Recall.
# - If a model has Precision = 1.0 and Recall = 0.0, Arithmetic Mean is 0.5 (misleadingly acceptable), but Harmonic Mean drops to 0.0 (correctly penalizing failure).
#
# [INTERVIEWER QUESTION 4]: "What do 'support', 'macro avg', and 'weighted avg' mean in the report?"
# [ANSWER]:
# - Support: The actual number of ground truth instances of each class in y_test (109 for Class 0, 69 for Class 1).
# - Macro Avg: Unweighted arithmetic average of the metric across all classes: (Metric_Class0 + Metric_Class1) / 2.
# - Weighted Avg: Average weighted by the support of each class: (Metric_Class0 * 109 + Metric_Class1 * 69) / 178.

j = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(j)

