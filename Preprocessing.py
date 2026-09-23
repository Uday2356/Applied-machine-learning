# ==============================================================================
# SCRIPT: Preprocessing.py (Feature Encoding & Type Conversion)
# TOPIC: Categorical Encoding & Data Type Transformation
# STYLE: Interviewer Q&A + Technical Breakdown
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: IMPORT CLEANED DATASET
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION]: "Why modularize the EDA and Preprocessing scripts?"
# [ANSWER]: Enables clean data pipelines where EDA handles cleaning/imputation,
# and Preprocessing handles encoding/scaling, making components testable.
from EDA import df


# ------------------------------------------------------------------------------
# STEP 2: CATEGORICAL FEATURE ENCODING
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "What is the difference between Nominal and Ordinal categorical data?"
# [ANSWER]:
# - Nominal: Categories with NO intrinsic order/ranking (e.g., Sex: Male/Female, Embarked: C/Q/S).
# - Ordinal: Categories WITH a meaningful order/ranking (e.g., Ticket Class: 1st > 2nd > 3rd, Education: High School < Bachelor < PhD).
#
# [INTERVIEWER QUESTION 2]: "When should you use LabelEncoder vs One-Hot Encoder (pd.get_dummies)?"
# [ANSWER]:
# - LabelEncoder: Converts categories into integers (0, 1, 2...). 
#   * Best used for target variable 'y' or ordinal features.
#   * Caution in linear models: The model may mistakenly interpret integer ranks (e.g., Class 2 > Class 1 > Class 0) as mathematical magnitude.
# - One-Hot Encoding: Creates binary 0/1 indicator columns for each category.
#   * Best for nominal features with low cardinality in linear models to avoid artificial ordering.
#
# [INTERVIEWER QUESTION 3]: "What is the difference between .fit(), .transform(), and .fit_transform()?"
# [ANSWER]:
# - .fit(): Learns the mapping from categories to integers (e.g., 'male' -> 1, 'female' -> 0).
# - .transform(): Applies the learned mapping to transform categories into integers.
# - .fit_transform(): Combines both steps in one call.
#   * NOTE (Data Leakage): In production ML, fit on train set only, then transform on test set.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Embarked column: Ports of embarkation ('S', 'C', 'Q') -> Encoded to [0, 1, 2]
df['embarked'] = le.fit_transform(df['embarked'])

# Sex column: Gender ('male', 'female') -> Encoded to [0, 1]
df['sex'] = le.fit_transform(df['sex'])


# ------------------------------------------------------------------------------
# STEP 3: DATA TYPE CASTING
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Why do we cast the entire dataframe to integer with .astype(int)?"
# [ANSWER]:
# - Converts floats (like Age and Fare) and booleans (like alone) into pure integers.
# - Ensures homogeneous numeric data types for linear algorithms and reduces memory footprint.
#
# [INTERVIEWER QUESTION 2]: "What is the side effect of casting float to int?"
# [ANSWER]:
# - Truncates fractional decimals (e.g., Age 29.69 becomes 29). For exact continuous distributions, keeping float is often preferred.

df = df.astype(int)

# Verify transformed dataset preview
print("\n--- Transformed Dataset Preview (First 10 Rows) ---")
print(df.head(10))








