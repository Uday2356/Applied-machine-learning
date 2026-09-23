# ==============================================================================
# SCRIPT: EDA.py (Exploratory Data Analysis & Data Cleaning)
# TOPIC: Data Inspection, Feature Pruning, & Missing Value Imputation
# STYLE: Interviewer Q&A + Technical Breakdown
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: IMPORTING ESSENTIAL DATA SCIENCE LIBRARIES
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION]: "What is the primary role of each library in the ML stack?"
# [ANSWER]:
# - pandas (pd): Data manipulation, cleaning, and tabular DataFrame structures.
# - seaborn (sns): Statistical data visualization and built-in benchmark datasets.
# - numpy (np): Fast numerical computing, multi-dimensional arrays, and vectorized math.
# - matplotlib.pyplot (plt): Fundamental 2D plotting engine and canvas customization.

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load Titanic dataset from Seaborn's repository
df = sns.load_dataset("titanic")


# ------------------------------------------------------------------------------
# STEP 2: DATASET INSPECTION & STRUCTURAL INTEGRITY
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "What is the difference between df.shape, df.size, and len(df)?"
# [ANSWER]:
# - df.shape: Returns a tuple (num_rows, num_columns) -> e.g., (891, 15).
# - df.size: Returns total number of individual cells (num_rows * num_columns) -> e.g., 891 * 15 = 13365.
# - len(df): Returns the number of rows (891).
#
# [INTERVIEWER QUESTION 2]: "Why is df.info() and df.describe() the first step in EDA?"
# [ANSWER]:
# - df.info(): Reveals column data types, non-null counts, and memory usage.
# - df.describe(): Provides 5-number summary (min, 25%, 50%/median, 75%, max) + mean & std to detect skewness and outliers.

a = df.size          # Total data points in DataFrame
b = df.columns       # Index object containing all column headers
c = df.isna().sum()  # Count of null/NaN values per column
d = df.shape         # (rows, cols) dimensions

print("Dataset Info:")
df.info()

print("\nStatistical Summary of Numerical Features:")
print(df.describe())


# ------------------------------------------------------------------------------
# STEP 3: FEATURE SELECTION & DROPPING REDUNDANT / LEAKAGE COLUMNS
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Why do we drop 'deck', 'embark_town', 'alive', 'class', 'who', and 'adult_male'?"
# [ANSWER]:
# 1. 'deck': Over 77% missing values (688 nulls out of 891). Imputing too many missing values adds noise.
# 2. 'alive': Direct duplicate of target 'survived' ('yes'/'no' vs 1/0). Keeping it causes TARGET LEAKAGE (100% false accuracy).
# 3. 'embark_town': Exact duplicate of 'embarked' ('Southampton' vs 'S'). Keeping both introduces MULTICOLLINEARITY.
# 4. 'class': Exact duplicate of 'pclass' ('First' vs 1).
# 5. 'who' & 'adult_male': Derived combinations of 'sex' and 'age', causing redundant multicollinearity.

df.drop(
    columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'],
    inplace=True
)


# ------------------------------------------------------------------------------
# STEP 4: HANDLING MISSING VALUES (IMPUTATION STRATEGIES)
# ------------------------------------------------------------------------------
# [INTERVIEWER QUESTION 1]: "Why do we use dropna() on 'embarked' but fillna() on 'age'?"
# [ANSWER]:
# - 'embarked' has only 2 missing values out of 891 (< 0.3%). Dropping 2 rows has zero statistical impact.
# - 'age' has 177 missing values (~20%). Dropping 20% of rows would discard valuable training data.
#
# [INTERVIEWER QUESTION 2]: "When should you use Mean vs Median vs Mode for imputation?"
# [ANSWER]:
# - Mean: Best for normally distributed data without extreme outliers.
# - Median: Best for skewed data or distributions with heavy outliers (robust to extreme values).
# - Mode: Used for categorical/discrete variables.
# - Advanced methods: KNN Imputation, IterativeImputer (MICE), or domain-specific grouping (e.g., median age by pclass).

# Drop the 2 rows where 'embarked' is missing
df.dropna(subset=['embarked'], inplace=True)

# Impute missing 'age' values with the column mean
df['age'] = df['age'].fillna(df['age'].mean())

print("\n--- Cleaned Dataset Null Check ---")
print(df.isna().sum())

