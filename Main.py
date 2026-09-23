import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns






# Load Titanic dataset from Seaborn's repository
df = sns.load_dataset("titanic")




a = df.size          # Total data points in DataFrame
b = df.columns       # Index object containing all column headers
c = df.isna().sum()  # Count of null/NaN values per column
d = df.shape         # (rows, cols) dimensions

print("Dataset Info:")
df.info()

print("\nStatistical Summary of Numerical Features:")
print(df.describe())



df.drop(
    columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'],
    inplace=True
)


 

# Drop the 2 rows where 'embarked' is missing
df.dropna(subset=['embarked'], inplace=True)

# Impute missing 'age' values with the column mean
df['age'] = df['age'].fillna(df['age'].mean())







from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Embarked column: Ports of embarkation ('S', 'C', 'Q') -> Encoded to [0, 1, 2]
df['embarked'] = le.fit_transform(df['embarked'])

# Sex column: Gender ('male', 'female') -> Encoded to [0, 1]
df['sex'] = le.fit_transform(df['sex'])


df = df.astype(int)

# Verify transformed dataset preview
# print("\n--- Transformed Dataset Preview (First 10 Rows) ---")
# print(df.head(10))







# Drop target column to isolate all independent features
x = df.drop('survived', axis=1) 

# Extract target column as dependent variable
y = df['survived']




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    random_state=42
)


 #FOR KNN WE NEED TO SCALE THE COLUMN SO WE USE STANDARDSCALING

from sklearn.preprocessing  import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


from sklearn.neighbors import KNeighborsClassifier

#here we need to specify the value of K = n_neighbors
model_knn = KNeighborsClassifier(n_neighbors=5) 


#NOw make model as 

model_knn.fit(x_train_scaled,y_train)

y_pred_k = model_knn.predict(x_test_scaled)



 #MODEL EVALUATION 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

a = accuracy_score(y_test,y_pred_k)

v = confusion_matrix(y_pred_k,y_test)



b = classification_report(y_pred_k,y_test)


#**********************************NAIVE BYES ********************************************#


from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()

model_nb.fit(x_train,y_train)

y_pred_nb = model_nb.predict(x_test)


c = accuracy_score(y_test,y_pred_nb)

n = confusion_matrix(y_test,y_pred_nb)


k = classification_report(y_test,y_pred_nb)


# with non standard daata the accuracy and the preciosn was arround 77 to 78 % both 
# Now wee will use standardized data 


model_nb.fit(x_train_scaled,y_train)

y_scaled_prdict_nb = model_nb.predict(x_test_scaled)


t = accuracy_score(y_test,y_scaled_prdict_nb)

f = classification_report(y_test,y_scaled_prdict_nb)




#******************************DECISION TREE MODEL******************#

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier(random_state=42)

model_tree.fit(x_train_scaled,y_train)


y_tree_pred = model_tree.predict(x_test_scaled)

g = confusion_matrix(y_test, y_tree_pred)

f = classification_report(y_test, y_tree_pred)





#***********************************Support Vector Machine (SVM)***************

from sklearn.svm import SVC

model_svm = SVC(kernel='rbf')


model_svm.fit(x_train_scaled,y_train)

y_svm = model_svm.predict(x_test_scaled)

# print(y_svm)

y = classification_report(y_test,y_svm)

r = accuracy_score(y_test,y_svm)
# print(r)

t = confusion_matrix(y_test,y_svm)
print(t)


#FOR TITANIC DATASET THE SVM GIVES A PROPER CALLSIFICATION REPORT BETTER THAN OTHER MODELS

