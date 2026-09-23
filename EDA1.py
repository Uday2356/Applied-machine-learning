import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  StandardScaler
from EDA import df






#stablish the corraltion between the heart disease 

#we use heat map for coorelatin check 

plt.figure(figsize=[8,6])
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.savefig("Correlation Between All domain ")
plt.close('all')

f = df.groupby('Sex')['HeartDisease'].value_counts()
print(f)

#now apply encoding over the CHESTPAINTYPE coloumn

# df_cleaned = pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)


heart_gender = df.groupby(['Sex', 'HeartDisease']).size().reset_index(name='Count')

print(heart_gender)
sns.barplot(data = heart_gender, x = 'Sex' , y='Count' , hue = 'HeartDisease')
plt.savefig('Heart_gender')

df = pd.get_dummies(df,columns=['ChestPainType'],drop_first=True)


df['ChestPainType_ATA'] = df['ChestPainType_ATA'].map({False:0,True:1})
df['ChestPainType_NAP'] = df['ChestPainType_NAP'].map({False:0,True:1})
df['ChestPainType_TA'] = df['ChestPainType_TA'].map({False:0 , True : 1})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y':1 , 'N' :0})
df = pd.get_dummies(df,columns=['Sex'] , drop_first=True) #FOR MALE ITS 1 AND FOR FEMALE ITS 0
df['Sex_M'] = df['Sex_M'].map({False:0 , True : 1})

df['ST_Slope'] = df['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': 2})
df['RestingECG'] = df['RestingECG'].map({'Normal': 1, 'ST': 0, 'LVH': 2})

# print(df.head())
# print(df.isnull().sum())

df_encoded = pd.get_dummies(df , drop_first=True)
df_encoded = df_encoded.astype(int)
# print(df_encoded.head())




# first see the count og each value in the column for categorical data 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score


X = df_encoded.drop('HeartDisease',axis=1)
Y = df_encoded['HeartDisease']


X_train , X_test , Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)



models = {
    'Logistic' : LogisticRegression(),
    'Naive' : GaussianNB(),
    'KNN' : KNeighborsClassifier(),
    'SVM(RBF)' : SVC(probability=True),
    'Decision' : DecisionTreeClassifier()
}
result = []
for name , model in models.items():
    model.fit(x_train_scaled,Y_train)
    y_pred = model.predict(x_test_scaled)

    accu = accuracy_score(Y_test,y_pred)
    f1 = f1_score(Y_test,y_pred)

    result.append({
        'model' : name,
        "accuracy" : round(accu,4),
        "f1_score" : round(f1,4)
    })
print(result)





#here we get maximum accuracy of KNN as 86 percent

#now we store the height in the pickel file as we can take it out whever we want as bytestream.
import joblib
joblib.dump(models['KNN'],'KNN_heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(X.columns.to_list(),'columns.pkl')


# Import the pickle module in your Python script.

# To save data temporarily in memory, use pickle.dumps(): Pass the Python object to pickle.dumps() to convert it into a byte stream.

# To save data permanently to a file, use pickle.dump(): Open a file in write binary mode (wb) using the open() function. Pass the Python object and the file object to pickle.dump() to serialize and save the data.

# Loading Data (Deserialization)

# To load data from memory, use pickle.loads(): Pass the byte stream to pickle.loads() to convert it back into the original Python object.

# To load data from a file, use pickle.load(): Open the file in read binary mode (rb) using the open() function. Pass the file object to pickle.load() to deserialize and retrieve the original Python object.