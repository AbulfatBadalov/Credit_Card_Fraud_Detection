import pandas as pd
data=pd.read_csv('creditcard.csv')
pd.options.display.max_columns=None
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))
data=data.drop(['Time'],axis=1)
data.duplicated().any()
data=data.drop_duplicates()
X=data.drop('Class',axis=1)
y=data['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
classifier={
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"\n Accuracy : {accuracy_score(y_test,y_pred)}")
    print(f"\n Recall : {recall_score(y_test,y_pred)}")
    print(f"\n Precision : {precision_score(y_test,y_pred)}")
    print(f"\n F1 Score : {f1_score(y_test,y_pred)}")
    #Undersampling
normal=data[data['Class']==0]
fraud=data[data['Class']==1]
normal_sample=normal.sample(n=473)
normal_sample.shape
new_data=pd.concat([normal_sample,fraud],ignore_index=True)
X=new_data.drop('Class',axis=1)
y=new_data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)
classifier={
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"\n Accuracy : {accuracy_score(y_test,y_pred)}")
    print(f"\n Recall : {recall_score(y_test,y_pred)}")
    print(f"\n Precision : {precision_score(y_test,y_pred)}")
    print(f"\n F1 Score : {f1_score(y_test,y_pred)}")
    #OverSampling
X=data.drop('Class',axis=1)
y=data['Class']
from imblearn.over_sampling import SMOTE
X_res,y_res=SMOTE().fit_resample(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)
classifier={
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"\n Accuracy : {accuracy_score(y_test,y_pred)}")
    print(f"\n Recall : {recall_score(y_test,y_pred)}")
    print(f"\n Precision : {precision_score(y_test,y_pred)}")
    print(f"\n F1 Score : {f1_score(y_test,y_pred)}")
dtc=DecisionTreeClassifier()
dtc.fit(X_res,y_res)
import joblib
joblib.dump(dtc,"credit_card_model.pkl")
model=joblib.load("credit_card_model.pkl")