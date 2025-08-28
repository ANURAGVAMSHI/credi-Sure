import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
label_encoder={}
df=pd.read_csv("/Users/anuragvamshi/Library/Mobile Documents/com~apple~CloudDocs/crediSure/Final dataset.csv")
le_dict = {} 

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X=df.drop("Loan_Status", axis=1)
y=df["Loan_Status"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy : {accuracy * 100:.2f}%")
import joblib
joblib.dump(model,"anurag_model.pkl")
joblib.dump(le_dict,"anurag_encoder.pkl")