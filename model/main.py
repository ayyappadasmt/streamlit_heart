import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle as pickle





def create_model(data):
    X=data.drop(["target"],axis=1)
    y=data['target']


    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)
    
    model=LogisticRegression()
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    print("accuracy of the model is:",accuracy_score(y_test,y_pred))
    print("classification report:",classification_report(y_test,y_pred))


    return model,scaler


def get_clean_data():
    data=pd.read_csv("data/heart.csv")


    data=data.drop(["age","sex"],axis=1)
    
    return data


def main():
    data=get_clean_data()

    model,scaler=create_model(data)

    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

   
if __name__=='__main__':
    main()