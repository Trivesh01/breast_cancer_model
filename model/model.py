import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle 




def create_model(data):
    X=data.drop('diagnosis',axis=1).values
    y=data['diagnosis'].values

    #scale the data 
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)

    #split the data 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


    #train the model
    model=LogisticRegression()
    model.fit(X_train,y_train)

    #test
    y_pred=model.predict(X_test)
    print('accuracy is :',accuracy_score)
    print('classification report: \n',classification_report(y_test,y_pred))

    

    return model,scaler





def get_clean_data():
    df=pd.read_csv('data/b_c.csv')
    df=df.drop(['Unnamed: 32','id'],axis=1)
    df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
    return df


def main():
    df=get_clean_data()
    model,scaler=create_model(df)
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('model/scaler.pkl','wb') as g:
        pickle.dump(scaler,g)


if __name__=='__main__':
    main()