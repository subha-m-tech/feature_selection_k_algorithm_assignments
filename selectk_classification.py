# import all the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt


def selectkbest(indep_X,dep_Y,n):
    """This method helps to find best features based on n value using k-algorithm
    with the help of chi square score function"""

    test = SelectKBest(score_func=chi2, k=n)
    fit1= test.fit(indep_X,dep_Y)
    selectk_features = fit1.transform(indep_X)
    return selectk_features, test.get_feature_names_out()
    
def split_scalar(indep_X,dep_Y):
    """This method takes independent and dependent varaibles and split the dataset into training
    and test data"""

    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

 
def cm_prediction(classifier,X_test,y_test):
    """This method gives conflusion matrix values based on the  test data and model"""
    y_pred = classifier.predict(X_test)
        
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
        
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
        
    Accuracy=accuracy_score(y_test, y_pred )
        
    report=classification_report(y_test, y_pred)
    return  classifier,Accuracy,report,X_test,y_test,cm

def logistic(X_train,y_train,X_test, y_test):  
    """This method takes training data and input test data, create logistic models
    and finally calculate confusion matrix and returns model object with metrics"""
      
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm      
    
def svm_linear(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_linear models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
    
def svm_NL(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate confusion matrix and returns model object with metrics"""
                
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
   
def Navie(X_train,y_train,X_test, y_test):   
    """This method takes training data and input test data, create Navie models
    and finally calculate confusion matrix and returns model object with metrics"""
    
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm         

    
def knn(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create knn models
    and finally calculate confusion matrix and returns model object with metrics"""

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
    
def Decision(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create Decision models
    and finally calculate confusion matrix and returns model object with metrics"""

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm      


def random_forest(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create random forest models
    and finally calculate confusion matrix and returns model object with metrics"""

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test, y_test)
    return  classifier,Accuracy,report,X_test,y_test,cm
    
def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
    """This method returns dataframe with accuracy of different alogorithms"""
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])
    for number,idex in enumerate(dataframe.index):      
        dataframe['Logistic'][idex]=acclog[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['KNN'][idex]=accknn[number]
        dataframe['Navie'][idex]=accnav[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe