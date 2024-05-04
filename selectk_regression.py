# Imports all the required libraries
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
    
def r2_prediction(regressor,X_test,y_test):
    """This method gives r2 score values based on the  test data and model"""
    
    y_pred = regressor.predict(X_test)
    from sklearn.metrics import r2_score
    r2=r2_score(y_test,y_pred)
    return r2
 
def Linear(X_train,y_train,X_test, y_test):       
   """This method takes training data and input test data, create Linear models
    and finally calculate R2 score and returns r2 score"""
   from sklearn.linear_model import LinearRegression
   regressor = LinearRegression()
   regressor.fit(X_train, y_train)
   r2=r2_prediction(regressor,X_test,y_test)
   return  r2   
    
def svm_linear(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_linear models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'linear')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
    
def svm_NL(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
 

def Decision(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
 

def random_forest(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create random forest models
    and finally calculate R2 score and returns r2 score"""
    
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2 
    
    
def selectk_regression(acclin,accsvml,accsvmnl,accdes,accrf): 
    """This method returns dataframe with accuracy of different alogorithms"""
    
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Linear','SVMl','SVMnl','Decision','Random'
                                                                                     ])

    for number,idex in enumerate(dataframe.index):
        
        dataframe['Linear'][idex]=acclin[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe