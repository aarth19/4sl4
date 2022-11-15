import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt

#COMPENG 4SL4 Assignment 02
#Aarth Patel patea73 400171361

#Getting data from source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
##########################

#Randomly shuffling data using a seed of the last 4 digits of student number 400171361
np.random.seed(1361)
p = np.random.permutation(len(target))
data = data[p]
target = target[p]
##########################

#Splitting the data in half into test and training points
test_data = data[:int(len(target)/2)]
data = data[int(len(target)/2):]
test_target = target[:int(len(target)/2)]
target = target[int(len(target)/2):]
##########################

#Reformatting data into vertical matrices
target.shape = len(target),1
test_target.shape = len(test_target),1
final_features = []
cross_errors = []
x = list(zip(*data))
test_x = list(zip(*test_data))
for i in range(0,13):
    x[i] = np.asarray(x[i]).reshape(len(target),1)
    test_x[i] = np.asarray(test_x[i]).reshape(len(target),1)
##########################

#least squares error function
def get_error(X_test,y_test,w):
    y_pred = X_test.dot(w)
    error = float((1/len(y_test))*(((y_pred - y_test).transpose()).dot((y_pred - y_test)))) #least squares error function is 1/N * (y-t)^T * (y-t)
    return error
###########################

#linear regression training function
def train_lin_reg(X_train,y_train):
    X_train_T = X_train.transpose()
    w = (((np.linalg.inv((X_train_T).dot(X_train))).dot(X_train_T)).dot(y_train)) #applying w = (X^T * X)^-1 * X^t * t
    return w
##########################

#function to create and format the X matrix
def get_X(k,S,x,size,final_features):
    X = np.ones((size,1))
    if k!=0:
        for i in final_features:
            X = np.concatenate((X,x[i]),axis=1)
    X = np.concatenate((X,x[S]),axis=1)
    return X
##########################

#forward stepwise selection algorithm function
def stepwise_sel(x,target):
    for k in range(0,13): #outer loop for each additional selected feature
        feature_error = []
    
        for S in range(0,13): #inner loop to select the next feature
            sub_err = []

            if S in final_features: #ensures a feature can't be used twice
                feature_error.append(np.nan)
                continue

            X = get_X(k,S,x,len(target),final_features)

            kf = KFold(n_splits=10, random_state=None, shuffle = False) #10-fold cross validation
            for train_index, test_index in kf.split(X,target):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = target[train_index], target[test_index]
                w = train_lin_reg(X_train,y_train) #calling the training function
                sub_err.append(get_error(X_test,y_test,w)) #getting the error

            feature_error.append(sum(sub_err)/len(sub_err)) #The cross validation error is the average of the errors in each fold
    
        final_features.append(np.nanargmin(feature_error)) #the feature with the least error is added to the feature list
        cross_errors.append(np.nanmin(feature_error)) #the new error at the current value of k with the selected feature
####################

#basis expansion function
def basis_exp(x,target):
    for S in range(0,13):
        sub_err = []
        X = get_X(S,final_features[S],x,len(target),final_features[:S])
        kf = KFold(n_splits=10, random_state=None, shuffle = False) #10-fold cross validation
        for train_index, test_index in kf.split(X,target):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = target[train_index], target[test_index]
            w = train_lin_reg(X_train,y_train) #calling the training function
            sub_err.append(get_error(X_test,y_test,w)) #getting the error
        cross_errors[S] = (sum(sub_err)/len(sub_err))
####################

#function to plot the test-error and cross-validation error
def make_plots(train_x,train_target,test_x,test_target,error,basis1_error,basis2_error):
    k = range(0,13)
    plt.plot(k,error,'-k',label="Cross-Validation Error")
    plt.plot(k,basis1_error,'-b',label="Cross-Validation Error Basis Function: X^(2)")
    plt.plot(k,basis2_error,'-r',label="Cross-Validation Error Basis Function: X^(0.5)")
    test_error = []
    for i in range(0,13): #training the model on training points, and testing on testing points that we set aside at the start
        X = get_X(i,final_features[i],train_x,len(train_target),final_features[:i])
        w = train_lin_reg(X,train_target)
        X_test = get_X(i,final_features[i],test_x,len(test_target),final_features[:i])
        test_error.append(get_error(X_test,test_target,w))
    plt.plot(k,test_error,'-g',label="Test Error")
    test_basis1_error = []
    for i in range(0,13): #training the model on training points for Basis Function 1
        X = get_X(i,final_features[i],np.power((train_x),2),len(train_target),final_features[:i])
        w = train_lin_reg(X,train_target)
        X_test = get_X(i,final_features[i],np.power((test_x),2),len(test_target),final_features[:i])
        test_basis1_error.append(get_error(X_test,test_target,w))
    plt.plot(k,test_basis1_error,'-c',label="Test Error Basis Function: X^(2)")
    test_basis2_error = []
    for i in range(0,13): #training the model on training points for Basis Function 2
        X = get_X(i,final_features[i],np.sqrt(train_x),len(train_target),final_features[:i])
        w = train_lin_reg(X,train_target)
        X_test = get_X(i,final_features[i],np.sqrt(test_x),len(test_target),final_features[:i])
        test_basis2_error.append(get_error(X_test,test_target,w))
    plt.plot(k,test_basis2_error,'-m',label="Test Error Basis Function: X^(0.5)")
    plt.legend()
    plt.show()
##################

stepwise_sel(x,target)
error = np.copy(cross_errors)
basis_exp(np.power((x),2),target)
basis1_error = np.copy(cross_errors)
basis_exp(np.sqrt(x),target)
basis2_error = np.copy(cross_errors)
make_plots(x,target,test_x,test_target,error,basis1_error,basis2_error)