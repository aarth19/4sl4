import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

#COMPENG 4SL4 Assignment 03
#Aarth Patel patea73 400171361

#My own gradient descent implementation for logistic regression
def train_logreg(alpha,eta,X_train,target_train,X_test,target_test):
    w_minus_1 = np.zeros((31,1))
    while(True):
        y = 1/(1 + np.exp(-1*X_train.dot(w_minus_1)))
        delta_c = (1/len(target_train))*X_train.transpose().dot(y - target_train)
        if(all(abs(ele) < eta for ele in delta_c)):
            break
        w_minus_1 = w_minus_1 - alpha*delta_c
    w = w_minus_1
    pred = X_test.dot(w)
    #returning precision recall curve using thresholds from prediction
    precision = []
    recall = []
    for j in range(len(pred)):
        pred = X_test.dot(w)
        #using each value of the prediction as a theta value
        theta = pred[j]
        for i in range(len(pred)):
            if (pred[i] >= theta):
                pred[i] = 1
            else:
                pred[i] = 0
        #true positive is when pred=1 and target=1
        true_positive = np.sum(np.logical_and(pred,target_test))
        precision.append(true_positive/np.sum(pred))
        recall.append(true_positive/np.sum(target_test))
    return w,precision,recall
########################################

#function for getting statistics (misclassification, precision, recall) based on the prediction and the target
def pred_statistics(pred,target_test):
    #a target is misclassified when pred=1 and target=0 or when pred=0 and target=1, thus we can use XOR
    misclassification_rate = np.sum(np.logical_xor(pred,target_test))/len(target_test)
    true_positive = np.sum(np.logical_and(pred,target_test))
    zero_precision = true_positive/np.sum(pred)
    zero_recall = true_positive/np.sum(target_test)
    f1 = 2*zero_precision*zero_recall/(zero_precision + zero_recall)
    return misclassification_rate,zero_precision,zero_recall,f1
##############################################

#algorithm for getting nearest neighbor based prediction
def nearest_neighbor(X_test_kf,X_train_kf,y_train_kf,k):
    pred = []
    for x in X_test_kf:
        #find distance to each training point for each test point
        dist = []
        for i in range(len(X_train_kf)):
            dist.append(np.linalg.norm(x - X_train_kf[i]))
        #sort by distance to training point
        order = np.asarray(dist).argsort()
        sorted_y_train_kf = y_train_kf[order]
        #assign a target by looking at k closest training points
        ksum = np.sum(sorted_y_train_kf[0:k])
        if(ksum > k/2):
            pred.append(1)
        elif(ksum < k/2):
            pred.append(0)
        elif(ksum == k/2):
            ksum = np.sum(sorted_y_train_kf[0:k+1])
            if(ksum > (k+1)/2):
                pred.append(1)
            else:
                pred.append(0)
    return pred
################

#Getting data and targets from the source and shuffling
data, target = load_breast_cancer(return_X_y=True)
target.shape = len(target),1
X_train, X_test, target_train, target_test = train_test_split(data, target, test_size=0.33, random_state=1361)

#Feature standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = np.concatenate((np.ones((len(target_train),1)),X_train),axis=1)
X_test = np.concatenate((np.ones((len(target_test),1)),X_test),axis=1)

w,precision,recall = train_logreg(0.15,0.01,X_train,target_train,X_test,target_test)
pred = X_test.dot(w)
for i in range(len(pred)):
    #zero threshold minimizes misclassification rate
    if (pred[i] >= 0):
        pred[i] = 1
    else:
        pred[i] = 0
misclassification_rate,zero_precision,zero_recall,f1 = pred_statistics(pred,target_test)

#scikit implementation for logistic regression
clf = LogisticRegression(random_state=1361).fit(X_train, target_train.reshape(len(target_train), ))
pred = clf.predict(X_test).reshape(len(pred),1)
pred_p  = list(zip(*(clf.predict_proba(X_test))))
precision_scikit, recall_scikit, threshold = precision_recall_curve(np.array(target_test.tolist()).ravel(), pred_p[0])
misclassification_rate,zero_precision,zero_recall,f1 = pred_statistics(pred,target_test)

#k-nearest-neighbors implementation
avg_error = []
for k in range(1,6):
    cross_error = []
    #Kfold cross validation to choose which value of k for nearest neighbors
    kf = KFold(n_splits=5, random_state=1361, shuffle = True) #5-fold cross validation
    for train_index, test_index in kf.split(X_train,target_train):
        X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
        y_train_kf, y_test_kf = target_train[train_index], target_train[test_index]
        pred = nearest_neighbor(X_test_kf,X_train_kf,y_train_kf,k)
        #cross-validation error is misclassification rate
        cross_error.append(np.sum(np.logical_xor(pred,y_test_kf.ravel().tolist()))/len(y_test_kf))
    #error for each value of k
    avg_error.append(np.sum(cross_error)/len(cross_error))

pred = nearest_neighbor(X_test,X_train,target_train,5)
test_error = np.sum(np.logical_xor(pred,target_test.ravel().tolist()))/len(target_test)

#scikit k nearest neighbor implementation
avg_error = []
for k in range(1,6):
    cross_error = []
    kf = KFold(n_splits=5, random_state=1361, shuffle = True) #5-fold cross validation
    for train_index, test_index in kf.split(X_train,target_train):
        X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
        y_train_kf, y_test_kf = target_train[train_index], target_train[test_index]
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train_kf,y_train_kf.ravel().tolist())
        pred = neigh.predict(X_test_kf)
        cross_error.append(np.sum(np.logical_xor(pred,y_test_kf.ravel().tolist()))/len(y_test_kf))
    avg_error.append(np.sum(cross_error)/len(cross_error))

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train,target_train.ravel().tolist())
pred = neigh.predict(X_test)
test_error = np.sum(np.logical_xor(pred,target_test.ravel().tolist()))/len(target_test)