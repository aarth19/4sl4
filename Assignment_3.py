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
def train_logreg(alpha,eta,X_train,target_train):
    w_minus_1 = np.zeros((31,1))
    while(True):
        y = 1/(1 + np.exp(-1*X_train.dot(w_minus_1)))
        delta_c = (1/len(target_train))*X_train.transpose().dot(y - target_train)
        if(all(abs(ele) < eta for ele in delta_c)):
            break #break when change in cost function is below threshold of eta
        w_minus_1 = w_minus_1 - alpha*delta_c
    return w_minus_1
######################################

#Function to obtain precision recall curve values
def prec_recall_curve(pred,target_test):
    pred = X_test.dot(w)
    precision = []
    recall = []
    thres = np.sort(pred.ravel().tolist())
    for j in range(len(pred)):
        new_pred = np.copy(pred)
        theta = thres[j] #each value of predicition becomes a possible threshold
        for i in range(len(pred)):
            if (new_pred[i] >= theta):
                new_pred[i] = 1
            else:
                new_pred[i] = 0
        zero_precision,zero_recall,f1 = pred_statistics(new_pred,target_test)
        precision.append(zero_precision)
        recall.append(zero_recall)
    return precision,recall
######################################

def misclass_rate(pred,target_test):
    #a target is misclassified when pred=1 and target=0 or when pred=0 and target=1, thus we can use XOR
    return np.sum(np.logical_xor(pred,target_test))/len(target_test)

#function for getting precision, recall, and f1 based on the prediction and the target
def pred_statistics(pred,target_test):
    #true positive is when pred = 1 and target = 1, thus we can use AND
    true_positive = np.sum(np.logical_and(pred,target_test))
    zero_precision = true_positive/np.sum(pred)
    zero_recall = true_positive/np.sum(target_test)
    f1 = 2*zero_precision*zero_recall/(zero_precision + zero_recall)
    return zero_precision,zero_recall,f1
######################################

#function for getting nearest neighbor based prediction
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
        sorted_dist = np.asarray(dist)[order]
        #assign a target by looking at k closest training points
        ksum = np.sum(sorted_y_train_kf[0:k])
        if(ksum > k/2):
            pred.append(1)
        elif(ksum < k/2):
            pred.append(0)
        elif(ksum == k/2): #tie-breaker by weighing distances
            one_distance = 0
            zero_distance = 0
            for targ in range(0,k):
                if sorted_y_train_kf[targ] > 0:
                    one_distance += sorted_dist[targ]
                else:
                    zero_distance += sorted_dist[targ]
            if one_distance < zero_distance:
                pred.append(1)
            else:
                pred.append(0)
    return pred
######################################

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

#obtaining prediction and precision recall curve
w = train_logreg(0.15,0.01,X_train,target_train)
pred = X_test.dot(w)
precision, recall = prec_recall_curve(pred,target_test)
plt.figure(0)
plt.plot(recall,precision,'-k')
plt.xlabel("Recall")
plt.ylabel("Precision")

#Classifying with zero threshold to minimize misclassification rate
for i in range(len(pred)):
    if (pred[i] >= 0):
        pred[i] = 1
    else:
        pred[i] = 0
misclassification_rate = misclass_rate(pred,target_test)
zero_precision,zero_recall,f1 = pred_statistics(pred,target_test)
print("Aarth's Logistic Regression\nMisclassification Rate: " + str(misclassification_rate) + "\nF1 Score: " + str(f1))

#scikit implementation for logistic regression
clf = LogisticRegression(random_state=1361).fit(X_train, target_train.reshape(len(target_train), ))
pred = clf.predict(X_test).reshape(len(pred),1)
pred_p  = list(zip(*(clf.predict_proba(X_test))))

misclassification_rate = misclass_rate(pred,target_test)
zero_precision,zero_recall,f1 = pred_statistics(pred,target_test)
print("\nScikit's Logistic Regression\nMisclassification Rate: " + str(misclassification_rate) + "\nF1 Score: " + str(f1))

#scikit implementation for precision recall curve
precision_scikit, recall_scikit, threshold = precision_recall_curve(target_test.ravel().tolist(), pred_p[1])
plt.figure(1)
plt.plot(recall_scikit,precision_scikit,'-k')
plt.xlabel("Recall")
plt.ylabel("Precision")
ax = plt.gca()
ax.set_ylim([0.5,1.01])

#k-nearest-neighbors implementation with Kfold to decide k
avg_error = []
for k in range(1,6):
    cross_error = []
    kf = KFold(n_splits=5, random_state=1361, shuffle = True) #5-fold cross validation
    for train_index, test_index in kf.split(X_train,target_train):
        X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
        y_train_kf, y_test_kf = target_train[train_index], target_train[test_index]
        pred = nearest_neighbor(X_test_kf,X_train_kf,y_train_kf,k)
        cross_error.append(misclass_rate(pred,y_test_kf.ravel().tolist())) #cross-validation error is misclassification rate
    avg_error.append(np.sum(cross_error)/len(cross_error)) #error for each value of k
print("\nAarth's Nearest Neighbor\nCross Validation Error for each k: " + str(avg_error))

#using k=4 since it minimizes the cross validation error
pred = nearest_neighbor(X_test,X_train,target_train,4)
test_error = misclass_rate(pred,target_test.ravel().tolist())
zero_precision,zero_recall,f1 = pred_statistics(pred,target_test.ravel().tolist())
print("Misclassification Rate: " + str(test_error) + "\nF1 Score: " + str(f1))

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
        cross_error.append(misclass_rate(pred,y_test_kf.ravel().tolist()))
    avg_error.append(np.sum(cross_error)/len(cross_error))
print("\nScikit's Nearest Neighbor\nCross Validation Error for each k: " + str(avg_error))

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train,target_train.ravel().tolist())
pred = neigh.predict(X_test)
test_error = misclass_rate(pred,target_test.ravel().tolist())
zero_precision,zero_recall,f1 = pred_statistics(pred,target_test.ravel().tolist())
print("Misclassification Rate: " + str(test_error) + "\nF1 Score: " + str(f1))

plt.show()