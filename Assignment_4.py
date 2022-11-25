import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt

#COMPENG 4SL4 Assignment 04
#Aarth Patel patea73 400171361

def misclass_rate(pred,target):
    #a target is misclassified when pred=1 and target=0 or when pred=0 and target=1, thus we can use XOR
    return np.sum(np.logical_xor(pred,target))/len(target)

dataset = pd.read_csv('spambase.data', header=None)
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values
X_train, X_test, target_train, target_test = train_test_split(X, t, test_size=0.33, random_state=1361)

plt.figure(1)
avg_error = []
for i in range(2,401):
    cross_error = []
    kf = KFold(n_splits=5, random_state=1361, shuffle = True) #5-fold cross validation
    for train_index, test_index in kf.split(X_train,target_train):
        X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
        y_train_kf, y_test_kf = target_train[train_index], target_train[test_index]
        pred = (DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train_kf,y_train_kf)).predict(X_test_kf)
        cross_error.append(misclass_rate(pred,y_test_kf)) #cross-validation error is misclassification rate
    avg_error.append(np.sum(cross_error)/len(cross_error)) #error for each number of leaves
plt.plot(range(2,401),avg_error)

plt.figure(2)
estimators = range(50,2501,50)
best_leaves = np.argmin(avg_error) + 2
pred = (DecisionTreeClassifier(max_leaf_nodes=best_leaves).fit(X_train,target_train)).predict(X_test)
error = misclass_rate(pred,target_test)
plt.plot(estimators,np.ones(len(estimators))*error,'-m',label='Best Decision Tree Classifier')

avg_error = []
for i in estimators:
    pred = (BaggingClassifier(n_estimators=i, random_state=1361).fit(X_train, target_train)).predict(X_test)
    avg_error.append(misclass_rate(pred,target_test))
plt.plot(estimators,avg_error,'-r',label='Bagging Classifier')

avg_error = []
for i in estimators:
    pred = (RandomForestClassifier(n_estimators=i, random_state=1361).fit(X_train, target_train)).predict(X_test)
    avg_error.append(misclass_rate(pred,target_test))
plt.plot(estimators,avg_error,'-g',label='Random Forest Classifier')

avg_error = []
for i in estimators:
    pred = (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=i, random_state=1361).fit(X_train, target_train)).predict(X_test)
    avg_error.append(misclass_rate(pred,target_test))
plt.plot(estimators,avg_error,'-b',label='Adaboost with Decision Stump')

avg_error = []
for i in estimators:
    pred = (AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=10),n_estimators=i, random_state=1361).fit(X_train, target_train)).predict(X_test)
    avg_error.append(misclass_rate(pred,target_test))
plt.plot(estimators,avg_error,'-k',label='Adaboost with Decision Tree with 10 Leaves')

avg_error = []
for i in estimators:
    pred = (AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=i, random_state=1361).fit(X_train, target_train)).predict(X_test)
    avg_error.append(misclass_rate(pred,target_test))
plt.plot(estimators,avg_error,'-c',label='Adaboost with Unlimited Decision Tree')
plt.legend()
plt.show()

avg_error = []
for i in estimators:
    pred = (AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=i, random_state=1361).fit(X_train, target_train)).predict(X_test)
    avg_error.append(misclass_rate(pred,target_test))
plt.plot(estimators,avg_error,'-c',label='Adaboost with Unlimited Decision Tree')
plt.legend()
plt.show()