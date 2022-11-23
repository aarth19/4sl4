from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

def misclass_rate(pred,target):
    #a target is misclassified when pred=1 and target=0 or when pred=0 and target=1, thus we can use XOR
    return np.sum(np.logical_xor(pred,target))/len(target)

dataset = pd.read_csv('spambase/spambase.data', header=None)
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values
X_train, X_test, target_train, target_test = train_test_split(X, t, test_size=0.33, random_state=1361)

avg_error = []
#for
cross_error = []
kf = KFold(n_splits=5, random_state=1361, shuffle = True) #5-fold cross validation
for train_index, test_index in kf.split(X_train,target_train):
    X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
    y_train_kf, y_test_kf = target_train[train_index], target_train[test_index]
    pred = (DecisionTreeClassifier(max_leaf_nodes=2).fit(X_train_kf,y_train_kf)).predict(X_test_kf)
    cross_error.append(misclass_rate(pred,y_test_kf))
avg_error.append(np.sum(cross_error)/len(cross_error))
print(cross_error)