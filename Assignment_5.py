import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler

#COMPENG 4SL4 Assignment 05
#Aarth Patel patea73 400171361

#reading data from file and splitting it 60% training, 20% validation, and 20% test
data,target,empty = np.hsplit(genfromtxt('data_banknote_authentication.txt', delimiter=','),[4,5])
X_train, X_val_test, target_train, target_val_test = train_test_split(data, target, test_size=0.4, random_state=1361)
X_val, X_test, target_val, target_test = train_test_split(X_val_test, target_val_test, test_size=0.5, random_state=1361)

#standardizing data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)