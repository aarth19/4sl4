import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#COMPENG 4SL4 Assignment 01
#Aarth Patel 400171361 patea73

def train_models(X_train,t_train,w):
    X = np.ones((10,1)) #initializing X as a column of 1s
    X_train.shape = 10,1
    t_train.shape = 10,1 #making X_t and t_t into column vectors
    for i in range(0,10):
        w.append(((np.linalg.inv((X.transpose()).dot(X))).dot(X.transpose())).dot(t_train)) #applying w = (X^T * X)^-1 * X^t * t for each M
        X = np.concatenate((X,np.power(X_train,i+1)),axis=1) #expanding the X matrix for each additional dimension added by the next M

def get_error(X_values,t_values,XX_values,w,err):
    size = len(X_values) #getting the size of the set, as this function works for both training and validation error
    X = np.ones((size,1))
    X_values.shape = size,1
    t_values.shape = size,1 #making everything into column vectors
    for i in range(0,10):
        y = X.dot(w[i])
        err.append(float((1/size)*(((y - t_values).transpose()).dot((y - t_values))))) #least squares error function is 1/N * (y-t)^T * (y-t)
        X = np.concatenate((X,np.power(X_values,i+1)),axis=1) #expanding X matrix for next M
    y = XX_values[0].dot(w[10])
    err.append(float((1/size)*(((y - t_values).transpose()).dot((y - t_values)))))
    y = XX_values[0].dot(w[11])
    err.append(float((1/size)*(((y - t_values).transpose()).dot((y - t_values)))))
    

def disp_plots(X_train,X_valid,X_true,t_train,t_valid,t_true,w,XX_valid,train_err,valid_err,avg_error):
    X = np.ones((100,1)) 
    X_values = X_true.reshape(100,1) #Need a 1000 set of X values to make plots of predictor function f(x) clean
    for i in range(0,12):
        plt.figure(i+1)
        plt.title("M = " + str(i))
        plt.plot(X_train,t_train, 'b.',label="Training Points") #plotting training set in blue dots
        plt.plot(X_valid,t_valid, 'r.',label="Validation Points") #plotting validation set in red dots
        plt.plot(X_true,t_true, '-g',label="F_true(x)") #f_true with green line
        if i == 10:
            plt.title("M = 9, lambda = e^-18")
            y = XX_valid.dot(w[10])
            plt.plot(X_valid,y, '-k',label="Predictor Function") #predictor with black line
        elif i == 11:
            plt.title("M = 9, lambda = 1")
            y = XX_valid.dot(w[11])
            plt.plot(X_valid,y, '-k',label="Predictor Function") #predictor with black line
        else:
            y = X.dot(w[i])
            plt.plot(X_true,y, '-k',label="Predictor Function") #predictor with black line
        plt.legend()
        X = np.concatenate((X,np.power(X_values,i+1)),axis=1)
    plt.figure(13)
    plt.title("Error vs M")
    M = range(0,12)
    plt.plot(M,train_err,"-r",label="Training Error MSE")
    plt.plot(M,valid_err, "-g",label="Validation Error MSE")
    plt.plot(M,np.ones(12)*avg_error, "-b",label="Average Error of Validation vs. F_true")
    plt.legend()
    plt.show()

def regularize(X_train,X_valid,t_train,XX_train,XX_valid,w,lambda_1):
    XX_t = X_train
    XX_v = X_valid
    X_train.shape = (10,1)
    X_valid.shape = (100,1)
    for i in range(2,10):
        XX_t = np.concatenate((XX_t,np.power(X_train,i)),axis=1)
        XX_v = np.concatenate((XX_v,np.power(X_valid,i)),axis=1)#Creating feature matrices from the training and validation points
    sc = StandardScaler()
    XX_t = sc.fit_transform(XX_t)
    XX_v = sc.transform(XX_v) #Standardizing the feature matrices so we can use regularization
    XX_t = np.concatenate(((np.ones((10,1))),XX_t),axis=1)
    XX_v = np.concatenate(((np.ones((100,1))),XX_v),axis=1)
    t_train.shape = (10,1)
    B = np.ones(10)*lambda_1*2
    B[0] = 0
    B = np.diag(B) #Creating the B matrix which is 2*lambda on the diagonal except for B00, which is 0
    w.append(((np.linalg.inv((XX_t.transpose()).dot(XX_t) + 5*B)).dot(XX_t.transpose())).dot(t_train)) #Using the formula for w with regularization
    XX_train.append(XX_t)
    XX_valid.append(XX_v)

X_train = np.linspace(0.,1.,10) # training set
X_valid = np.linspace(0.,1.,100) # validation set
X_true = np.linspace(0.,1.,100) #true value set
np.random.seed(1361) # last 4 digits of 400171361
t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)
t_true = np.sin(4*np.pi*X_true)
w = []
train_models(X_train,t_train,w)
XX_train = []
XX_valid = []
regularize(X_train,X_valid,t_train,XX_train,XX_valid,w,0.00000001523) #regularizing with lambda = e^-18 as an optimal value
XX_train2 = []
XX_valid2 = []
regularize(X_train,X_valid,t_train,XX_train2,XX_valid2,w,1) #regularizing with lambda = 1 resulting in underfitting
train_err = []
get_error(X_train,t_train,XX_train,w,train_err)
valid_err = []
get_error(X_valid,t_valid,XX_valid,w,valid_err)
t_valid.shape = (100,1)
t_true.shape = (100,1)
avg_error = (1/100)*np.sum(np.power((np.power((t_valid - t_true),2)),0.5))
disp_plots(X_train,X_valid,X_true,t_train,t_valid,t_true,w,XX_valid[0],train_err,valid_err,avg_error)