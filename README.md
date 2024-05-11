# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### STEP-1: 
Start program.

### STEP-2:
Import the libraries and Load the dataset.

### STEP-3:
Define X and Y array and Define a function for costFunction,cost and gradient.

### STEP-4:
Define a function to plot the decision boundary.

### STEP-5:
Define a function to predict the Regression value. 

### STEP-6:
End program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHYAM S
RegisterNumber: 212223240156
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1])
plt.scatter(X[y==0][:,0],X[y==0][:,1])
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
## Array value of x:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/a40812c9-1537-4f8b-94ea-42cbbe7e78d3)


## Array value of y:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/deea4c3b-7fb8-4bf4-9077-1ea6f5876138)

## Score Graph:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/6ddaa8b5-1335-4598-82a5-6bda53746155)

## Sigmoid Function Graph:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/91e33261-cd3c-4ba3-b296-bcea4799a018)

## x train grad value:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/34243e32-85d3-46e1-b782-0226276e5f12)

## y train grad value:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/f666ba10-ee04-4b38-9965-54b88dad51ef)

## Print res.x:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/59d6fbb3-fc5e-4c1d-b5f5-c8ec265830c3)

## decision boundary graph:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/d0a3e1c3-0f8d-4552-b8e2-5d2727f4f57c)

## Probablity value:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/72ae2fc6-5989-4726-bba1-9383cb15deee)


## Prediction value of mean:
![image](https://github.com/SanthoshThiru/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/148958618/124494a8-ef30-4e58-8764-cd1f4a7813c3)



## Result:
Thus, the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

