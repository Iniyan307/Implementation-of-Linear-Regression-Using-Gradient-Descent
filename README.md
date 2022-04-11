# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implementation-of-Linear-Regression-Using-Gradient-Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implementation-of-Linear-Regression-Using-Gradient-Descent.
Developed by: Iniyan S
RegisterNumber:  212220040053
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("student_scores - student_scores.csv")
dataset.head() #no need just for understand
dataset.tail()  #no need just for understand
X=dataset.iloc[:,:-1].values #assigning column scores to X
y=dataset.iloc[:,1].values   #assigning column scores to y
print(X)
print(y)
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='black')
plt.title("h vs s(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("h vs s(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![output 1](.\output1.png)
![output 2](.\output2.PNG)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
