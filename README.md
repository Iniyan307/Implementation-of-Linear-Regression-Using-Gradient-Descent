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
data=pd.read_csv("/content/student_scores.csv")
data.head()
data.isnull().sum() #returns the number of missing values
x=data.Hours
x.head()
y=data.Scores
y.head()
n=len(x)
m=0
c=0
L=0.01
loss=[]
for i in range(10000):
  ypred=m*x+c
  MSE=(1/n)*sum((ypred-y)*2)
  dm=(2/n)*sum(x*(ypred-y))
  dc=(2/n)*sum(ypred-y)
  c=c-L*dc
  m=m-L*dm
  loss.append(MSE)
print(m,c)
y_pred=m*x+c
plt.scatter(x,y,color="red")
plt.plot(x,y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Scores")
plt.plot(loss)
plt.xlabel("iterations")
plt.ylabel("loss")

```

## Output:
![output 1](.\output1.png)
![output 2](.\output2.PNG)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
