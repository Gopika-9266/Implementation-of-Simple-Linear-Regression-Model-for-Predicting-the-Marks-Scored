# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gopika R
RegisterNumber: 212222240031

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="yellow")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

### df.head():
![exp2-1](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/b279ce53-6b28-4f40-922e-71d37a4bfc03)

### Values of x:
![exp2-2](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/285b5e4b-2020-4861-955d-18205c8a7f49)

### Values of y:
![exp2-3](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/8fbb6e61-4959-476b-bba9-4f2c616e8800)

### Values of y_train:
![exp2-4](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/29280a4b-424d-4e1b-90db-d4dc89f679c0)

### Values of y_pred:
![exp2-5](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/245fba19-5b24-4ad5-b6cb-dd6f4d0a2773)

### Training Set Graph:
![exp2-6](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/496a6e6e-781c-4f9f-940f-5c63f52ca88c)

### Test set Graph:
![Screenshot 2024-05-09 220140](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/33e7a000-93e6-4ed6-a974-95bb1255e74e)

### Values of MSE, MAE, RMSE:
![exp2-7](https://github.com/Gopika-9266/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122762773/79539abe-3b77-4e83-b7b8-320cf1cba482)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
