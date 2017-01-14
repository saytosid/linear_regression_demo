import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import random
#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[[0]]
y_values = dataframe[[1]]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#predict a point 
point = dataframe.iloc[[random.randint(0,dataframe.shape[0])]]
y_predicted = body_reg.predict(point[['x']])
print "PREDICTION"
print "X =",point[['x']].as_matrix(),"Predicted Y =",y_predicted,"Error =",(point[['y']]-y_predicted).as_matrix()
#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
