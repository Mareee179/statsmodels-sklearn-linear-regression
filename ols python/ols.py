import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.model_selection import train_test_split
import os;
path="/home/michaeljgrogan/Documents/a_documents/computing/data science/datasets"
os.chdir(path)
os.getcwd()

variables = pd.read_csv('gasoline.csv')
consumption = variables['consumption']
capacity = variables['capacity']
price = variables['price']
hours = variables['hours']

y = consumption
x = np.column_stack((capacity,price,hours))
x = sm.add_constant(x, prepend=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

results = smf.OLS(y_train,x_train).fit()
print(results.summary())

import statsmodels

name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
bp = statsmodels.stats.diagnostic.het_breushpagan(results.resid, results.model.exog)
bp
pd.DataFrame(name,bp)

z = np.column_stack((capacity,price))
z = sm.add_constant(z, prepend=True)

from sklearn.metrics import r2_score
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(z,hours)
predictions = lm.predict(z)
print(predictions)

rsquared=r2_score(hours, predictions)
rsquared

vif=1/(1-(rsquared))

y_pred = results.predict(x_test)
print(y_pred)

mse=(y_pred-y_test)/y_test
mse
np.mean(mse)
