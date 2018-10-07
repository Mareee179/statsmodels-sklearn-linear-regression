import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os;
path="/home/michaeljgrogan/Documents/a_documents/computing/data science/datasets"
os.chdir(path)
os.getcwd()

variables=pd.read_csv('dividendinfo.csv')
dividend=variables['dividend']
freecashflow_pershare=variables['fcfps']
earnings_growth=variables['earnings_growth']
debt_to_equity=variables['de']
mcap=variables['mcap']
current_ratio=variables['current_ratio']

# debt_to_equity.shape

y=dividend
x=np.column_stack((freecashflow_pershare,earnings_growth,debt_to_equity,mcap,current_ratio))
x=sm.add_constant(x,prepend=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
logreg=LogisticRegression().fit(x_train,y_train)
logreg
print("Training set score: {:.3f}".format(logreg.score(x_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test,y_test)))

import statsmodels.api as sm
logit_model=sm.Logit(y_train,x_train)
result=logit_model.fit()
print(result.summary())

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
falsepos,truepos,thresholds=roc_curve(y_test,logreg.decision_function(x_test))

plt.plot(falsepos,truepos,label="ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

cutoff=np.argmin(np.abs(thresholds))
plt.plot(falsepos[cutoff],truepos[cutoff],'o',markersize=10,label="cutoff",fillstyle="none")
plt.show()

from sklearn import metrics
metrics.auc(falsepos, truepos)
