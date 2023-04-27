#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6


# In[3]:


location=r"C:\Users\Admin1\Desktop\dllad\AirPassengers.csv"
df= pd.read_csv(location, encoding='gbk',parse_dates=['Month'],infer_datetime_format=True)
indf=df.set_index(['Month'])


# In[4]:


indf.describe()
indf.rename(columns={'#Passengers':'Passengers'},inplace=True)
indf.head()


# In[5]:


plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.plot(indf)


# In[6]:


indf['months'] = [x.month for x in indf.index]
indf['years'] = [x.year for x in indf.index]


# In[7]:


indf.reset_index(drop=True, inplace=True)


# In[8]:


X=indf.drop("Passengers",axis=1)
Y= indf["Passengers"]
X_train=X[:int (len(Y)*0.75)] 
X_test=X[int(len(Y)*0.75):]
Y_train=Y[:int (len(Y)*0.75)] 
Y_test=Y[int(len(Y)*0.75):]


# In[9]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)


# In[10]:


df1=df.set_index(['Month'])
df1.rename(columns={'#Passengers':'Passengers'},inplace=True)
train=df1.Passengers[:int (len(indf.Passengers)*0.75)]
test=df1.Passengers[int(len(indf.Passengers)*0.75):]
preds=rf.predict(X_test)
predictions=pd.DataFrame(preds,columns=['Passengers'])
predictions.index=test.index
plt.plot(train)
plt.plot(test, color='orange', label='actual')
plt.plot(predictions,color='green', label='Forecasts')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title("Forecast of AirPassengers")


# In[13]:


rf.score(X_train, Y_train)


# In[14]:


rf.score(X_test, Y_test)


# In[ ]:




