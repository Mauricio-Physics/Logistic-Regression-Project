#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import seaborn as sbn

from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


data = pd.read_excel('C:/Users/MAURICIO/Desktop/insurance_data.xlsx',header=0)


# In[52]:


data.head()


# In[53]:


sbn.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="Greens")


# In[54]:


plt.scatter(data.age,data.bought_insurance,marker='.',color='black')


# In[55]:


sbn.set_style('whitegrid')
sbn.countplot(x='bought_insurance',data=data,palette='RdBu_r')


# In[56]:


x = data['age'] 
y = data['bought_insurance']     
print(len(x))


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)


# In[58]:


logistic_regression = LogisticRegression()


# In[59]:


print(x_train.shape)
print(y_train.shape)

print(type(x_train))
print(type(y_train))


# In[60]:


x_train = x_train.to_frame()
y_train = y_train.to_frame()

x_test = x_test.to_frame()
y_test = y_test.to_frame()


# In[61]:


logistic_regression.fit(x_train,y_train)


# In[62]:


logistic_regression.score(x_train,y_train)


# In[63]:


y_predicted = logistic_regression.predict(x_test)


# In[64]:


y_predicted


# In[65]:


logistic_regression.score(x_test,y_test)


# In[66]:


confusion_matrix(y_test,y_predicted)


# In[68]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predicted))
print("Precision:",metrics.precision_score(y_test, y_predicted))
print("Recall:",metrics.recall_score(y_test, y_predicted))

