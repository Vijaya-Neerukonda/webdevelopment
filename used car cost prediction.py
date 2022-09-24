#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#2.Load Data
df=pd.read_csv("C:\car data.csv")
df


# In[3]:


#exploring the data
df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


#categorical columns
df.select_dtypes(include='object').columns


# In[7]:


#numerical columns
df.select_dtypes(include=["float64","int64"]).columns


# In[8]:


df.describe()


# In[9]:


#checking missing values
df.isnull().sum()


# In[10]:


#drop unwanted column
df=df.drop(columns='Car_Name')


# In[11]:


df.head()


# In[12]:


#add a column
df['current year']=2022


# In[13]:


df.head()


# In[14]:


df['years old']=df['current year']-df['Year']


# In[15]:


df.head()


# In[16]:


df=df.drop(columns=['current year','Year'])
df.head()


# In[17]:


#encoding the categorical variables
df.select_dtypes(include='object').columns


# In[18]:


sns.set_style("darkgrid")
sns.FacetGrid(df,hue="years old",height=6).map(plt.scatter,"Present_Price","Selling_Price").add_legend()
plt.show()


# In[19]:


sns.set_style("darkgrid")
sns.FacetGrid(df,hue="Present_Price",height=6).map(plt.scatter,"Kms_Driven","Selling_Price").add_legend()
plt.xlabel("kms driven")
plt.ylabel("selling price")
plt.show()


# In[20]:


sns.set_style("darkgrid")
sns.FacetGrid(df,height=6).map(sns.histplot,"Selling_Price")
plt.ylabel("no of cars sold")
plt.show()


# In[21]:


sns.set_style("darkgrid")
sns.FacetGrid(df,height=6).map(sns.histplot,"Kms_Driven")
plt.xlabel("distance travel")
plt.ylabel("demand")
plt.show()


# In[22]:


df=pd.get_dummies(data=df,drop_first=True)


# In[23]:


df.head()


# In[24]:


#Perform data visualization
df.corr()


# In[25]:


plt.figure(figsize=(5,5))
sns.pairplot(df)


# In[26]:


corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[27]:


df.shape


# In[28]:


#splitting the data
df.head()


# In[29]:


#matrix of features
x=df.drop(columns='Selling_Price')


# In[30]:


#target varible
y=df['Selling_Price']
df['years old'].min()
x


# In[31]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x,y)
importance=np.sort(model.feature_importances_)
plt.barh(x.columns,importance)
plt.show()


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_test


# In[33]:


y_test


# In[34]:


x_train.shape


# In[35]:


x_test.shape


# In[36]:


y_train.shape


# In[37]:


y_test.shape


# In[38]:


#building the model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(x_train,y_train)
rf_test=regressor.score(x_test,y_test)
rf_train=regressor.score(x_train,y_train)


# In[39]:


rf_train


# In[40]:


rf_test


# In[42]:


y_pred=regressor.predict(np.array([9.4,61381,0,7,1,0,0,1]).reshape(1,-1))
y_pred


# In[43]:


y_pred=regressor.predict(np.array([5.7,61381,0,7,1,0,0,1]).reshape(1,-1))
y_pred


# In[ ]:


# Computing MSE and RMSE
from sklearn.metrics import mean_squared_error
lin_mse1 = mean_squared_error(y_test,y_pred)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)


# In[ ]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# In[ ]:


result=pd.DataFrame({"actual":y_test,"predicted":y_pred})
result


# In[ ]:


plt.plot(result)


# In[ ]:




