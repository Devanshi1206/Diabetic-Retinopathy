#!/usr/bin/env python
# coding: utf-8

# # Diabetic Retinopathy

# In[1]:


from platform import python_version
print(python_version())


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## IMPORTING DATA

# In[3]:


retinopathy_df=pd.read_csv('Retinopathy.csv')


# In[4]:


retinopathy_df


# ## EXPLORING DATA

# In[5]:


retinopathy_df.info


# In[6]:


retinopathy_df.describe


# ## Missing Values

# In[7]:


retinopathy_df.isna().sum()


# ## Duplicates

# In[8]:


retinopathy_df.duplicated().sum()


# ## Outliers 

# In[9]:


plt.subplot(2,2,1) 
retinopathy_df['image_size'].plot(kind='box') 

plt.subplot(2,2,2) 
retinopathy_df['height'].plot(kind='box') 

plt.subplot(2,2,3) 
retinopathy_df['width'].plot(kind='box') 

plt.subplot(2,2,4) 
retinopathy_df['channel'].plot(kind='box') 

plt.tight_layout()


# In[10]:


def find_outlier_limits(col_name):
    Q1,Q3=retinopathy_df[col_name].quantile([.25,.75])
    IQR=Q3-Q1
    low=Q1-(2* IQR)
    high=Q3+(2* IQR)
    return (high,low)

high_height,low_height=find_outlier_limits('height')
print('Height: ','upper limit: ',high_height,' lower limit: ',low_height)
high_width,low_width=find_outlier_limits('width')
print('Widht: ','upper limit: ',high_width,' lower limit:',low_width)


# In[11]:


retinopathy_df.loc[retinopathy_df['height']>high_height,'height']=high_height
retinopathy_df.loc[retinopathy_df['height']<low_height,'height']=low_height
retinopathy_df.loc[retinopathy_df['width']>high_width,'width']=high_width
retinopathy_df.loc[retinopathy_df['width']<low_width,'width']=low_width


# In[12]:


plt.subplot(2,2,1) 
retinopathy_df['image_size'].plot(kind='box') 

plt.subplot(2,2,2) 
retinopathy_df['height'].plot(kind='box') 

plt.subplot(2,2,3) 
retinopathy_df['width'].plot(kind='box') 

plt.subplot(2,2,4) 
retinopathy_df['channel'].plot(kind='box') 

plt.tight_layout()


# In[13]:


retinopathy_df['image_no'] = retinopathy_df['image'].apply(lambda x:x.split('_')[0])
retinopathy_df['image_side'] = retinopathy_df['image'].apply(lambda x:x.split('_')[1])

retinopathy_df.head()


# In[14]:


image = retinopathy_df['image']
del retinopathy_df['image']
image_shape = retinopathy_df['image_shape']
del retinopathy_df['image_shape']

retinopathy_df.head()


# In[15]:


correlation_df=retinopathy_df[['image_size','height','width','image_no','image_side']].corr()
correlation_df


# In[16]:


import seaborn as sns
sns.heatmap(correlation_df, cmap="coolwarm", annot=True)


# ## Modeling 

# In[17]:


X = retinopathy_df.iloc[:, 1:8]
y = retinopathy_df.iloc[:, 0]
print(X)
print(y)


# In[18]:


X=pd.get_dummies(retinopathy_df, columns=['image_side'])
X


# ## Train and Test data split

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# In[20]:


#Applying standard scaler on the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit_transform(X_train)
scale.transform(X_test)


# ## Training

# In[21]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# In[22]:


reg.fit(X_train, y_train)


# ## Testing

# In[23]:


y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)


# ## Evaluation

# In[24]:


from sklearn.metrics import r2_score
r2_S = r2_score(y_train, y_pred_train)
r2_S


# In[25]:


from sklearn.metrics import r2_score
r2_S = r2_score(y_test, y_pred_test)
r2_S


# Hence prediction is very like to be accurate.
