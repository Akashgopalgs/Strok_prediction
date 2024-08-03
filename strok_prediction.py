#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier


# In[85]:


stroke_data = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\Strok prediction.zip")
stroke_data


# In[86]:


# Drop column = 'id'
stroke_data.drop(columns='id', inplace=True)


# In[87]:


stroke_data.info()


# In[88]:


stroke_data.describe()


# In[89]:


stroke_data.isna().sum()


# In[90]:


print((stroke_data.isna().sum()/len(stroke_data))*100)


# In[91]:


stroke_data.dropna(how='any', inplace=True)


# In[80]:


# Categorical columns to visualize
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Set up subplots
fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 5 * len(categorical_columns)))

# Plot count plots for each categorical column
for col, ax in zip(categorical_columns, axes):
    sns.countplot(x=col, hue='stroke', data=stroke_data, ax=ax)
    ax.set_title(f'Countplot of {col} vs Stroke')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()


# In[62]:


# categorical_columns=X.select_dtypes(['object'])
# nemerical_columns=X.select_dtypes(['int','float'])


# In[63]:


# sns.heatmap(nemerical_columns.corr())


# In[81]:


# plt.subplot(1,2,1)
# plt.subplot(1,2,1)
# stroke_data[stroke_data['stroke']==0].hypertension.value_counts().plot.pie(autopct='%1.1f%%',colors=['#feaa25','#ccccc5'])
# plt.title('stroke = 0')
# plt.ylabel('')
# plt.subplot(1,2,2)
# stroke_data[stroke_data['stroke']==1].hypertension.value_counts().plot.pie(autopct='%1.1f%%',colors=['#feaa25','#ccccc5'])
# plt.title('stroke = 1')
# plt.ylabel('')


# In[45]:


# Numerical columns to visualize
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

# Set up subplots
fig, axes = plt.subplots(nrows=len(numerical_columns), ncols=1, figsize=(10, 5 * len(numerical_columns)))

# Plot histograms for each numerical column
for col, ax in zip(numerical_columns, axes):
    sns.histplot(x=col, hue='stroke', data=stroke_data, kde=True, ax=ax, bins=30)
    ax.set_title(f'Histogram of {col} vs Stroke')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()


# In[65]:


stroke_data['stroke'].value_counts()


# In[67]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

sns.countplot(x='hypertension', hue='stroke', data=stroke_data, ax=axes[0])
axes[0].set_title('Countplot of Hypertension vs Stroke')
axes[0].set_xlabel('Hypertension')
axes[0].set_ylabel('Count')

sns.countplot(x='heart_disease', hue='stroke', data=stroke_data, ax=axes[1])
axes[1].set_title('Countplot of Heart Disease vs Stroke')
axes[1].set_xlabel('Heart Disease')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()


# ### Data preprocessing

# In[92]:


# One-hot encode categorical columns
stroke_data = pd.get_dummies(stroke_data, columns=categorical_columns)


# In[93]:


stroke_data.head()


# In[94]:


# Assuming stroke_data is your DataFrame
X = stroke_data.drop('stroke', axis=1)
y = stroke_data['stroke']

# Specify train_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[95]:


# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Model

# In[132]:


from sklearn.tree import DecisionTreeClassifier


# In[180]:


models=[DecisionTreeClassifier(),KNeighborsClassifier()]


# In[194]:


def check_models():
    for model in models:
        model.fit(X_train,y_train)
        predion = model.predict(X_test)
        accuracy= accuracy_score(y_test,predion)
        print("Accuracy % of",model,":" ,round(accuracy,2)*100,"%")
        print(classification_report(y_test, predion))
        
        print("---- ------------------------------------------")
check_models()
    


# In[195]:


# ### Accuracy of the Support vector machine model is 95.0%

# In[ ]:

import pickle

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(models[0], f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(models[1], f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
