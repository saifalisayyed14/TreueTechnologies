#!/usr/bin/env python
# coding: utf-8

# # Title : Customer Churn Prediction 
# ### Author : Saif Ali Gulab Sayyed
# ### Description : This script trains a machine learning model to predict customer churn.

# In[1]:


# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# ### Data Preprocessing

# In[2]:


#Load the DataSet
Churn_data = pd.read_csv("Churn_Modelling.csv")


# In[3]:


#Checking the columns 
Churn_data.columns


# In[4]:


#Checking the datatypes
Churn_data.dtypes


# In[5]:


#Checking the data head 
Churn_data.head(10)


# In[6]:


#Changing the data types of Balance and EstimatedSalary Columns
Churn_data.Balance = Churn_data.Balance.astype('int64')

Churn_data.EstimatedSalary = Churn_data.EstimatedSalary.astype('int64')
Churn_data.dtypes


# In[7]:


#Checking for the duplicates in dataset
duplicate = Churn_data.duplicated()
duplicate
sum(duplicate)


# In[8]:


Churn_data.isna().sum()


# In[9]:


#Visualizing the boxplot of CreditScore column
sb.boxplot(Churn_data.CreditScore)


# In[10]:


#Visualizing the boxplot of Age column
sb.boxplot(Churn_data.Age)


# In[11]:


#Visualizing the boxplot of Tenure column
sb.boxplot(Churn_data.Tenure)


# In[12]:


#Visualizing the boxplot of Balance column
sb.boxplot(Churn_data.Balance)


# In[13]:


#Visualizing the boxplot of NumOfProducts column
sb.boxplot(Churn_data.NumOfProducts)


# In[14]:


#Visualizing the boxplot of EstimatedSalary column
sb.boxplot(Churn_data.EstimatedSalary)


# In[15]:


#Treating the outliers in CreditScore Column using winsorizer method
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['CreditScore'])

Churn_data['CreditScore'] = winsor.fit_transform(Churn_data[['CreditScore']])

#Visualizing the CreditScore column  
sb.boxplot(Churn_data['CreditScore'])


# In[16]:


#Treating the outliers of Age column using Replace method 

from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Age'])

Churn_data['Age'] = winsor.fit_transform(Churn_data[['Age']])

#Visualizing the Age column again 
sb.boxplot(Churn_data['Age'])


# In[17]:


#Treating the outliers of NumOfProducts column using Replace method 

from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['NumOfProducts'])

Churn_data['NumOfProducts'] = winsor.fit_transform(Churn_data[['NumOfProducts']])

#Visualizing the NumOfProducts column again 
sb.boxplot(Churn_data['NumOfProducts'])


# In[18]:


# Drop the columns which are irrelevant and contains categorical information
Churn_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

Churn_data.head()


# In[19]:


#Changing the data types of Balance and EstimatedSalary Columns
Churn_data.CreditScore = Churn_data.CreditScore.astype('int64')

Churn_data.Age = Churn_data.Age.astype('int64')

Churn_data.NumOfProducts = Churn_data.NumOfProducts.astype('int64')

Churn_data.dtypes


# In[20]:


#Visualizing Scatter plot: CreditScore vs. Age
sb.pairplot(Churn_data, hue='Exited')


# In[21]:


# Creating dummy variables of Gender Column
Gender=pd.get_dummies(Churn_data['Gender'], drop_first=True )
Gender.head(5)


# In[22]:


# Creating dummy variables of Tenure Column
Tenure_dummies = pd.get_dummies(Churn_data['Tenure'], prefix='Tenure', columns=['Tenure'])
Tenure_dummies.head(5)


# In[23]:


# Creating dummy variables of NumOfProducts Column
num_of_products_dummies = pd.get_dummies(Churn_data['NumOfProducts'], prefix='NumOfProducts')
num_of_products_dummies.head()


# In[24]:


# Get dummies for 'Geography' column
Geography_dummies = pd.get_dummies(Churn_data['Geography'], prefix='Geography')
Geography_dummies.head()


# In[25]:


# Concatenate the dummies columns with the original DataFrame
Churn_data = pd.concat([Churn_data, Gender, Tenure_dummies, num_of_products_dummies, Geography_dummies], axis=1)
Churn_data.head(5)


# In[26]:


Churn_data.columns


# In[27]:


# Drop the columns
Churn_data.drop(['Geography', 'Gender', 'Tenure', 'NumOfProducts'], axis=1, inplace=True)

# Show the resulting DataFrame
Churn_data.head(5)


# In[29]:


#Normalizing the columns with wide ranges
from sklearn.preprocessing import StandardScaler, MinMaxScaler

columns_to_normalize = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

# Create a copy of the DataFrame for preprocessing
Churn_data_normalized = Churn_data.copy()


# In[30]:


#Normalizing the columns with wide ranges
min_max_scaler = MinMaxScaler()
Churn_data_normalized[columns_to_normalize] = min_max_scaler.fit_transform(Churn_data_normalized[columns_to_normalize])


# In[31]:


Churn_data_normalized.head()


# ### Data Splitting

# In[32]:


#Split the data into variable (X) and target (y)

target = Churn_data_normalized["Exited"]

variable = Churn_data_normalized.drop(["Exited"],axis =1)

variable.head()


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


#Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(variable, target, test_size = 0.30, random_state = 42)


# In[35]:


#create a K-Nearest Neighbors (KNN) classifier object

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()


# In[36]:


#to create hyperparameter for the classifier object

parameters = {"n_neighbors": np.arange(1, 100)}


# In[37]:


# to perform hyperparameter tuning for the KNN

knnclassifier = GridSearchCV(knn, parameters, cv = 10)


# ### Model Training

# In[38]:


#train the model on the training data

knnclassifier.fit(X_train, y_train)


# In[39]:


print(knnclassifier.best_score_)


# In[40]:


print(knnclassifier.best_params_)


# In[47]:


print(knnclassifier.cv_results_)


# In[49]:


results = pd.DataFrame(knnclassifier.cv_results_)
results.head()


# In[50]:


import matplotlib.pyplot as plt


# In[51]:


# Plotting Mean Test Scores for Different K Values
plt.plot(results["param_n_neighbors"], results["mean_test_score"])
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Test Score')
plt.title('Mean Test Score vs Number of Neighbors')
plt.show()


# In[52]:


# Fitting the Best Model on Test Data
best_knnclassifier = KNeighborsClassifier(n_neighbors=knnclassifier.best_params_['n_neighbors'])
best_knnclassifier.fit(X_train, y_train)


# ### Model Evaluation 

# In[53]:


# Generating Predictions on Test Data
y_pred = best_knnclassifier.predict(X_test)


# In[55]:


# Evaluating the Model
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




