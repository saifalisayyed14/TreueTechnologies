#!/usr/bin/env python
# coding: utf-8

# # Tittle : Segment Customers
# 
# ### Author : Saif Ali Gulab Sayyed 
# 
# ### Discription : To analyze customer data from an online retail platform to segment customers on basis of their purchasing behavior

# In[1]:


# Import necessary libraries

import pandas as pd               
import matplotlib.pyplot as plt   
import numpy as np                
import seaborn as sns             


# In[2]:


#Load the DataSet
Retail_data = pd.read_excel("Online_Retail.xlsx")                            


# In[3]:


#Getting head of data
Retail_data.head(10)


# In[4]:


#getting the information about data and their datatypes
Retail_data.info()                


# In[5]:


Retail_data.isna().sum()


# In[6]:


missing_data = Retail_data.isnull()
plt.figure(figsize=(10, 6))
sns.heatmap(missing_data, cmap='viridis', cbar=False, yticklabels=False)

plt.title('Missing Value Heatmap')
plt.show()


# In[7]:


# Drop the column 'StockCode' 
Retail_data.drop(['StockCode'], axis=1, inplace=True)

Retail_data.head()


# In[8]:


# Impute missing values in specific columns
Retail_data['CustomerID'] = Retail_data['CustomerID'].ffill()


# In[9]:


# Drop rows with missing values 
Retail_data.dropna(subset=['Description'], inplace=True)


# In[10]:


Retail_data.isna().sum()


# In[11]:


### Identify duplicates records in the data ###
duplicate = Retail_data.duplicated()
duplicate
sum(duplicate)


# In[12]:


# Removing Duplicates and Identify duplicates records in the data #
Retail_data = Retail_data.drop_duplicates()
duplicate = Retail_data.duplicated()
duplicate
sum(duplicate)


# In[13]:


Retail_data.head(10)


# In[14]:


Retail_data.CustomerID = Retail_data.CustomerID.astype('int64')

Retail_data.dtypes


# In[15]:


# Set up the subplots
plt.figure(figsize=(15, 5))

# Plot the histogram for 'Quantity'
plt.subplot(1, 2, 1) 
plt.hist(Retail_data['Quantity'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity')

# Plot the histogram for 'UnitPrice'
plt.subplot(1, 2, 2) 
plt.hist(Retail_data['UnitPrice'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('UnitPrice')
plt.ylabel('Frequency')
plt.title('Histogram of UnitPrice')


# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


# In[16]:


############## Outlier Treatment ###############
# let's find outliers in Salaries
sns.boxplot(Retail_data.Quantity)


# In[17]:


sns.boxplot(Retail_data.UnitPrice)


# In[18]:


#Using winsorization method
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',  tail='both', fold=1.5, variables=['Quantity'])

Retail_data['Quantity'] = winsor.fit_transform(Retail_data[['Quantity']])

# lets see boxplot
sns.boxplot(Retail_data['Quantity'])


# In[19]:


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['UnitPrice'])

Retail_data['UnitPrice'] = winsor.fit_transform(Retail_data[['UnitPrice']])

# lets see boxplot
sns.boxplot(Retail_data['UnitPrice'])


# In[20]:


# Monthly Sales Trend
Retail_data['Month'] = Retail_data['InvoiceDate'].dt.month
monthly_sales = Retail_data.groupby('Month')['Quantity'].sum()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values)
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.title('Monthly Sales Trend')
plt.xticks(range(1, 13))
plt.show()


# In[21]:


#Country-wise Sales

plt.figure(figsize=(18, 6))
country_sales = Retail_data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
sns.barplot(x=country_sales.index, y=country_sales.values)
plt.xlabel('Country')
plt.ylabel('Total Quantity Sold')
plt.title('Country-wise Sales')
plt.xticks(rotation=90)
plt.show()


# In[22]:


# Group the data by customer country and product description
grouped_data = Retail_data.groupby(["Country", "Description"])["Quantity"].sum()

# Get the most popular products for each country
most_popular_products = grouped_data.groupby("Country").idxmax()

# Create a new column for the most popular product in each country
Retail_data["MostPopularProduct"] = Retail_data["Country"].apply(lambda country: most_popular_products[country])

# Print the most popular products by country
for country, product in most_popular_products.items():
    print(f"The most popular product in {country} is {product}")


# In[23]:


print(Retail_data.head())


# In[24]:


# Group the data by invoice date and product description
grouped_data = Retail_data.groupby(["InvoiceDate", "Description"])["UnitPrice"].max()

# Get the most expensive products for each invoice date
most_expensive_products = grouped_data.groupby("InvoiceDate").idxmax()

# Convert most_expensive_products index into a dictionary
most_expensive_dict = most_expensive_products.to_dict()

# Create a new column for the most expensive product for each invoice date
Retail_data["most_expensive_products"] = Retail_data["InvoiceDate"].apply(lambda date: most_expensive_dict.get(date))


# In[25]:


#checking the count of unique values in area_type column
unique_count = Retail_data['Country'].nunique()
print(unique_count)


# In[26]:


Retail_data.head()


# In[27]:


#checking the count of unique values in MostPopularProduct column
unique_count = Retail_data['MostPopularProduct'].nunique()
print(unique_count)


# In[28]:


#checking the count of unique values in most_expensive_products column
unique_count = Retail_data['most_expensive_products'].nunique()
print(unique_count)


# In[29]:


Retail_data.head()


# In[32]:


Retail_data.drop([ 'Description', 'Country', 'Month', 'most_expensive_products'], axis=1, inplace=True)
Retail_data.dtypes


# In[33]:


Retail_data.head(10)


# In[30]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the scaler on your numerical columns
Retail_data[['Quantity', 'UnitPrice']] = scaler.fit_transform(Retail_data[['Quantity', 'UnitPrice']])


# In[31]:


Retail_data.head(10)


# In[36]:


# Convert 'FamousProduct' to binary columns using one-hot encoding
MostPopularProduct_encoded = pd.get_dummies(Retail_data['MostPopularProduct'], prefix='MostPopularProduct')


# In[37]:


# Combine encoded columns with other features
cluster_features = pd.concat([Retail_data[['Quantity', 'UnitPrice']], MostPopularProduct_encoded], axis=1)


# In[32]:


# Select relevant features for clustering
cluster_features = Retail_data[['Quantity', 'UnitPrice', 'MostPopularProduct']]


# In[38]:


# Perform clustering
from sklearn.cluster import KMeans

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
Retail_data['Cluster'] = kmeans.fit_predict(cluster_features)


# In[39]:


# Analyze cluster profiles
cluster_profile = Retail_data.groupby('Cluster').mean()


# In[42]:


# Visualize clusters using scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', data=Retail_data, hue='Cluster', palette='Set1')
plt.title('Cluster Visualization')
plt.xlabel('Quantitye')
plt.ylabel('UnitPrice')
plt.show()


# In[ ]:




