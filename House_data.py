#!/usr/bin/env python
# coding: utf-8

# 
# # Title - Predicting House Prices
# ### Author - Saif Ali Gulab Sayyed
# ### objective - to create a machine learning model to predict house prices using varous features

# In[1]:


#Importing essential libraries and uploading the dataset House.csv
import pandas as pd               
import matplotlib.pyplot as plt   
import numpy as np                
import seaborn as sns             

house_data = pd.read_csv("House.csv")


# In[2]:


#Checking the info DataFrame
house_data.info()


# In[3]:


#Checking the head DataFrame
house_data.head(10)


# In[4]:


#Checking the number of rows and columns
house_data.shape


# In[5]:


#As we can check the total_sqft column is categorical column. So we will clean and convert the datatype from this column

import re

def clean_total_sqft(total_sqft):
    # Function to clean 'total_sqft' values and convert them to square feet (float)
    
    # Remove any spaces and '-' symbols from the string
    total_sqft = total_sqft.replace(' ', '').replace('-', '')
    
    # Check if the value contains other units (e.g., "34.46Sq. Meter")
    if 'Sq. Meter' in total_sqft:
        # Extract the numeric value and convert to square feet
        sq_meter_value = re.findall(r'\d+\.\d+', total_sqft)
        if sq_meter_value:
            total_sqft = float(sq_meter_value[0]) * 10.7639  # 1 Sq. Meter = 10.7639 Sq. Feet (approx.)
        else:
            total_sqft = None  # Handle cases where no numeric value is found
            
    # Check if the value contains 'Sq. Yard'
    elif 'Sq. Yards' in total_sqft:
        # Extract the numeric value and convert to square feet
        sq_yard_value = re.findall(r'\d+\.\d+', total_sqft)
        if sq_yard_value:
            total_sqft = float(sq_yard_value[0]) * 9  # 1 Sq. Yard = 9 Sq. Feet (approx.)
        else:
            total_sqft = None  # Handle cases where no numeric value is found
    
    # Convert any remaining values to float
    else:
        try:
            total_sqft = float(total_sqft)
        except ValueError:
            total_sqft = None  # Invalid values can be replaced with None or handled accordingly
    
    return total_sqft

# Apply the function to the 'total_sqft' column
house_data['total_sqft'] = house_data['total_sqft'].apply(clean_total_sqft)


# In[6]:


#As we can see there is important information in size column i.e. number of bedrooms. 
#Since it is a categorical column we need to clean the column and extract valuable information.

import re

# Step 1: Extract numeric values from 'size' column
def extract_bedrooms(x):
    x = str(x).lower()  # Convert to lowercase for case-insensitive matching
    if 'bedroom' in x:
        match = re.search(r'\d+', x)
        return int(match.group()) if match else None
    elif 'bhk' in x:
        match = re.search(r'\d+', x)
        return int(match.group()) if match else None
    elif 'rk' in x:
        return 0  # Set the number of bedrooms to 0 for 'RK' type
    else:
        return None

house_data['num_bedrooms'] = house_data['size'].apply(extract_bedrooms)

# Step 2: Drop the original 'size' column
house_data.drop(columns=['size'], inplace=True)


# In[7]:


house_data.head(10)


# In[8]:


#Checking the datatypes
house_data.dtypes


# In[9]:


#checking if there are any duplicates 
duplicate = house_data.duplicated()
duplicate
sum(duplicate)


# In[10]:


#droping the duplicates
house_data = house_data.drop_duplicates()
sum(duplicate)


# In[12]:


#checking the missing values 
house_data.isna().sum()


# In[13]:


#visualizing the missing values on heatmap
missing_data = house_data.isnull()
plt.figure(figsize=(10, 6))
sns.heatmap(missing_data, cmap='viridis', cbar=False, yticklabels=False)

plt.title('Missing Value Heatmap')
plt.show()


# In[14]:


from sklearn.impute import SimpleImputer

# Drop rows with missing values 
house_data.dropna(subset=['location'], inplace=True)

house_data.dropna(subset=['total_sqft'], inplace=True)

house_data.dropna(subset=['bath'], inplace=True)

house_data.dropna(subset=['num_bedrooms'], inplace=True)


# In[15]:


# Impute missing values in specific columns
imputer = SimpleImputer(strategy='mean')
numeric_columns = ['balcony']
house_data.loc[:, numeric_columns] = imputer.fit_transform(house_data[numeric_columns])


# In[16]:


# droping society column as it has huge number of missing values, also it contains categorical values which is not that useful
house_data.drop(['society'], axis=1, inplace=True)
house_data.dtypes


# In[18]:


# Now we will convert 'float64' into 'int64' type. 
house_data.bath = house_data.bath.astype('int64')

house_data.balcony = house_data.balcony.astype('int64')

house_data.num_bedrooms = house_data.num_bedrooms.astype('int64')

house_data.dtypes


# In[19]:


house_data.head(10)


# In[20]:


house_data.shape


# In[21]:


#visualizing the boxplot of total_sqft column to check if there is any outliers or not
sns.boxplot(house_data.total_sqft)


# In[22]:


#visualizing the boxplot of bath column to check if there is any outliers or not
sns.boxplot(house_data.bath)


# In[23]:


#visualizing the boxplot of balcony column to check if there is any outliers or not
sns.boxplot(house_data.balcony)


# In[24]:


#visualizing the boxplot of price column to check if there is any outliers or not
sns.boxplot(house_data.price)


# In[25]:


#visualizing the boxplot of num_bedrooms column to check if there is any outliers or not
sns.boxplot(house_data.num_bedrooms)


# In[26]:


#treating the outliers 

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['total_sqft'])

house_data['total_sqft'] = winsor.fit_transform(house_data[['total_sqft']])

# lets see boxplot
sns.boxplot(house_data['total_sqft'])


# In[27]:


#treating the outliers

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['bath'])

house_data['bath'] = winsor.fit_transform(house_data[['bath']])

sns.boxplot(house_data['bath'])


# In[28]:


#treating the outliers

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['price'])
 
house_data['price'] = winsor.fit_transform(house_data[['price']])

sns.boxplot(house_data['price'])


# In[29]:


#treating the outliers

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['num_bedrooms'])

house_data['num_bedrooms'] = winsor.fit_transform(house_data[['num_bedrooms']])

sns.boxplot(house_data['num_bedrooms'])


# In[30]:


house_data.head(10)


# In[31]:


#checking the count of unique values in location column
unique_count = house_data['location'].nunique()
print(unique_count)


# In[32]:


#checking the count of unique values in availability column
unique_count = house_data['availability'].nunique()
print(unique_count)


# In[33]:


#checking the count of unique values in area_type column
unique_count = house_data['area_type'].nunique()
print(unique_count)


# In[34]:


location_freq = house_data['location'].value_counts().to_dict()
house_data['location_freq'] = house_data['location'].map(location_freq)


# In[35]:


mean_price_by_location = house_data.groupby('location')['price'].mean().to_dict()
house_data['location_mean_price'] = house_data['location'].map(mean_price_by_location)


# In[38]:


# drop availability and location column 
house_data.drop(['availability', 'location'], axis=1, inplace=True)
house_data.dtypes


# In[39]:


house_data = pd.get_dummies(house_data, columns=['area_type'])

# Print the encoded DataFrame
house_data.head()


# In[42]:


house_data.isna().sum()


# In[44]:


#Normalizing the columns with wide ranges
from sklearn.preprocessing import StandardScaler, MinMaxScaler

columns_to_normalize = ['total_sqft', 'price', 'location_freq', 'location_mean_price', ]

# Create a copy of the DataFrame for preprocessing
house_data_normalized = house_data.copy()


# In[45]:


#Normalizing the columns with wide ranges
min_max_scaler = MinMaxScaler()
house_data_normalized[columns_to_normalize] = min_max_scaler.fit_transform(house_data_normalized[columns_to_normalize])


# In[46]:


house_data_normalized.head()


# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into features (X) and target variable (y)
X = house_data_normalized.drop('price', axis=1)
y = house_data_normalized['price']


# In[50]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[52]:


# Fit the model on the training data
rf_model.fit(X_train, y_train)


# In[53]:


# Make predictions on the test data
y_pred = rf_model.predict(X_test)


# In[54]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[55]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[58]:


from sklearn.metrics import mean_absolute_error

predicted_values = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, predicted_values)
print("Mean Absolute Error:", mae)


# In[59]:


from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
print("Root Mean Squared Error:", rmse)


# In[61]:


target_stats = house_data_normalized['price'].describe()
print(target_stats)


# In[62]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




