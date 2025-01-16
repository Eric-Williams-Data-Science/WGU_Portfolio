#!/usr/bin/env python
# coding: utf-8

# In[1]:


12478
import pandas as pd
file_path = r"On_Time_Reporting.csv"

df = pd.read_csv(file_path)


# In[2]:


df


# In[5]:


# Specify the columns to keep
columns_to_keep = [
    'Year', 'Month', 'DayofMonth', 'DayOfWeek',
    'OriginAirportID', 'DestAirportID', 'DestAirportSeqID',
    'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes',
    'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes'
]

# Filter the DataFrame to keep only the specified columns
df_filtered = df[columns_to_keep]
df_filtered


# In[7]:


# Select the relevant columns and create a copy
df_filtered = df_filtered[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                  'OriginAirportID', 'DestAirportID', 
                  'DestAirportSeqID', 'CRSDepTime', 
                  'DepTime', 'DepDelay', 'DepDelayMinutes', 
                  'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes']].copy()

# Rename the columns
df_filtered.rename(columns={
    'Year': 'YEAR',
    'Month': 'MONTH',
    'DayofMonth': 'DAY',
    'DayOfWeek': 'DAY_OF_WEEK',
    'OriginAirportID': 'ORG_AIRPORT',
    'DestAirportID': 'DEST_AIRPORT',
    'DestAirportSeqID': 'DEST_AIRPORT_SEQ_ID',
    'CRSDepTime': 'SCHEDULED_DEPARTURE',
    'DepTime': 'DEPARTURE_TIME',
    'DepDelay': 'DEPARTURE_DELAY',
    'CRSArrTime': 'SCHEDULED_ARRIVAL',
    'ArrTime': 'ARRIVAL_TIME',
    'ArrDelay': 'ARRIVAL_DELAY'
}, inplace=True)


# In[9]:


df_filtered


# In[11]:


#Filtering rows where DEST_AIRPORT_ID is 12478
df_filtered = df_filtered[df_filtered['DEST_AIRPORT'] == 12478].copy()

df_filtered


# In[13]:


# Check for missing values by column
missing_values = df_filtered.isnull().sum()

# Display the number of missing values for each column
print(missing_values)


# In[15]:


# Drop rows with missing values for the specified columns
df_filtered.dropna(subset=['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'], inplace=True)


# In[17]:


# Check for missing values by column
missing_values = df_filtered.isnull().sum()

# Display the number of missing values for each column
print(missing_values)


# In[19]:


df_filtered


# In[21]:


#Dropping spaces before and after each cell
df_filtered = df_filtered.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df_filtered


# In[23]:


#Saving filtered dataset
df_filtered.to_excel(r'C:\Users\18014\Desktop\filtered_dataset.xlsx', index=False)


# In[ ]:




