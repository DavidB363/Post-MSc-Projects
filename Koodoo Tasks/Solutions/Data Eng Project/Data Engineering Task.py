#!/usr/bin/env python
# coding: utf-8

# # Data Engineering Task for Koodoo job application process.
# # Creating a wine database using SQLite3.
# # David Brookes January/February 2021

# In[1]:


import os
import sqlite3, csv
import pandas as pd
import time


# In[2]:


print(os.getcwd())


# In[3]:


os.chdir(r'D:\My Documents\Python Code\Koodoo Tasks\Solutions\Data Eng Project')
print(os.getcwd())


# In[4]:


# Create a new database Wines.db.
conn = sqlite3.connect('Wine.db')

# Allow for foreign keys to be used in the database.
conn.execute("PRAGMA foreign_keys = 1")

# Create a cursor object.
cur = conn.cursor()


# In[5]:


# Create the staging_wines table.
staging_wines = """
    CREATE TABLE IF NOT EXISTS staging_wines (
    Vintage TEXT,
    Country TEXT,
    County TEXT,
    Designation TEXT,
    Points INTEGER,
    Price REAL,
    Province TEXT,
    Title TEXT,
    Variety TEXT,
    Winery TEXT
    ); 
"""

# Execute the SQL command.
cur.execute(staging_wines)
print('staging_wines table has been created.')

# Commit the changes.
conn.commit()


# In[6]:


# Read Wines.csv file into a Pandas dataframe.
dfWines = pd.read_csv('Wines.csv')
#print(dfWines.head())

# Find the number of rows and columns using the shape property of the DataFrame.
# This can be used to check against the SQLite calculation.
df_rows, df_cols = dfWines.shape

print('Number of rows of the Wines DataFrame = ', df_rows)
print('Number of columns of the Wines DataFrame = ', df_cols)


# In[7]:


dfWines.dtypes


# In[8]:


# Show the null values in dfWines.

dfWines.isnull().sum()


# In[9]:


# Process the DataFrame.
# Convert the Price column from a string to a float.

# First, strip the $ sign and also commas from the price.
# Then convert the string to a float.

dfWines['Price'] = dfWines['Price'].replace(regex = '[$,]', value ='').astype(float)

print(dfWines.head())


# In[10]:


# Find the average price of a bottle of wine using Pandas. 
# (To check the results obtained using SQLite )
print('Average price of a bottle of wine is $', dfWines['Price'].mean())

# Find the maximum price of a bottle of wine using Pandas.
# (To check the results obtained using SQLite )
print('Maximum price of a bottle of wine is $', dfWines['Price'].max())


# # Task 1. 
# # Stage the data.
# # Load data into staging_wines table of the database Wines.db

# In[11]:


# Populate the staging_wines table in the Wines database using the Pandas DataFrame.
# Determine the time that it takes for this operation.

start_time = time.time()

num_records = 0

for row in range(df_rows):
    cur.execute("""INSERT INTO staging_wines VALUES (?,?,?,?,?,?,?,?,?,?);""",dfWines.iloc[row,:])
    num_records += 1
conn.commit()
print('\n{} Records transferred.'.format(num_records))

stop_time = time.time()
print('Time taken {} seconds.'.format(stop_time-start_time))


# In[12]:


# Run SQLite queries using Pandas.

# Calculate the total number of rows in the staging_wine table.
df_count_query = pd.read_sql_query("SELECT COUNT(*) FROM staging_wines;", conn)
print(df_count_query.head(), '\n')

# Calculate the average price of a bottle of wine. 
df_average_query = pd.read_sql_query("SELECT AVG(Price) FROM staging_wines;", conn)
print(df_average_query.head(), '\n')

# Calculate the price of the most expensive bottle of wine.
df_max_query = pd.read_sql_query("SELECT MAX(Price) FROM staging_wines;", conn)
print(df_max_query.head(), '\n')


# # Task 2
# # Load the staged data into the Wine Mart.
# # (i.e. Load into 4 tables; DimWinery, DimGeography, DimVariety and FactWine)

# In[13]:


# Create the 4 tables.

DimWinery = """
    CREATE TABLE IF NOT EXISTS DimWinery (
    Winery_Id INTEGER PRIMARY KEY,
    WineryName TEXT
    ); 
"""

DimGeography = """
    CREATE TABLE IF NOT EXISTS DimGeography (
    Geography_Id INTEGER PRIMARY KEY,
    Country TEXT,
    Province TEXT,
    County TEXT
    ); 
"""

DimVariety = """
    CREATE TABLE IF NOT EXISTS DimVariety (
    Variety_Id INTEGER PRIMARY KEY,
    Variety TEXT
    ); 
"""

FactWine = """
    CREATE TABLE IF NOT EXISTS FactWine (
    Wine_Id INTEGER PRIMARY KEY,
    Title TEXT,
    Winery_Id INTEGER,
    Geography_Id INTEGER,
    Variety_Id INTEGER,
    Points INTEGER,
    Price REAL,
    Vintage TEXT,
    FOREIGN KEY (Winery_Id) REFERENCES DimWinery (Winery_Id),
    FOREIGN KEY (Geography_Id) REFERENCES DimGeography (Geography_Id),
    FOREIGN KEY (Variety_Id) REFERENCES DimVariety (Variety_Id)
    ); 
"""

# Execute the SQL commands.
cur.execute(DimWinery)
print('DimWinery table has been created.')

cur.execute(DimGeography)
print('DimGeography table has been created.')

cur.execute(DimVariety)
print('DimVariety table has been created.')

cur.execute(FactWine)
print('FactWine table has been created.')


# Commit the changes.
conn.commit()


# In[14]:


# Generate primary keys for Winery, Variety and Geography.
Winery_PK_dict = dict()
Variety_PK_dict = dict()
Geography_PK_dict = dict()


# Primary keys start from 1.
Winery_PK_Id = 1
Variety_PK_Id = 1
Geography_PK_Id = 1

for row in range(df_rows):
    Winery_name = dfWines.loc[row, 'Winery']
    if Winery_name not in Winery_PK_dict:
        Winery_PK_dict[Winery_name] = Winery_PK_Id 
        Winery_PK_Id += 1
          
    Variety_name = dfWines.loc[row, 'Variety'] 
    if Variety_name not in Variety_PK_dict:
        Variety_PK_dict[Variety_name] = Variety_PK_Id 
        Variety_PK_Id += 1
        
    Geography_names = (dfWines.loc[row, 'Country'], dfWines.loc[row, 'Province'], dfWines.loc[row, 'County']) # Return a tuple.
    if Geography_names not in Geography_PK_dict:
        Geography_PK_dict[Geography_names] = Geography_PK_Id
        Geography_PK_Id += 1
        

#print(Winery_PK_dict)
len_Winery_PK_dict = len(Winery_PK_dict)
print(len_Winery_PK_dict)

#print(Variety_PK_dict)
len_Variety_PK_dict = len(Variety_PK_dict) 
print(len_Variety_PK_dict)

#print(Geography_PK_dict)
len_Geography_PK_dict = len(Geography_PK_dict)
print(len_Geography_PK_dict)
      


# In[15]:


# Populate the DimWinery, Dim Variety and DimGeography tables.

num_records = 0
for WineryName in Winery_PK_dict:
    cur.execute("""INSERT INTO DimWinery VALUES (?,?);""",  (Winery_PK_dict[WineryName], WineryName))
    num_records += 1
print('\n{} Records transferred.'.format(num_records))


num_records = 0
for VarietyName in Variety_PK_dict:
    cur.execute("""INSERT INTO DimVariety VALUES (?,?);""",  (Variety_PK_dict[VarietyName], VarietyName))
    num_records += 1
print('\n{} Records transferred.'.format(num_records))

num_records = 0
for GeographyNames in Geography_PK_dict:
    cur.execute("""INSERT INTO DimGeography VALUES (?,?,?,?);""",  (Geography_PK_dict[GeographyNames], GeographyNames[0], GeographyNames[1], GeographyNames[2]))
    num_records += 1
print('\n{} Records transferred.'.format(num_records))
  
conn.commit()


# In[16]:


# Populate the FactWine table.

num_records = 0

for row in range(df_rows):
    Title = dfWines.loc[row, 'Title']
    Winery_Id = Winery_PK_dict[dfWines.loc[row, 'Winery']]
    Geography_Id = Geography_PK_dict[(dfWines.loc[row, 'Country'], dfWines.loc[row, 'Province'], dfWines.loc[row, 'County'])]
    Variety_Id = Variety_PK_dict[dfWines.loc[row, 'Variety']]  
    Points = dfWines.loc[row, 'Points']
    Price = dfWines.loc[row, 'Price']
    Vintage = dfWines.loc[row, 'Vintage']
    cur.execute("""INSERT INTO FactWine VALUES (?,?,?,?,?,?,?,?);""", (row+1, Title, Winery_Id, Geography_Id, Variety_Id, Points, Price, Vintage)) 
    #conn.commit()
    num_records += 1

conn.commit()
print('\n{} Records transferred.'.format(num_records))


# In[18]:


# Run SQLite queries using Pandas.

# Calculate the total number of rows in the FactWine table.
df_count_query = pd.read_sql_query("SELECT COUNT(*) FROM FactWine;", conn)
print(df_count_query.head(), '\n')


# In[19]:


# Close the cursor and database connection.
cur.close()
conn.close()

