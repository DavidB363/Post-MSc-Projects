#!/usr/bin/env python
# coding: utf-8

# # Using the Pandas software library. 
# # David Brookes November 2020.

# In[ ]:


# Pandas can be used for:
# 1. Data Analysis.
# 2. Data Manipulation.
# 3. Data Visualisation.
#
# Pandas makes it easy to work with data.


# # Reading a tabular data file into Pandas.

# In[1]:


import pandas as pd


# In[3]:


# Read a table of data from a URL address (for example).
# Note: the data file is in the correct format to be read by read_table().
# i.e. the data is separated by tabs.
        
orders = pd.read_table('http://bit.ly/chiporders') # The default separator is a tab.


# In[9]:


orders.head() # head() selects the first 5 rows.


# In[12]:


# In the next example the table data is not in the correct format for read_table().

users = pd.read_table('http://bit.ly/movieusers')
users.head()


# In[13]:


# The separtator is the | character.
# Looking in the Pandas documentation (Google: pandas.read_table) gives the solution.

users = pd.read_table('http://bit.ly/movieusers', sep = '|')
users.head()


# In[14]:


#... it's still not quite right!
# The first row is not a header.

users = pd.read_table('http://bit.ly/movieusers', sep = '|', header = None)
users.head()


# In[15]:


# Column names can also be added to the table.

user_cols = ['user_id','age','gender','occupation','zip_code']
users = pd.read_table('http://bit.ly/movieusers', sep = '|', header = None, names = user_cols)
users.head()


# # Selecting a Pandas Series from a DataFrame.

# In[ ]:


# Note: Series and DataFrame are examples of types of data.


# In[18]:


import pandas as pd
# UFO data file.
ufo = pd.read_table('http://bit.ly/uforeports', sep = ',') # Comma separated file format.
ufo.head()


# In[22]:


# ...or can use read_csv() instead. The default separator is a comma.

ufo = pd.read_csv('http://bit.ly/uforeports')
print(ufo.head())
print(type(ufo))


# In[28]:


# Selecting a column.

City_col = ufo['City']
print(City_col.head())
print(type(City_col))


# In[29]:


# Shortcut.
City_col = ufo.City # Column names are stored as attributes of a data frame.
print(City_col.head())
print(type(City_col))

# ... but be careful!!
#
# Colors_Reported_col = ufo.Colors Reported ,does not work! It doesn't like the space.
# Need to write:
# Colors_Reported_col = ufo['Colors Reported']
#
# So to be safe just use the bracket notation rather than the dot notation!


# In[30]:


# Adding a Series to a DataFrame.
# Silly example - first generate a new Series from two existing Series for convenience.

ufo['Location'] = ufo['City'] + ufo['State'] # + is defined for Series.
print(ufo.head())


# # Why do some Pandas commands end with parenthesis, and some don't?

# In[2]:


import pandas as pd


# In[4]:


movies = pd.read_csv('http://bit.ly/imdbratings')


# In[6]:


movies.head() # head() is a method (function) of movies.


# In[7]:


movies.describe()


# In[8]:


movies.shape # shape is an attribute of movies.


# In[11]:


movies.dtypes

# Note: Below 'object' basically means a string.


# In[12]:


type(movies)


# In[ ]:


# Note: To find a list of the attributes and methods of movies, then type -
#
#       movies. (and then press the Tab key).
movies.


# In[13]:


movies.describe(include=['object']) # Can include the appropiate optional arguments.


# In[ ]:


# Top tip: To see a desrcription of the method describe, then -
#
# Put the cursor anywhere in the parenthesis, and hit Shift + Tab.
# (Try hitting the Tab key 1, 2, 3 or 4 times!).


# # Renaming columns in a Pandas DataFrame.

# In[14]:


import pandas as pd


# In[15]:


ufo = pd.read_csv('http://bit.ly/uforeports')


# In[16]:


ufo.head()


# In[17]:


ufo.columns


# In[19]:


# Changing 'Colors Reported' to 'Colors_Reported' and 'Shape Reported' to 'Shape_Reported'.
# Notice that the argument is a Python dictionary.
# Note: inplace = True makes the changes to the DataFrame ufo.

ufo.rename(columns = {'Colors Reported':'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace = True)


# In[20]:


ufo.columns


# In[22]:


# Another way to change the column names.

ufo_cols = ['city', 'colors reported', 'shape reported', 'state', 'time'] # i.e. change to lower case.


# In[23]:


ufo.columns = ufo_cols


# In[24]:


ufo.head()


# In[25]:


# ...and another way.

ufo = pd.read_csv('http://bit.ly/uforeports', names=ufo_cols, header=0)


# In[28]:


ufo.head()


# In[30]:


# Changing all column names according to a rule.
# Can use a string method.

ufo.columns


# In[36]:


ufo.columns.str.replace(' ', '_') # Relace a space with an underscore character.


# # Removing columns from a Pandas DataFrame.

# In[62]:


import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head()


# In[63]:


# Dropping the 'Colors Reported' column.

ufo.drop('Colors Reported', axis = 1, inplace = True)


# In[64]:


ufo.head()


# In[65]:


# Dropping multiple columns.
ufo.drop(['City','State'], axis = 1, inplace = True)
ufo.head()


# In[66]:


# Dropping rows. 
# For example dropping row indices 1 and 3.

ufo.drop([1, 3], axis = 0, inplace = True)
ufo.head()


# # Sorting a DataFrame or Series.

# In[77]:


import pandas as pd


# In[78]:


movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head() # head() is a method (function) of movies.


# In[79]:


# Sorting a Series (e.g. the 'title' Series).

movies.title.sort_values() 


# In[80]:


# movies['title'].sort_values()
movies['title'].sort_values(ascending = False)


# In[81]:


# Note that the original DataFrame has not changed.
movies['title']


# In[82]:


# Sorting a DataFrame with respect to a particular column (Series).

movies.sort_values('duration')


# In[86]:


# Sorting using multiple columns.
# E.g. First sort in terms of genre, and then within
# each genre sort in terms of duration.

movies.sort_values(['genre','duration'])


# # Filtering rows of a Pandas DataFrame by column value.

# In[118]:


import pandas as pd


# In[119]:


movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head() # head() is a method (function) of movies.


# In[120]:


movies.shape


# In[121]:


# Filtering this DataFrame so that only rows with duration of at least 200 minutes are shown.


# In[122]:


# First Method. Step be step approach. (My approach - DB)
# Create a list of boolean types.

booleans = []
for length in movies['duration']:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)      
print(booleans[0:5])
        


# In[123]:


len(booleans)


# In[124]:


for index in range(len(booleans)):
    if not booleans[index]:
        movies.drop(index, axis = 0, inplace = True)
        
movies  


# In[126]:


# Second method.
movies = pd.read_csv('http://bit.ly/imdbratings')

# Create a list of booleans as before.
booleans = []
for length in movies['duration']:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False) 
        
# Convert this list to a Series.

is_long = pd.Series(booleans)

is_long.head()


# In[128]:


# This Series can then be passed to the DataFrame and the appropriate rows are deleted.
# Note: the DataFrame knows how to deal with a Series inside the brackets [].
# i.e. rows are selected that correspond to the booleans with True values.

movies[is_long]


# In[129]:


# Third method. Using concise code.

movies = pd.read_csv('http://bit.ly/imdbratings')

is_long = movies['duration'] >= 200
# Note: movies['duration'] is a Series, and the comparison is element-wise. The result is
# a Series containing boolean values.

is_long.head()


# In[130]:


# ...and finally. 
movies[is_long]


# In[131]:


# One more thing! To save even more typing, the above code can be condensed to.
movies = pd.read_csv('http://bit.ly/imdbratings')
movies[movies['duration'] >= 200] # ...or even movies[movies.duration >= 200].


# In[132]:


type(movies[movies['duration'] >= 200])


# In[134]:


# Since movies[movies['duration'] >= 200] is a DataFrame it is possible to select certain columns only.

movies[movies['duration'] >= 200][['genre','duration']]


# In[138]:


# However, best practice is to use the .loc method.

movies = pd.read_csv('http://bit.ly/imdbratings')
movies.loc[movies['duration'] >= 200, ['genre','duration']] # Note: .loc[rows, cols] format.


# # Applying multiple filter criteria to a Data Frame.

# In[149]:


import pandas as pd


# In[150]:


movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()


# In[143]:


# Selecting movies that are of the genre drama and duration of at least 200 minutes.

movies[(movies['duration'] >= 200) & (movies['genre'] == 'Drama') ] # Note: '&' must be used with Series. 'and' does not work.


# In[146]:


# Also note that '|' should be used instead of 'or'.

movies[(movies['duration'] >= 220) | (movies['star_rating'] >= 9) ]


# In[154]:


# A shortcut for multiple 'or' conditions.

(movies['genre'] == 'Crime') | (movies['genre'] == 'Drama') | (movies['genre'] == 'Action')


# In[155]:


# The above can be written as below.

movies['genre'].isin(['Crime', 'Drama', 'Action'])
# This is useful if there are many 'or' conditions.


# # A variety of topics.

# In[ ]:


# Reading in a subset of columns from a .csv file.


# In[ ]:


import pandas as pd


# In[156]:


ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.columns


# In[157]:


ufo = pd.read_csv('http://bit.ly/uforeports', usecols = ['City', 'State', 'Time'])
ufo.columns


# In[158]:


ufo = pd.read_csv('http://bit.ly/uforeports', usecols = [0, 3, 4])
ufo.columns


# In[162]:


# Reading in the first 4 rows of a .csv file, for example.

ufo = pd.read_csv('http://bit.ly/uforeports', nrows = 4)
ufo


# In[165]:


# Iterating through a Series.
# Just use a Python for loop.

for c in ufo['City']:
    print(c)


# In[168]:


# Iterating through a DataFrame.
# Use the DataFrame method iterrows().

for index, row in ufo.iterrows():
    print(index, row['City'], row['State'])


# In[171]:


# Dropping non-numeric columns from a DataFrame.

drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.dtypes


# In[176]:


import numpy as np

drinks.select_dtypes(include = [np.number]).dtypes


# # Using the 'axis' parameter in Pandas.

# In[ ]:


# It's a bit confusing!
# Just experiment with axis=0 or axis=1 and see what you get!


# In[2]:


import pandas as pd


# In[4]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()


# In[5]:


drinks.drop('continent', axis=1).head() # axis=1 means it is a 'column-wise operation'.
                                        


# In[6]:


drinks.drop(2, axis=0).head() # axis=0 means it is a 'row-wise operation'.
                          
                            


# In[7]:


drinks.head() # Note: the drinks DataFrame has not been changed since inplace=False by default.


# In[8]:


drinks.mean() 


# In[9]:


drinks.mean(axis=0) # axis=0 means it is a row-wise operation.


# In[12]:


drinks.mean(axis=1).head() # axis=1 means it is a column-wise operation.
                          


# In[15]:


# Alternatively, can use axis='index' or axis='columns'
# (To add to the confusion!).

drinks.mean(axis='index').head()


# In[16]:


drinks.mean(axis='columns').head()


# # Using string methods in Pandas.

# In[18]:


'hello'.upper()


# In[30]:


import pandas as pd


# In[31]:


orders = pd.read_table('http://bit.ly/chiporders')


# In[32]:


orders.head()


# In[33]:


orders['item_name'].str.upper() # Using the string method str.


# In[36]:


orders['item_name'].str.contains('Chicken')


# In[38]:


# Chaining string methods.
# E.g. Removing the brackets in the choice_description column.

orders['choice_description'].str.replace('[', '').str.replace(']', '')


# In[39]:


# Can do the same thing with regular expressions.
orders['choice_description'].str.replace('[\[\]]', '').str.replace(']', '')


# # Changing the data type of a Pandas Series.

# In[40]:


import pandas as pd


# In[41]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()


# In[43]:


drinks.dtypes


# In[46]:


# Changing beer_servings data type.

drinks['beer_servings'] = drinks['beer_servings'].astype(float)
drinks.dtypes


# In[48]:


# Alternatively.
drinks = pd.read_csv('http://bit.ly/drinksbycountry', dtype={'beer_servings':float})
drinks.dtypes


# In[59]:


orders = pd.read_table('http://bit.ly/chiporders')
orders.head()


# In[60]:


orders.dtypes
# Note: item_type is an 'object' (basically a string).


# In[61]:


# In order to consider the item_types as numbers then the type must be converted.
orders['item_price'].str.replace('$','').astype(float).mean()


# In[66]:


# Coverting booleans to integers.
orders['item_name'].str.contains('Chicken').astype(int).head()


# In[ ]:


# Using 'groupby' in Pandas.


# In[67]:


import pandas as pd


# In[68]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()


# In[69]:


# Finding the mean of the beer_servings column.
drinks['beer_servings'].mean()


# In[73]:


# Average beer_servings by continent.
drinks.groupby('continent')['beer_servings'].mean()


# In[77]:


drinks[drinks['continent']=='Africa'].head()


# In[78]:


drinks[drinks['continent']=='Africa']['beer_servings'].mean()


# In[79]:


drinks.groupby('continent')['beer_servings'].max()


# In[80]:


drinks.groupby('continent')['beer_servings'].agg(['count', 'min', 'max', 'mean'])


# In[82]:


# Can find the mean across all columns.
drinks.groupby('continent').mean()


# In[83]:


# Plotting in the notebook.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


drinks.groupby('continent').mean().plot(kind='bar')


# # Exploring a Pandas Series.

# In[88]:


import pandas as pd

movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()


# In[89]:


movies.dtypes


# In[90]:


movies['genre'].describe()


# In[91]:


movies['genre'].value_counts()


# In[92]:


movies['genre'].value_counts(normalize=True)


# In[93]:


type(movies['genre'].value_counts())


# In[94]:


movies['genre'].value_counts().head()


# In[95]:


movies['genre'].unique()


# In[96]:


movies['genre'].nunique()


# In[98]:


pd.crosstab(movies['genre'], movies['content_rating'])


# In[102]:


movies['duration'].describe()


# In[103]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[105]:


movies['duration'].plot(kind='hist')


# In[107]:


movies['genre'].value_counts().plot(kind='bar')


# # Handling missing values with Pandas.

# In[108]:


import pandas as pd


# In[110]:


ufo = pd.read_csv('http://bit.ly/uforeports')


# In[112]:


ufo.tail()


# In[113]:


ufo.isnull().tail()


# In[114]:


ufo.notnull().tail()


# In[116]:


ufo.isnull().sum() # isnull() is a DataFrame method.


# In[120]:


# Finding the number of missing values in each column.
pd.Series([True, False, True])


# In[121]:


pd.Series([True, False, True]).sum() # True is treated as 1, False as 0.


# In[122]:


ufo[ufo['City'].isnull()] # isnull() is a Series method.


# In[123]:


ufo.shape


# In[124]:


# Drop rows in which any of its values are missing.

ufo.dropna(how='any').shape # how='any' is the default.


# In[125]:


ufo.shape


# In[126]:


# Drop rows in which all of its values are missing.
ufo.dropna(how='all').shape


# In[128]:


# Only consider a subset of columns.
ufo.dropna(subset=['City', 'Shape Reported'],how='any').shape


# In[129]:


# Only consider a subset of columns.
ufo.dropna(subset=['City', 'Shape Reported'],how='all').shape


# In[131]:


# Filling missing values.

ufo['Shape Reported'].value_counts()


# In[133]:


# Count the missing values also.
ufo['Shape Reported'].value_counts(dropna=False)


# In[137]:


ufo['Shape Reported'].fillna(value='VARIOUS', inplace=True)


# In[138]:


ufo['Shape Reported'].value_counts(dropna=False)


# # Pandas indices.

# In[ ]:


# Part 1.


# In[139]:


import pandas as pd


# In[140]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')


# In[141]:


drinks.head()


# In[145]:


drinks.index


# In[146]:


drinks.columns


# In[147]:


drinks.shape # The index and columns are not included in the DataFrame.


# In[150]:


pd.read_table('http://bit.ly/movieusers', header=None, sep='|').head()


# In[151]:


drinks[drinks['continent']=='South America']


# In[152]:


# Selecting an element of the DataFrame.
drinks.loc[23, 'beer_servings']


# In[154]:


# Can set 'country' as the new index.

drinks.set_index('country', inplace=True)
drinks.head()


# In[155]:


drinks.index


# In[156]:


drinks.columns
# Note: 'country' is no longer a column.


# In[157]:


drinks.shape
# Note: one fewer column.


# In[158]:


# Alternative way of selecting an element of the DataFrame.
drinks.loc['Brazil', 'beer_servings']


# In[159]:


# Note: The index name 'country' is not absolutely necessary (although descriptive).
drinks.index.name=None
drinks.head()


# In[160]:


# These steps can be reversed.
drinks.index.name='country'
drinks.reset_index(inplace=True)
drinks.head()


# In[161]:


drinks.describe()


# In[162]:


type(drinks.describe())


# In[163]:


drinks.describe().index


# In[164]:


drinks.describe().columns


# In[166]:


drinks.describe().loc['25%','beer_servings']


# # Part 2.

# In[7]:


import pandas as pd


# In[8]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()


# In[9]:


drinks['continent'].head()


# In[10]:


# Change 'country' to the index.
drinks.set_index('country', inplace=True)
drinks.head()


# In[11]:


drinks['continent'].head()


# In[12]:


drinks['continent'].value_counts()


# In[14]:


type(drinks['continent'].value_counts())


# In[16]:


# Since a Series is returned when using value_counts() on the DatFrame,
# its index and values can be accessed.
drinks['continent'].value_counts().index


# In[17]:


drinks['continent'].value_counts().values


# In[18]:


# Can select the value of the Series indexed by 'Africa'
drinks['continent'].value_counts()['Africa']


# In[21]:


# Sorting values.
drinks['continent'].value_counts().sort_values()


# In[22]:


# Sorting indices.
drinks['continent'].value_counts().sort_index()


# In[23]:


# Alignment.
people =pd.Series([3000000,85000], index=['Albania','Andorra'], name='population')
people.head()


# In[28]:


# Multiplying two Series of different lengths.
# Note: the appropriate multiplication is performed
# with respect to the indices that are in common.

drinks['beer_servings']*people


# In[31]:


# Concatenating a Seies to a DataFrame.

pd.concat([drinks,people], axis=1, sort=True).head()


# # Selecting multiple rows and columns from a Pandas DataFrame.

# In[1]:


# DataFrame selection using .loc[], .iloc[], .ix[].


# In[2]:


import pandas as pd


# In[3]:


ufo = pd.read_csv('http://bit.ly/uforeports')


# In[4]:


ufo.head(3) # Shw the first 3 rows.


# In[ ]:


# .loc[] - used for selecting rows and columns by label.


# In[5]:


ufo.loc[0, :] # Select row 0 and all columns.


# In[6]:


type(ufo.loc[0, :])


# In[7]:


ufo.loc[[0,1,2], :]


# In[9]:


# Equivalently...
ufo.loc[0:2, :] # NOTE: that 0:2 means 0,1,2 in .loc[]. Different from slicing!
                # i.e. 2 is included.


# In[11]:


# Column selection.
ufo.loc[:, ['City', 'State']].head()


# In[13]:


ufo.loc[:, 'City': 'State'].head() # City through to State.


# In[ ]:


# Using .loc[] with boolean conditions.


# In[19]:


ufo[ufo['City'] ==  'Oakland'] # Selecting rows that correspond to the city Oakland.


# In[20]:


# Alternatively, using .loc[].
ufo.loc[ufo['City'] ==  'Oakland', :]


# In[21]:


# .loc[] is a bit more flexible. Can select certain columns.
ufo.loc[ufo['City'] ==  'Oakland', ['Shape Reported','State']]


# In[24]:


# .iloc[] - used for selecting rows and columns by integer position.

ufo.iloc[0:5, 0:4] # Note: that 0:4 means 0,1,2,3 for .iloc[]!!!!!
                    # i.e. 4 in not included!


# In[27]:


# .ix[] - allows intgers and labels to be mixed.
# It's confusing - don't use it.
drinks = pd.read_csv('http://bit.ly/drinksbycountry', index_col = 'country')
drinks.head()


# In[28]:


drinks.ix['Albania', 0]


# # Making Pandas DataFrames smaller and faster.

# In[2]:


import pandas as pd


# In[3]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')


# In[4]:


drinks.head()


# In[5]:


drinks.info()


# In[6]:


# Note 'object' is a reference to a python data structure e.g. string, list etc.
# Therefore to find the true memory taken up by the dataframe then the code below is needed.
drinks.info(memory_usage='deep')


# In[7]:


drinks.memory_usage(deep=True) # Memory usage for each column (in bytes).


# In[8]:


type(drinks.memory_usage(deep=True))


# In[9]:


drinks.memory_usage(deep=True).sum() # Total memory usage in bytes.
                                     # Note: 31224 bytes = 30.49Kbytes.


# In[10]:


# Saving memory in a DataFrame.
sorted(drinks['continent'].unique())


# In[ ]:


# The continents could be saved as the integers 0, 1, 2, 3... to save memory.
# Also need to store 0 -> 'Africa', 1 -> 'Asia', 2 ->'Europe'etc. (For example, a Python dictionary).


# In[11]:


# Pandas can achieve the same result with the following code.

drinks['continent'] = drinks['continent'].astype('category') # The continents are stored as integers.


# In[12]:


drinks.dtypes


# In[15]:


drinks['continent'].head()


# In[17]:


drinks['continent'].cat.codes.head()


# In[21]:


drinks.memory_usage(deep=True)


# In[22]:


# Try to do the same for the 'country' column.
drinks['country'] = drinks['country'].astype('category') 


# In[23]:


# More memory is required!!!!
drinks.memory_usage(deep=True)


# In[24]:


# More memory is need because there are many different countries, and each of these
# countries must be stored in a look up table (dictionary). Hence more memory.

drinks['country'].cat.categories


# In[1]:


# Another example using 'category'.
import pandas as pd

df = pd.DataFrame({'ID':[100,101,102,103], 'quality':['good', 'very good', 'good', 'excellent']})


# In[2]:


df


# In[3]:


# Sort in alphabetical order.
df.sort_values('quality')


# In[6]:


# It would be better to sort in the order: good, very good, excellent.
from pandas.api.types import CategoricalDtype

cats = ['good', 'very good', 'excellent']
cat_type = CategoricalDtype(categories=cats, ordered=True)
df['quality'] = df['quality'].astype(cat_type)
print (df['quality'])


# In[8]:


df.sort_values('quality')


# In[10]:


# Print out the rows with 'quality > good'.

df.loc[df['quality'] > 'good', :]


# # Using Pandas with scikit-learn to create Kaggle submissions.

# In[14]:


import pandas as pd


# In[15]:


# Titanic data set.
# This is considered the training data.

train = pd.read_csv('http://bit.ly/kaggletrain')


# In[16]:


train.head()


# In[17]:


feature_cols = ['Pclass', 'Parch']


# In[18]:


X = train.loc[:, feature_cols]


# In[19]:


X.shape


# In[20]:


y = train['Survived']


# In[21]:


y.shape


# In[22]:


# Note: scikit-learn is happy with DataFrame and Series objects.
# i.e. it is not necessary to convert them to Numpy arrays for example.

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)


# In[23]:


# Read in the test data.
# Note: the 'survived' column is missing.

test = pd.read_csv('http://bit.ly/kaggletest')


# In[24]:


test.head()


# In[26]:


X_new = test.loc[:, feature_cols]


# In[27]:


X_new.shape


# In[30]:


new_pred_class = logreg.predict(X_new)


# In[36]:


# The kaggle task was to provide the 'PassengerId' and 'new_pred_class' data.
# Create a DataFrame initilised by a dictionary object.

passenger_survival = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':new_pred_class}).set_index('PassengerId')


# In[37]:


pwd # print working directory


# In[40]:


passenger_survival.to_csv('passenger_survival.csv')


# In[45]:


# Saving a DataFrame to disk.
train.to_pickle('train.pkl')


# In[46]:


# Reading from disk.
my_df = pd.read_pickle('train.pkl')
my_df.head()


# In[84]:


# Random sampling of rows

import pandas as pd

ufo = pd.read_csv('http://bit.ly/uforeports')

# Sample 3 rows at random.
ufo.sample(n=3, random_state=42) # random_state is a random seed.


# In[85]:


# Sample a (decimal) fraction of the rows.
ufo_sample = ufo.sample(frac=0.0006, random_state=88)
ufo_sample


# In[86]:


# Producing a train and test split for machine learning.
train = ufo_sample.sample(frac=0.75, random_state=999) # 75%  is training data.
test = ufo_sample.loc[~ufo_sample.index.isin(train.index), :] # The remaining 25% is the test data.


# In[87]:


train


# In[88]:


test


# # Creating dummy variables in Pandas.

# In[91]:


import pandas as pd


# In[94]:


train = pd.read_csv('http://bit.ly/kaggletrain')


# In[95]:


train.head()


# In[97]:


# Creating a dummy variable for the 'Sex' column.
# Create a new column.
train['Sex_male'] = train['Sex'].map({'female':0, 'male':1})


# In[98]:


train.head()


# In[140]:


# Another way to do it.

train = pd.read_csv('http://bit.ly/kaggletrain')

pd.get_dummies(train['Sex'])


# In[141]:


# If there are x values in the column then only need x-1 dummy variables.

SEX_dummies = pd.get_dummies(train['Sex'], prefix = 'SEX').iloc[:, 1:]


# In[142]:


SEX_dummies


# In[143]:


train.head()


# In[144]:


pd.concat([train, SEX_dummies], axis=1).drop('Sex', axis=1) # Drop 'Sex' and append 'SEX_male'.


# In[145]:


# Can create dummy variables for 'Embarked' column which has three values. 
train['Embarked'].value_counts()


# In[146]:


Embarked_dummies = pd.get_dummies(train['Embarked'], prefix = 'Embarked')
Embarked_dummies


# In[147]:


X_Embarked_dummies = Embarked_dummies.iloc[:, 1:] # Drop the first column.
X_Embarked_dummies


# In[148]:


train = pd.concat([train, X_Embarked_dummies], axis=1).drop('Embarked', axis=1)
train


# In[152]:


train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()


# In[156]:


# Passing several columns to get_dummies().
# Note: the first dummy variables are dropped.

pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)


# # Working with dates and times in Pandas.

# In[1]:


import pandas as pd


# In[2]:


ufo = pd.read_csv('http://bit.ly/uforeports')


# In[3]:


ufo.head()


# In[4]:


ufo.dtypes


# In[8]:


# Selecting the number of hours from the 'Time' column.
# Can use string slicing (e.g. 5 characters from the end to 3 characters from the end).
# Also, may want to convert the string to an integer.

ufo['Time'].str.slice(-5,-3).astype(int).head()


# In[9]:


# Alternatively convert the 'Time' column to Pandas date time format.

ufo['Time'] = pd.to_datetime(ufo['Time'])


# In[10]:


ufo.head()


# In[11]:


ufo.dtypes


# In[ ]:


# Note: there are many options with dat time if it isn't producing
# the desired result.


# In[13]:


# Selecting the number of hours from the 'Time' column.

ufo['Time'].dt.hour.head()


# In[16]:


# Selecting the day name from the 'Time' column.

ufo['Time'].dt.weekday_name.head()


# In[18]:


# Selecting the day of the year from the 'Time' column.

ufo['Time'].dt.dayofyear.head()


# In[21]:


# producing timestamps.
ts = pd.to_datetime('1/1/1999')
ts


# In[23]:


# Filtering using a timestamp.

ufo.loc[ufo['Time']>ts, :].head()


# In[25]:


ufo['Time'].max()


# In[26]:


ufo['Time'].max() - ufo['Time'].min()


# In[27]:


(ufo['Time'].max() - ufo['Time'].min()).days


# In[28]:


# Plot of the number of ufo reports by year.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


# Create a new column 'Year'
ufo['Year'] = ufo['Time'].dt.year


# In[31]:


ufo.head()


# In[33]:


ufo['Year'].value_counts().head()


# In[43]:


ufo['Year'].value_counts().sort_index().head()


# In[45]:


ufo['Year'].value_counts().sort_index().plot()


# # Removing duplicate rows with Pandas.

# In[47]:


import pandas as pd


# In[49]:


# Read a dataset of movie reviewers (modifying the default parameter values for read_table).
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('http://bit.ly/movieusers', sep='|', header=None, names=user_cols, index_col='user_id')


# In[50]:


users.head()


# In[52]:


users.shape


# In[56]:


users['zip_code'].duplicated()


# In[59]:


users['zip_code'].duplicated().sum()


# In[58]:


# duplicated() is also a DataFrame method to indicate duplicate rows.
users.duplicated()


# In[60]:


users.duplicated().sum()


# In[62]:


users.loc[users.duplicated(keep='first'), :]


# In[64]:


users.drop_duplicates(keep='first').shape


# In[69]:


# Dropping duplicates considering a subset of the columns.

users.duplicated(subset=['age', 'zip_code']).sum()


# In[70]:


users.drop_duplicates(subset=['age', 'zip_code']).shape


# # How to avoid SettingWithCopy warnings in Pandas.

# In[96]:


import pandas as pd


# In[97]:


movies = pd.read_csv('http://bit.ly/imdbratings')


# In[98]:


movies.head()


# In[99]:


movies['content_rating'].isnull().sum()


# In[100]:


movies[movies['content_rating'].isnull()]


# In[101]:


movies['content_rating'].value_counts()


# In[102]:


# Change 'NOT RATED' to a missing value.

movies[movies['content_rating']=='NOT RATED']


# In[103]:


import numpy as np

# Set the 'content_rating' column to NaN.

movies[movies['content_rating']=='NOT RATED']['content_rating'] = np.nan


# In[104]:


# Unfortunately this doesn't work! SettingWithCopyWarning generated.

movies['content_rating'].isnull().sum()


# In[105]:


movies.loc[movies['content_rating']=='NOT RATED', 'content_rating'] = np.nan
movies['content_rating'].isnull().sum()


# In[106]:


top_movies = movies.loc[movies['star_rating']>=9, :]


# In[107]:


top_movies.head()


# In[108]:


# Change a cell of the DataFrame.

top_movies.loc[0, 'duration'] = 150


# In[109]:


# Again, warning generated and but a change is made to the DataFrame.

top_movies.head()


# In[110]:


# The solution to the problem is to explicity make a copy of the DataFrame.

top_movies = movies.loc[movies['star_rating']>=9, :].copy()
top_movies.loc[0, 'duration'] = 150
top_movies.head()


# In[112]:


# Note that the original DataFrame is unaltered.
movies.head()


# # Changing display options in Pandas.

# In[1]:


import pandas as pd


# In[3]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks


# In[4]:


# In order to change display options, look at the documentation for pandas.get_option().
pd.get_option('display.max_rows') # Note: The value is 60...but 60 rows are NOT being displayed!!!


# In[6]:


pd.set_option('display.max_rows', None) # Show all rows.


# In[7]:


drinks


# In[15]:


pd.reset_option('display.max_rows') # Reset back to default.
drinks


# In[ ]:


# Note: There is an option for the maximum number of columns too.


# In[17]:


train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()


# In[19]:


pd.get_option('display.max_colwidth')


# In[22]:


pd.set_option('display.max_colwidth', 1000) # Note: 'None' doesn't work in this case. 


# In[23]:


train.head()


# In[24]:


pd.get_option('display.precision')


# In[31]:


pd.set_option('display.precision', 2) # Note: it's not completely clear how this works to me!


# In[32]:


train.head()


# In[33]:


drinks.head()


# In[34]:


# Create two more columns with large numbers in them.

drinks['x'] = drinks['wine_servings']*1000
drinks['y'] = drinks['total_litres_of_pure_alcohol']*1000
drinks.head()


# In[37]:


# Putting commas into numbers. i.e. 23456 -> 12,456.

pd.set_option('display.float_format', '{:,}'.format)
drinks.head()


# In[ ]:


# Note: The above only works on floats (y) and not integers (x)!


# In[38]:


# Displaying all the options.
pd.describe_option()


# In[39]:


# Displaying a subset of the options.
pd.describe_option('rows')


# In[41]:


# Resetting all options.

pd.reset_option('all')


# # Creating a Pandas DataFrame from another object.

# In[1]:


import pandas as pd


# In[8]:


# Create a DataFrame from a dictionary.
# Order the columns uding 'columns' keyword.
df =pd.DataFrame({'id':[100,101,102], 'colour':['red','blue', 'red']}, columns=['id', 'colour'], index=['a', 'b', 'c'])
df


# In[10]:


# Create a DataFame from a list of lists.

pd.DataFrame([[100,'red'],[101,'blue'],[102,'red']], columns=['id', 'colour'], index=['a', 'b', 'c'])


# In[11]:


# Create a DataFame from a Numpy array.

import numpy as np


# In[13]:


arr = np.random.rand(4,2)
arr


# In[15]:


pd.DataFrame(arr, columns=['one','two'])


# In[19]:


pd.DataFrame({'student':np.arange(100,110,1), 'test':np.random.randint(60,101,10)})


# In[21]:


# Can set one of the columns as an index.

pd.DataFrame({'student':np.arange(100,110,1), 'test':np.random.randint(60,101,10)}).set_index('student')


# In[26]:


# Attaching a Series to a DataFrame.

s = pd.Series(['round','square'], index=['c','b'], name='shape')


# In[27]:


df


# In[31]:


pd.concat([df,s], axis=1, sort=False)


# # Applying functions to a Pandas Series or DataFrame.

# In[32]:


import pandas as pd

train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()


# In[35]:


# map() is a Series method.

train['Sex_num'] = train['Sex'].map({'female':0, 'male':1})
train.head()


# In[37]:


train.loc[0:4, ['Sex','Sex_num']]


# In[38]:


# apply() is a Series and dataFrame method.

train['Name_length'] = train['Name'].apply(len) # len is a function.


# In[39]:


train.loc[0:4, ['Name','Name_length']]


# In[40]:


import numpy as np


# In[44]:


train['Fare_ceil'] = train['Fare'].apply(np.ceil)
train.loc[0:4, ['Fare_ceil','Fare']]


# In[46]:


train['Name'].str.split(',').head()


# In[47]:


def get_element(my_list,position):
    return my_list[position]


# In[49]:


train['Name'].str.split(',').apply(get_element, position=0).head()


# In[50]:


# Can be achieved using a lambda function.
train['Name'].str.split(',').apply(lambda x: x[0]).head()


# In[55]:


# apply() as a DataFrame method.

drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()


# In[58]:


drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)


# In[59]:


drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=1).head()


# In[61]:


# argmax() is sometimes needed.
drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis=1).head()


# In[63]:


# applymap() is a DataFrame method.
# It applies a function to every element of a DataFrame.

drinks.loc[:, 'beer_servings':'wine_servings'] = drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float)
drinks.head()


# # Using MultiIndex in Pandas.

# In[1]:


import pandas as pd


# In[24]:


stocks = pd.read_csv('http://bit.ly/smallstocks')
stocks


# In[6]:


stocks.index


# In[9]:


stocks.groupby('Symbol')['Close'].mean()


# In[16]:


ser = stocks.groupby(['Symbol', 'Date'])['Close'].mean()
ser


# In[17]:


type(ser)


# In[12]:


ser.index


# In[13]:


# Unstacking a multiIndex Series into a DataFrame
df = ser.unstack()
df


# In[14]:


type(df)


# In[18]:


# Alternatively.

stocks.pivot_table(values = 'Close', index = 'Symbol', columns = 'Date')


# In[19]:


ser


# In[20]:


ser.loc['AAPL']


# In[22]:


ser.loc['AAPL', '2016-10-03']


# In[25]:


ser.loc[:, '2016-10-03']


# In[26]:


df.loc['AAPL', '2016-10-03']


# In[27]:


df.loc[:, '2016-10-03']


# In[28]:


stocks


# In[ ]:


# Create a multiIndex on this DataFrame.


# In[29]:


stocks.set_index(['Symbol', 'Date'], inplace=True)
stocks


# In[30]:


stocks.index


# In[31]:


stocks.sort_index(inplace=True)


# In[32]:


stocks


# In[33]:


stocks.loc['AAPL']


# In[35]:


stocks.loc[('AAPL', '2016-10-03' ),:]


# In[36]:


stocks.loc[('AAPL', '2016-10-03' ), 'Close']


# In[37]:


stocks.loc[(['AAPL', 'MSFT'], '2016-10-03' ), :]


# In[38]:


stocks.loc[(['AAPL', 'MSFT'], '2016-10-03' ), 'Close']


# In[41]:


stocks.loc[('AAPL', ['2016-10-03', '2016_10_04']), 'Close']


# In[44]:


# Selecting all stocks on the two dates requires using slice(None) - for some reason!
stocks.loc[(slice(None), ['2016-10-03', '2016-10-04']), 'Close']


# In[ ]:


# Concatenating on more than one level.


# In[47]:


close = pd.read_csv('http://bit.ly/smallstocks', usecols=[0,1,3], index_col=['Symbol', 'Date'])
close


# In[48]:


volume = pd.read_csv('http://bit.ly/smallstocks', usecols=[0,2,3], index_col=['Symbol', 'Date'])
volume


# In[50]:


# Merging DataFrames.

both = pd.merge(close, volume, left_index=True, right_index=True)
both


# In[ ]:




