#!/usr/bin/env python
# coding: utf-8

# # Pipelines using Pandas.
# 
# # David Brookes July 2021.

# In[1]:


import pandas as pd

df = pd.read_csv(r'D:\My Documents\Python Code\Pipelines\Special_Events_Permits.csv')
df.head()


# In[2]:


# Convert column names to lower case with underscores.
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.lower()

print(df.columns)
print(df.shape)


# In[3]:


print(type(df['event_start_date'][0]))
print(df['event_start_date'][0])


# In[4]:


# Function to extract time information.

time_info = df['event_start_date'][0]

def extract_info(time_info):
    month = int(time_info[0:2])
    day = int(time_info[3:5])
    year = int(time_info[6:10])
    time = time_info[11:19]
    am_pm = time_info[20:22]
    
    return (month, day, year, time, am_pm)
    
month, day, year, time, am_pm = extract_info(time_info)

print(month)
print(day)
print(year)
print(time, am_pm)


# In[5]:


# Just look at a selection of the data i.e. from 2016.
booleans=[]
for time_info in df['event_start_date']:
    month, day, year, time, am_pm = extract_info(time_info)
    if year == 2016:
        booleans.append(True)
    else:
        booleans.append(False)
        
df_2016 = df[booleans]

print(df_2016.head())
print(df_2016.shape)


# # Build a machine learning model.
# - Outcome - permit_status
#     Binary - will event be "Complete" or not
# 
# - Features 
#     - everything else!
#     - Raw, transformed, combinations etc.

# # Modelling with scikit-learn.
# - Set aside test data. No peeking!

# In[6]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_2016)


# - Define outcome, and also one feature only.

# In[7]:


import numpy as np

y_train = np.where(df_train['permit_status'] == 'Complete', 1, 0)
y_test = np.where(df_test['permit_status'] == 'Complete', 1, 0)

# One feature used.
X_train = df_train[['attendance']].fillna(value=0)
X_test = df_test[['attendance']].fillna(value=0)


# In[8]:


# Fit the model.
# Create model object.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Fit model and predict on training data.

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
p_pred_train = model.predict_proba(X_train)[:,1]


# In[9]:


# Evaluation.

# Predict on test data.
p_baseline = [y_train.mean()]*len(y_test) # Simple model that predicts the mean.
p_pred_test = model.predict_proba(X_test)[:,1]

# Measure performance on the test set.
from sklearn.metrics import roc_auc_score
auc_base = roc_auc_score(y_test, p_baseline)
auc_test = roc_auc_score(y_test, p_pred_test)

print('auc_base:', auc_base)
print('auc_test:', auc_test)


# # Transformers.

# In[10]:


# Several transformations of the data may be required.
# For example, imputation followed by creating of polynomial features
# followed by standardisation.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (PolynomialFeatures,
                                  StandardScaler)
#imputer = SimpleImputer()
#quadratic = PolynomialFeatures()
#standardiser = StandardScaler()


# Instead of writing this:-   \
# X_train_imp = imputer.fit_transform(X_train_raw)   \
# X_train_quad = quadratic.fit_transform(X_train_imp)   \
# X_train = standardiser.fit_transform(X_train_quad)   
# 
# and
# 
# X_test_imp = imputer.transform(X_test_raw)   \
# X_test_quad = quadratic.transform(X_test_imp)   \
# X_test = standardiser.transform(X_test_quad)   
# 
# Create a pipeline instead!

# In[11]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([('imputer', SimpleImputer())
                        ,('quadratic', PolynomialFeatures())
                        ,('standardiser', StandardScaler())])

X_train_pipeline_processed = pipeline.fit_transform(X_train)
X_test_pipeline_processed = pipeline.transform(X_test)


# In[12]:


print(X_train)


# In[13]:


print(X_train_pipeline_processed)


# In[14]:


# A useful function - FunctionTransformer().

from sklearn.preprocessing import FunctionTransformer

logger = FunctionTransformer(np.log1p) # Choose any function you like.

X_train_log = logger.transform(X_train)
print(X_train_log)


# Or, create a custom transformer!

# In[15]:


from sklearn.base import TransformerMixin

class Log1pTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        Xlog = np.log1p(X)
        return Xlog
    
logger_custom = Log1pTransformer()
X_train_logger_custom = logger_custom.fit_transform(X_train) # Note TransformerMixin creates fit_transform method.

print(X_train_logger_custom)

