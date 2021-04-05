#!/usr/bin/env python
# coding: utf-8

# # Pre-Modelling: Data Preprocessing and Feature Exploration
# # in Python.

#  **Goal.**
#  - Goal:
#      - Pre-modelling/modelling is 80%/20% of work.
#      - Show the importance of data processing, feature exploration, and \
#        feature engineering on modelling performance.
#      - Go over a few effective pre-modelling steps.
#      - This is only a small subset of pre-modelling.
#  - Format:
#      - Tutorial style.
#      - Walk through concepts and code (and point out libraries).
#      - Use an edited version of the 'adult' dataset (to predict income) with \
#        the objective of buliding a binary classification model.
#  - Python libraries:
#      - Numpy.
#      - Pandas.
#      - Scikit Learn.
#      - Matplotlib.
#      - Almost all work flow is covered by these four libraries.
#      
# Source of 'adult' dataset: (http://archive.ics.uci.edu/ml/datasets/Adult).

# # Agenda

# 1. Modelling Overview.
# 2. Introduce the Data.
# 3. Basic Data Cleaning.
#     1. Dealing with data types.
#     2. Handling missing data.
# 4. More Data Exploration
#     1. Outlier detection.
#     2. Plotting distributions.
# 5. Feature Engineering
#     1. Interactions between features.
#     2. Dimensionality reduction using PCA.
# 6. Feature Selection and Model Building.
#     

# In[1]:


# Part 1: Modelling Overview


# __Review of predictive modelling__
# - __definition__
#     - Statistical technique to predict unknown outcomes.
#     - Example used in this Notebook
#         - Binary classification model - determine the probability that \
#         an observation belong to one of two groups. 
#         - Examples
#             - Wheather a person votes for one of two political candidates.
#             - Whether a credit card transaction is fraud.
#             - Whether or not a person will be diagnosed with a given disease \
#             in the next year.
# - __Data terminology__
#     - Inputs - independent variables (also called features)
#         - Predictors.
#     - Outputs - Dependent variable (also called the outcome)
#         - The target variable  for prediction.
#     - Models explain the effect that features have on the outcome.
# - __Assessing model performance__
#     - Randomly split observations into train/test sets.
#     - Build model on train set and assess performance on test set.
#     - AUC (Area Under the Curve) of ROC (Receiver Operating Characteristic) \
#     is a common performance metric
#         - True positive versus false positive rates.
# - __Types of models for binary classification__
#     - Logistic regression
#     - Random forest
#     - Gradient boosted trees
#     - Support vectore machines
#     - and so on.

# # Part 2: Introduce the Data

# Task: Given attributes about a person, predict whether their income is <=50000 or >50000.

# In[2]:


# Import data and take a look.
import numpy as np
import pandas as pd

user_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income' ]

# Note: 'r' allows backslashes (and forward slashes) in the file path name.
df = pd.read_csv(r'D:\My Documents\Python Code\Data Preprocessing/adult.csv', na_values=['#NAME?'], names=user_cols)

# First remove whitespace. (I'm not sure why there's a leading whitespace character!).
for col in user_cols:
    # print('df type:',df[col].dtype )
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(' ','')

print(df.head(5))


# In[3]:


# How much data is missing? No data is missing!!

df.isnull().sum(axis = 0).sort_values(ascending=False).head()


# In[4]:


# Insert 107 missing values into 'fnlwgt',
# 57 missing values in 'education_num',
# and 48 missing values in 'age'.
import random

random.seed(12)

num_rows = df.shape[0]
print('num_rows: ', num_rows)

def gen_rand_list(limit, num_rows):
    RandomListOfIntegers = [random.randrange(num_rows) for iter in range(limit)]
    return RandomListOfIntegers

fnlwgt_nan_indices = gen_rand_list(107, num_rows)
ed_num_nan_indices = gen_rand_list(57, num_rows)
age_nan_indices = gen_rand_list(48, num_rows)

df.loc[fnlwgt_nan_indices, 'fnlwgt'] = np.nan
df.loc[ed_num_nan_indices, 'education_num'] = np.nan
df.loc[age_nan_indices, 'age'] = np.nan


# In[5]:


# How much data is missing now?
df.isnull().sum(axis = 0).sort_values(ascending=False).head()


# In[6]:


# Take a look at the variable 'income'.
print(df['income'].value_counts())


# In[7]:


# Assign outome as 0 if income is <=50K, and 1 if income is >50K.

df['income'] = [0 if x ==  '<=50K' else 1 for x in df['income']]

# Assign X as a Dataframe of features and y as a Series of outcomes.
X = df.drop('income', 1) # 1 indicates column.
y = df['income']

print(df['income'].value_counts())
print()

print(X.head(5))


# In[8]:


print(y.head(5))


# # Part 3: Basic Data Cleaning

# __A. Dealing with data types__
# - There are three main data types:
#     - Numeric, e.g. income, age.
#     - Categorical, e.g. gender, nationality.
#     - Ordinal, e.g. low/medium/high.
#     
# - Models can only handle numeric features.
# 
# - Must convert ordinal and categorical features into numeric features.
#     - Create dummy features.
#     - Tranform a categorical feature into a set of dummy features, each representing
#     a unique category.
#     - In the set of dummy features, 1 indicates that the observation belongs to that category.

# In[9]:


# Education is a categorical feature.
print(X['education'].head(5))


# In[10]:


# Use get_dummies in pandas.
# Another option: OneHotEncoder in sckit learn.
# Note: if you have K categories, then K-1 dummies are required.
# (Notice how 'Bachelors' column is missing below).

print(pd.get_dummies(X['education']).head(5))


# In[11]:


# Decide which categorical variables you want to use in the model.
# (Don't automatically convert them to dummy features).

for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        print("Feature '{col_name}' has '{unique_cat}'' unique categories".format(
                col_name=col_name, unique_cat=unique_cat))


# In[12]:


# Although 'native_country' has a lot of unique categories, most categories
# only have a few observations.

print(X['native_country'].value_counts().sort_values(ascending=False).head(10))

print(X['native_country'].head(10))


# In[13]:


# In this case bucket low frequency events as 'Other'.

X['native_country'] = ['United-States' if x == 'United-States' else 'Other' for x in X['native_country']]

print(X['native_country'].value_counts().sort_values(ascending=False))


# In[14]:


# Print the data type of each feature.
X.dtypes


# In[15]:


# Create a feature of lists to dummy.
todummy_list = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'] 


# In[16]:


# Function to dummy all the categorical variables used for modelling.
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


# In[17]:


X = dummy_df(X, todummy_list)
print(X.head(5))


# In[18]:


for col in X.columns:
    print(col)


# __B. Handling missing data__
# - Models cannot handle missing data.
# - Simplest solution
#     - Remove observations/features that have missing data.
# - But, removing missing data can have lots of issues
#     - If data is randomly missing: then potentially a lot of data is removed.
#     - Data is not randomly missing: in addition to losing data  \
#     biases may be introduced.
#     - Ususally this is a poor solution.
# - An alternate solution is to use imputation
#     - Replace missing values with another value.
#     - Strategies: mean, median, highest frequency of a given feature.
# 

# In[19]:


# How much data is missing? 

X.isnull().sum(axis = 0).sort_values(ascending=False).head()


# In[20]:


# Impute missing values using SimpleImputer in sklearn.impute.
import sklearn
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='median')

X['fnlwgt'] = imp.fit_transform(X['fnlwgt'].values.reshape(-1,1))[:,0]
X['education_num'] = imp.fit_transform(X['education_num'].values.reshape(-1,1))[:,0]
X['age'] = imp.fit_transform(X['age'].values.reshape(-1,1))[:,0]


# In[21]:


# And how much data is missing now? None.
X.isnull().sum(axis = 0).sort_values(ascending=False).head()


# # Part 4: More Data Exploration

# - A large part of pre-modelling and modelling workflow can be automated.
# - But understanding the problem, domain, and data is extremely important  \
# for building high performance models.
# - This section covers some tools used for exploring data in order to make smarter decisions.

# __A. Outlier detection__
# - An outlier is an observation that deviates drastically from other observations in the dataset.
# - Occurrence
#     - Natural, e.g. Mark Zuckerberg's income.
#     - Error, e.g. Human weight of 2000lb, due to mistyping an extra 0.
# - Why are they problematic?
#     - Naturally occurring:
#         - Not necessarily problematic.
#         - But can skew the model by affecting the slope (see image below).
#     - Error
#         - Indicative of data quality issues.
#         - Treat in the same way as a missing value, i.e imputation.
# - There are many approaches for detecting outliers. Two of these:
#     - Tukey IQR.
#     - Kernel density estimation.

# In[22]:


# This code does not work!

# from IPython.display import Image
# Image(filename='outliers.jpg')


# __Outlier detection - Tukey IQR (Inter-quartile range)__
# - Identifies extreme values in data.
# - Outliers are defined as:
#     - Values below Q1 - 1.5(Q3-Q1) and above Q1 + 1.5(Q3-Q1).
# - Standard deviation from the mean is another method for detecting extreme values
#     - But it can be problematic:
#         - Assumes normality.
#         - Sensitive to extreme values.

# In[23]:


# This code does not work!

# from IPython.display import Image
# Image(filename='tukeyipr.jpg')


# In[24]:


def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x<floor) | (x>ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values


# In[25]:


tukey_indices, tukey_values = find_outliers_tukey(X['age'])
print(np.sort(tukey_values))


# __Outlier detection - kernel density estimation__
# - Non-parametric way to estimate the probability density function of a given feature.
# - Can be advantageous compared to extreme value detection (e.g. Tukey IQR).
#     - Captures outliers in bi-model distributions.

# In[26]:


from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

def find_outliers_kde(x):
    x_scaled = scale(list(map(float, x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw="scott", fft=True)
    pred = kde.evaluate(x_scaled)
    
    n = sum(pred<0.05)
    outlier_ind = np.asarray(pred).argsort()[:n]
    outlier_value = np.asarray(x)[outlier_ind]
    
    return outlier_ind, outlier_value


# In[27]:



kde_indices, kde_values = find_outliers_kde(X['age'][:5000]) # Use a subset of the data for illustrative purposes.

# kde_indices, kde_values = find_outliers_kde(X['age']) # This takes too long to process!
print(np.sort(kde_values))


# __B. Distribution of Features__
# - A histogram is a simple representation of the distribution of values of a given feature.
# - X-axis represents value bins and y-axis represents the frequency of an observations falling in that bin.
# - It is interesting to look at distributions by outcome categories.

# In[28]:


# Use pyplot in matplotlib to plot histograms.
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

def plot_histogram(x):
    plt.hist(x, color='gray', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


# In[29]:


plot_histogram(X['age'])
print(type(X['age']))


# In[30]:


# Plot histograms to show distributions of features by 
# dependent variable (DV) categories.

def plot_histogram_dv(x, y):
    plt.hist(list(x[y==0]), color='blue', alpha=0.5, label='DV=0') # y=0 -> <=50K.
    plt.hist(list(x[y==1]), color='red', alpha=0.5, label='DV=1') # y=1 -> >50K.
    plt.title("Histogram of '{var_name}' by DV category".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()


# In[31]:


plot_histogram_dv(X['age'], y)


# # Part 5: Feature Engineering

# __A. Interactions amongst features__
# - A simple two-way interaction is represented by:-
#     - X3 = X1*X2 where X3 is the interaction between X1 and X2.
# - Can add interaction terms as additional new feature to you model; useful for the  \
# model if the impact of two or more features on the outcome is non-additive.
# 
# - Example
#     - Interaction: education and political ideology; outcome: concerns about climate change.
#     - While an increase in education amongst liberals or moderates increases concerns about  \
#     climate change, an increase in education amongst conservatives has the opposite effect. 
#     - The education/political ideology interaction captures more than two features alone.
#     
# - Note that the interaction amongst dummy variables belonging to the same categorical features \
# are always zero.
# 
# - Although it is easy to calculate two-way interactions amongst all features, it is very  \
# computationally expensive. ('f choose 2' combinations,  which is approximately 0.5*f^2).
#     - 10 features = 45 two-way interaction terms.
#     - 50 features = 1225 two-way interaction terms.
#     - 100 features = 4950 two-way interaction terms.
#     - 500 features = 124750 two-way interaction terms.
#     - It is recommended to understand your data and domain if possible, and selectively  \
#     choosing interaction terms.

# In[32]:


# This code doesn't work.
#from IPython.display import Image
#Image(filename='interactions.jpg')


# In[33]:


# Use PolynomialFeatures in sklearn.preprocessing to create two-way interactions
# for all features.
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names.
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions.
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interactions with all zero values.
    noint_indices = [i for i, x in enumerate(list((df==0).all())) if x]
    df = df.drop(df.columns[noint_indices], axis=1)
    
    return df


# In[34]:


X = add_interactions(X)
print(X.head())


# __B. Dimensionality reduction using PCA__
# - Principal component analysis (PCA) is a technique that transforms a dataset of many
# features into principal components that 'summarise' the varaince that underlies the data.
# - Each principal component is calculated by finding the linear combination of features
# that maximises variance, while also ensuring zero correlation with previously calculated
# principal components.
# - Use cases for modelling:
#     - One of the most common dimensionality reduction techniques.
#     - Use if there are too many features or if observation/feature ratio is poor.
#     - Also, potentially good option if there are a lot of highly correlated variables
#     in the dataset.
# - Unfortunately PCA makes it a lot harder interpret models. 

# In[35]:


# This code does not work!
#from IPython.display imort Image
#Image(filename='pca.jpg')


# In[36]:


# Use PCA from sklearn.decomposition to find PCA.
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_pca = pd.DataFrame(pca.fit_transform(X))


# In[37]:


print(X_pca.head())


# # Part 6: Feature Selection and Model Building

# __Build model using processed data__
# 

# In[38]:


# Use train_test_spilt in sklearn.model_selection to split data into tain and test sets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)


# In[39]:


# The total number of features has grown substantially after dummying and adding interaction terms.

print(df.shape)
print(X.shape)


# In[40]:


# Such a large set of features can cause overfitting and slow computations.
# Use feature selection to choose the most important features.

import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]


# In[41]:


print(colnames_selected)


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def find_model_perf(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat = [x[1] for x in model.predict_proba(X_test)]
    
    auc = roc_auc_score(y_test, y_hat)
    
    return auc


# In[43]:


auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)
print(auc_processed)


# __Build model using unprocessed data__

# In[44]:


# Drop missing values so that model does not throw an error.
df_unprocessed = df
df_unprocessed = df_unprocessed.dropna(axis=0, how='any')
print(df.shape)
print(df_unprocessed.shape)


# In[45]:


# Remove non-numeric columns so model does not throw an error.
for col_name in df_unprocessed.columns:
    if df_unprocessed[col_name].dtypes not in ['int32','float32', 'int64','float64']:
        df_unprocessed = df_unprocessed.drop(col_name, 1)
        
print(df.shape)
print(df_unprocessed.shape)


# In[46]:


# Split into features and outcomes.
X_unprocessed = df_unprocessed.drop('income', 1)
y_unprocessed = df_unprocessed['income']


# In[47]:


# Take a look again at the what the unprocessed feature set looks like.
print(X_unprocessed.head())


# In[48]:


# Split unprocessed data into train and test sets.
# Build model and assess performance.

X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(
    X_unprocessed, y_unprocessed, train_size=0.7, random_state=1)
auc_unprocessed = find_model_perf(X_train_unprocessed, y_train, X_test_unprocessed, y_test)
print(auc_unprocessed)


# In[49]:


# Compare model performance.
print('AUC of model with data preprocessing: {auc}'.format(auc=auc_processed))
print('AUC of model without data preprocessing: {auc}'.format(auc=auc_unprocessed))
per_improve = ((auc_processed-auc_unprocessed)/auc_unprocessed)*100 # Percentage improvement.
print('Model improvement of preprocessing: {per_improve}%'. format(per_improve=per_improve))

