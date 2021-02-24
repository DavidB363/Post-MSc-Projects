#!/usr/bin/env python
# coding: utf-8

# # Data Science Task for Koodoo job application process.
# # Analysis of an Air B&B data set.
# # David Brookes January 2021

# In[1]:


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Print current working directory (folder).
curr_dir=os.getcwd()
print(curr_dir)

# Change working directory.
# Note: 'r' allows backslashes (and forward slashes) in the file path name.

os.chdir("D:\My Documents\Python Code\Koodoo Tasks\Solutions\Data Science Project")

curr_dir=os.getcwd()
print(curr_dir)


# In[3]:


# Read the comma separated value file into a Pandas DataFrame.
airbnb = pd.read_csv('airbnb_dataset.csv')
#print(airbnb.head())
#print(type(airbnb))


# In[4]:


# Count the number of rows and columns in the dataframe.

num_rows = airbnb.shape[0]
num_rows = airbnb.shape[1]

print('Number of rows: ', airbnb.shape[0])
print('Number of columns: ', airbnb.shape[1])


# In[5]:


# Drop the 'property_id' column as the values are arbitrary.

airbnb2 = airbnb.drop('property_id', axis = 1, inplace= False)
#print(airbnb2.head())


# In[6]:


# Calculate the correlation matrix.
corrMatrix = airbnb2.corr()
print(corrMatrix)


# In[7]:


# Plot the heat map for the correlation matrix.

ax = sns.heatmap(
    corrMatrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# Save plot to file.

path = os.path.join(curr_dir, "correlation_matrix", "correlation_matrix" + "." + "png")
print('path: ',path)
plt.savefig(path, format="png", dpi=300)

# Show the plot.
plt.show()


# In[8]:


# Histogram plots.

features =['State', 'room_type']

for f in features:
    f_names = airbnb2[f].unique()
    
    images_path = os.path.join(curr_dir, f)   
    
    for f_name in f_names:
        for price_range in [' room prices < $1000', ' room prices >= $1000' ]:
        
            if price_range == ' room prices < $1000':
                feature_prices = airbnb2[(airbnb2[f] == f_name) & (airbnb2['price'] < 1000)]['price']
                filename_extension = "_under_1000"
            else:
                feature_prices = airbnb2[(airbnb2[f] == f_name) & (airbnb2['price'] >= 1000)]['price']
                filename_extension = "_over_1000"

            n_bins = 100
            # Creating histogram.
            fig, axs = plt.subplots(1, 1, 
                            figsize =(10, 7),  
                            tight_layout = True) 

            # Add x, y gridlines.  
            axs.grid(b = True, color ='grey',  
            linestyle ='-.', linewidth = 0.5,  
            alpha = 0.6)  

            # Creating histogram. 
            N, bins, patches = axs.hist(feature_prices, bins = n_bins) 

            plt.xlabel(f_name)
            plt.ylabel('Frequency')
            plt.title('Histogram of '+ f_name + price_range)  

            if f_name == 'Entire home/apt': # Don't want filenames with forward slashes in them.
                f_name_path_friendly = 'Entire home or apt'
            else:
                f_name_path_friendly = f_name

            path = os.path.join(images_path, f_name_path_friendly + filename_extension + "." + "png")
            #print('path: ',path)
            plt.savefig(path, format="png", dpi=300)

            # Show plot.
            plt.show()
      

