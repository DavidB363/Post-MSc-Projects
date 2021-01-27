#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Generate the coordinate file to be used by program 'planet'.


# In[2]:


# Set the variable num_stations to any value you like.
# Note that the run time for the program goes up exponentially 
# with num_stations.
# E.g.
# numstations = 10 : 2 seconds
# numstations = 11 : 27 seconds
# numstations = 12 : 180 seconds


num_stations = 6 # The number of teleportation stations.
import random

random.seed(169)

def generate_rand_coords():
    rand_coord = []
    for i in range(3): # x, y and z coordinates.
        #rand_coord.append(random.randint(1,10)) # Test code!
        
        # Note: The original question specified random numbers between -10000.00 and 10000.00.
        # Generate a random number between -10000.00 and 10000.00.
        #rand1 = random.random()
        #rand2 = int(2*1000000*(rand1-0.5))
        #rand3 = rand2/100
        #rand_coord.append(rand3)
        
        
        # NOTE: changing the random numbers to be between 0 and 10000.00 makes for more interesting results!!!
        # Generate a random number between 0.00 and 10000.00.
        rand1 = random.random()
        rand2 = int(1000000*(rand1))
        rand3 = rand2/100
        rand_coord.append(rand3)
        
        
    return(rand_coord)
        
random_coords = generate_rand_coords()
print(random_coords)
        


# In[3]:


# Generate coordinates for all the stations except the first. 
# The first station - index 0 - (Earth) has coordinates (0.0, 0.0, 0.0).

coord_list = [[0.0, 0.0, 0.0]]
for i in range(1,num_stations):
    rand_coords = generate_rand_coords()
    coord_list.append(rand_coords)
    
print(coord_list)


# In[4]:




f = open('input.txt', 'w+') # Write and read to file 'f'.

def write_coords_to_file(index):
    for i in range(3):
        f.write(str(coord_list[index][i])+'\n')  

# Write the coordinates for the last station (Zearth).
write_coords_to_file(num_stations-1)

# Write the number of stations.
f.write(str(num_stations)+'\n')

# Write the coordinates from the first to the last station.
for i in range(num_stations):
    write_coords_to_file(i)
    
f.close()


# In[ ]:




