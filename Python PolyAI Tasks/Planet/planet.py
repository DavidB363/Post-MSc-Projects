#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PolyAI Python Task 1.
# Teleportation problem.
# David Brookes January 2021.


# In[2]:


import math


# In[3]:


# Read in data from input.txt

f = open('input.txt')
lines = f.readlines()
f.close

f_index = 0 # File index.

def read_coordinates():
    global f_index
    coords = []
    for i in range(3):
        #print('Were in read_coords - f_index:', f_index)
        #print('Were in read_coords - i:', i)
        #print('Were in read_coords - lines(f_index+i):', lines[f_index])    
        coords.append(float(lines[f_index]))
        f_index +=1
    return(coords)
           

# First, read in the contents of the file.
# Read in the number of stations num_stations, and the list of coordinates of the stations coord_list.
def read_file():
    global f_index
    # First, read in the coordinates of the final destination, and just print it out.
    coords = read_coordinates()
    #print('Final station coordinates- in read_file: ', coords)
    
    # Next, read in the number of stations.
    num_stations = int(lines[f_index]) 
    #print('num_stations- in read_file: ', num_stations) 
    f_index +=1 # Advance the file index.
    
    # Then read in the coordinates for ALL the stations, and save them in a coord_list.
    coord_list = []
    for i in range(num_stations):
        coords = read_coordinates()
        coord_list.append(coords)
    return num_stations, coord_list
        
num_stations, coord_list = read_file()  

#print('num_stations = ', num_stations)
#print('coord_list:', coord_list)


# In[4]:


# Print the elements of a list.
def print_list(lst):
    for el in lst:
        print(el)
    print('')


# In[5]:


# Python function to generate all permutations of a selection of
# n numbers from a given list.

# The list consists of consecutive integers [1, 2, ..., num_stations-1]
# n is the number of numbers chosen from this list.

def permutation_new(lst, n): 
 
    # If zero numbers to choose from the list.
    if n == 0: 
        return [] 
  
    # If one number to choose from the list. 
    list1 = []
    if n == 1:
        for i in range(len(lst)): 
            m = lst[i] 
            #print('m= ', m)
            list1.append([m])
            #print('list1= ', list1)
        return (list1) 
  
    # If more than one number to choose from the list. 
  
    list2 = [] # Empty list that will store the current permutation 
  
    # Iterate the input(lst) and calculate the permutation 
    for i in range(len(lst)): 
        m = lst[i] 
  
       # Extract lst[i] or m from the list.  remLst is 
       # the remaining list. 
        remLst = lst[:i] + lst[i+1:] 
  
       # Generating all permutations where m is first 
       # element 
        for p in permutation_new(remLst, n-1): 
            list2.append([m] + p) 
    return list2 


# In[6]:


class coordinates:
    def __init__(self, coord_lst):
        self.coords = coord_lst # [x, y, z]
        self.x = coord_lst[0]
        self.y = coord_lst[1]
        self.z = coord_lst[2]        


# In[7]:


def euclidean_distance(coord1,coord2):
    c1 = coordinates(coord1)
    c2 = coordinates(coord2)
    dist = math.sqrt( (c1.x-c2.x)**2 +(c1.y-c2.y)**2 +(c1.z-c2.z)**2 )
    return(dist)


# In[8]:


class station:
    def __init__(self, index, coords):
        self.index = index
        self.coords = coords
        
    def print_vars(self):
        print('Station Index:', self.index, ' Coordinates', self.coords)
        


# In[9]:


# Assign coordinates to the station list.

station_list = []
for i in range(num_stations):
    st = station(i, coord_list[i])
    station_list.append(st)
    
#for i in range(num_stations):
#    print(station_list[i].print_vars()) 


# In[10]:


def station_distance(station1, station2):
    st1_coords = station1.coords
    st2_coords = station2.coords
    return(euclidean_distance(st1_coords, st2_coords))
    


# In[11]:


# Generate a dictionary of distances between stations.

dist_dict = dict()

for i in range(num_stations):
    for j in range(i+1,num_stations):
        st1 = station_list[i]
        st2 = station_list[j]
        dist = station_distance(st1,st2)
        dist_dict[(i,j)] = dist
        dist_dict[(j,i)] = dist
        
#for k in dist_dict.keys():
#    print(k, '{:.2f}'.format(dist_dict[k]))


# In[12]:


# Create a new distance dictionary with decreasing values of distance.

sorted_dist_list = sorted(dist_dict.items(), key=lambda x: x[1], reverse=True)

# Convert the list to a dictionary.
sorted_dist_dict = {}
for k in sorted_dist_list:
    sorted_dist_dict[k[0]] = k[1]
        
#for k in sorted_dist_dict.keys():
#    print(k, '{:.2f}'.format(sorted_dist_dict[k]))


# In[13]:


# Create a list of station transitions for each element in the permutation list.
# e.g. [1,2,3] -> [(0,1), (1,2), (2,3), (2,4)] : Note 0 is first station, and 4 is the last.

def generate_transition_list(perm_list):
    transition_list = []
    
    if perm_list == []:
        tup = (0, num_stations-1)
        transition_list.append([tup])
    else:
        for perm in perm_list:
            tran = []
            perm_len = len(perm)
            for p_index in range(perm_len):
                if (p_index == 0):
                    # Transition for the first station.
                    tup = (0, perm[p_index])
                else:
                    tup = (perm[p_index-1], perm[p_index])
                tran.append(tup)
            # Transition to the last station.    
            tup = (perm[perm_len-1], num_stations-1)
            tran.append(tup)
            transition_list.append(tran)
    return(transition_list)


# In[14]:


# Search for transition paths that minimise the maximum distance over
# all transitions on a path.

def calc_best_paths(transition_list):
    best_transition_paths = []
    min_max_dist = math.inf # Set to infinity initially.
    for tr in transition_list:
        #print('Processing transition:', tr)
        #print('Working through the sorted dictionary...')
        for tup in sorted_dist_dict.keys():
            #print('Checking tuple', tup)
            if tup in tr:
                #print(tup, 'is in this transition')
                if sorted_dist_dict[tup] < min_max_dist:
                    min_max_dist = sorted_dist_dict[tup]
                    best_transition_paths = [tr]   
                    #print('Smaller than the min distance.')     
                elif sorted_dist_dict[tup] == min_max_dist: 
                    best_transition_paths.append(tr)
                    #print('Equal to the min distance.')
                break

        #print('min_max_dist so far ...', min_max_dist)
        #print('best_transition_paths so far... ',best_transition_paths)
        #print('')
    
    return min_max_dist, best_transition_paths
   


# In[15]:


# Each transition path may consist of from 1 to num_station-1 transitions.
# Therefore this involves from 2 to num_stations teleportation stations.
# Or 0 to num_stations-2 if the first and last stations are not counted.
# (As in the case when determining permutations).

station_indices = list(range(1,num_stations-1)) # Note: 0 and num_stations-1 excluded.
                                                # (i.e. first (0 = earth) and last station (num_stations-1 = Zearth)).
#print('station_indices: ', station_indices, '\n')

overall_min_max_dist = math.inf
overall_best_transition_paths = []
for n in range(num_stations-1):
    perm_list = permutation_new(station_indices, n)
    print('n=',n,)
    #print('perm_list:',perm_list,'\n')
    transition_list = generate_transition_list(perm_list)
    #print('transition_list:',transition_list,'\n')
    min_max_dist, best_transition_paths = calc_best_paths(transition_list)
    if min_max_dist < overall_min_max_dist:
        overall_min_max_dist = min_max_dist
        overall_best_transition_paths = best_transition_paths    
    elif min_max_dist == overall_min_max_dist:
        overall_best_transition_paths.extend(best_transition_paths)
    print('Best min_max_dist so far:', overall_min_max_dist) 
    #print('Best transition paths so far:')
    #print_list(overall_best_transition_paths)
        
#print('Best overall min_max_dist:', overall_min_max_dist) 
#print('Best overall transition paths:')
#print_list(overall_best_transition_paths)


# Print max_min_dist ONLY as specified in the task question
print('Best overall min_max_dist:', overall_min_max_dist)
        

        
        

    
    
    

