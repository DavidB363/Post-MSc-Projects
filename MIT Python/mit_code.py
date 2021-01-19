#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Memoisation.

def fast_fib(n, memo ={}):
    if n == 0 or n == 1:
        return 1
    try:
        return memo[n]
    except:
        result = fast_fib(n-1, memo) + fast_fib(n-2, memo)
        memo[n] = result
        return result
    
fib(1200)


# In[13]:


my_dict = {1:20, 2:40, 3:60}

i = 4
if i in my_dict.keys():
    print(my_dict[i])


# In[15]:


# Memoisation. My vesrion!

def fast_fib(n, memo ={}):
    if n == 0 or n == 1:
        return 1
    if n in memo.keys():
        return memo[n]
    else:
        result = fast_fib(n-1, memo) + fast_fib(n-2, memo)
        memo[n] = result
        return result

fib(1200)


# In[2]:


x = list(range(4))
print(x)


# In[5]:


# Towers of Hanoi.

# Global variables.
stack = [[],[],[]]
num_disks = 4
num_moves = 0

def initialise_stack(n = 0):
    global stack
    stack = [list(range(1,n+1)),[],[]]
    
def print_config():
        print(stack[0], stack[1], stack[2])
        print('\n')
    
initialise_stack(num_disks) # Put all the disks on peg 0.
print_config()

    


# In[6]:


def move (n, start_peg = 0, inter_peg = 1, end_peg = 2):
    global num_moves
    if n == 1:
        disk_to_move = stack[start_peg].pop(0)
        stack[end_peg].insert(0, disk_to_move)
        num_moves += 1
        print_config()
        
    else:
        move(n-1, start_peg, end_peg, inter_peg)
        disk_to_move = stack[start_peg].pop(0)
        stack[end_peg].insert(0, disk_to_move)
        num_moves += 1
        print_config()
        move(n-1, inter_peg, start_peg, end_peg)
        
print_config() 
move(num_disks)  
print('number of moves =', num_moves )


# In[17]:


t = [5]
u = []
x = t.pop(0)
print(t)
print(type(t))
print(x)
print(type(x))
u.append(x)
print(u)


# In[29]:


# list.insert(index, element) 

t = []
t.insert(0,11)
print(t)


# In[56]:


2**15


# In[57]:


2**25


# In[68]:


2**6


# In[ ]:




