#!/usr/bin/env python
# coding: utf-8

# # Data Structures Using Python.
# # March 2021.
# # David Brookes.

# In[1]:


# For each programming task there are appropriate data structures that can be used.


# # Python Built-In Data Structures.

# In[2]:


# The built-in data structures are covered in greater depth in the ThinkPython2 notebook.
# This is just an overview of strings, lists, tuples, sets and dictionaries.
#


# # Sequences: Strings, Lists and Tuples.

# # Indexing Sequences.

# In[3]:


# Any item in the sequence can be accessed using an index, starting from 0.

# String.
mystring = 'tree'

# List.
mylist = [3, 1, 4, 2]

# Tuple.
mytuple = ('The','quick','brown','fox')

print(mystring[1])
print(mylist[2])
print(mytuple[3])


# # Slicing Sequences.

# In[4]:


# Slicing. Slice out sunstrings, sublists and subtuples using indices.
# [start:end+1:step]

mystring = 'Optimism'

print(mystring[1:4])
print(mystring[1:6:2])
print(mystring[3:])
print(mystring[:5])
print(mystring[-1])
print(mystring[-3:])
print(mystring[:-2])


# # Adding/ Concatenating Sequences.

# In[5]:


# Multiplying a sequence using *.

# String.
mystring = 'tall'+'tree'

# List.
mylist = [3, 1, 4]+[1, 5, 9, 2, 6]

# Tuple.

mytuple = ('The','quick','brown','fox')+('jumped',)

print(mystring)
print(mylist)
print(mytuple)


# # Multipying Sequences.

# In[6]:


# Combining two sequences of the same type with +.

# String.
mystring = 'tree'*3

# List.
mylist = [3, 1, 4, 2]*2

# Tuple.

mytuple = ('The','quick','brown','fox')*2

print(mystring)
print(mylist)
print(mytuple)


# # Checking Membership.

# In[7]:


# Test whether an item is in or not in a sequence.

# String.
mystring = 'tree'
print('t' in mystring)

# List.
mylist = [3, 1, 4, 2]
print(3 not in mylist)

# Tuple.

mytuple = ('The','quick','brown','fox')
print('jumped' in mytuple)


# # Iterating a Sequence.

# In[8]:


# Iterating through the items of a sequence.

# Item.
x = [4, 5, 6]

for item in x:
    print(item)
    
# Index and item.
y = [7, 8, 9]

for index, item in enumerate(y):
    print(index, item)
    


# # Counting the Items in a Sequence.

# In[9]:


# String.
mystring = 'tree'
print(len(mystring))

# List.
mylist = [3, 1, 4, 2]
print(len(mystring))

# Tuple.
mytuple = ('The','quick','brown','fox')
print(len(mytuple))


# # Finding the Minimum of a Sequence.

# In[10]:


# Works with letters and numbers.

# String.
mystring = 'tree'
print(min(mystring))

# List.
mylist = [3, 1, 4, 2]
print(min(mylist))

# Tuple.
mytuple = ('The','quick','brown','fox')
print(min(mytuple))


# # Finding the Maximum of a Sequence.

# In[11]:


# Works with letters and numbers.

# String.
mystring = 'tree'
print(max(mystring))

# List.
mylist = [3, 1, 4, 2]
print(max(mylist))

# Tuple.
mytuple = ('The','quick','brown','fox')
print(max(mytuple))


# # Finding the Sum of Items in a Sequence.

# In[12]:


# Works only with numbers.

# List.
mylist = [3, 1, 4, 2]
print(sum(mylist))
print(sum(mylist[-3:]))


# # Sorting a Sequence.

# In[13]:


# sorted() function a new list of sorted items.
# The original list is not chaged.

# String.
mystring = 'tree'
print(sorted(mystring))
print(mystring)

# List.
mylist = [3, 1, 4, 2]
print(sorted(mylist))
print(mylist)

# Tuple.
mytuple = ('The','quick','brown','fox')
print(sorted(mytuple))
print(mytuple)


# # Counting the Occurences of an Item in a Sequence.

# In[14]:


# String.
mystring = 'tree'
print(mystring.count('e'))


# List.
mylist = [3, 1, 4, 2]
print(mylist.count(6))


# Tuple.
mytuple = ('The','quick','brown','fox')
print(mytuple.count('fox'))


# # Find the Index of an Item in a Sequence.

# In[15]:


# index(item) returns the index of the first occurence of the item in the sequence.

# String.
mystring = 'tree'
print(mystring.index('e'))


# List.
mylist = [3, 1, 4, 2]
print(mylist.index(3))


# Tuple.
mytuple = ('The','quick','brown','fox')
print(mytuple.index('fox'))


# # Unpacking Items of a Sequence.

# In[16]:


# String.
mystring = 'tree'
a,b,c,d = mystring
print(a, b, c, d)

# List.
mylist = [3, 1, 4, 2]
a,b,c,d = mylist
print(a, b, c, d)

# Tuple.
mytuple = ('The','quick','brown','fox')
a,b,c,d = mytuple
print(a, b, c, d)


# # Lists.

# Lists : -
# 1. General purpose.
# 2. Most widely used data structure.
# 3. Grow and shrink as needed.
# 4. Sortable.

# # Constructors - Creating a new List.

# In[17]:


# Different ways to create a new list.

x = list()
print(x)

y = ['a', 2, 'c']
print(y)

mytuple = (20, 30, 40)
z = list(mytuple)
print(z)

# Using a list comprehension.

a = [m for m in range(8)]
print(a)

b = [i**2 for i in range(10) if int(i/2) == i/2]
print(b)


# # Deleting a List, or Element in a List.

# In[18]:


x = [4,6,8,9]
del(x[1])
print(x)
del(x) # List x no longer exists.


# # Append an item to a List.

# In[19]:


x = [4,6,8,9]
x.append(7)
print(x)


# # Append a Sequence to a List.

# In[20]:


x = [4,6,8,9]
y = [30,40,50] # List.
x.extend(y)
print(x)

x = [4,6,8,9]
y = (30,40,50) # Tuple.
x.extend(y)
print(x)

x = [4,6,8,9]
y = 'hello' # String.
x.extend(y)
print(x)


# # Insert into a List an Item at a given Index.

# In[21]:


x = [4,6,8,9]
y = 88 # Number.
x.insert(3,y)
print(x)

y = (30,40,50) # Tuple.
x.insert(2,y)
print(x)

y = [7,'m'] # List.
x.insert(2,y)
print(x)


# # Pop the Last Item from a List.

# In[22]:


# pop() pops off the last item of the list and returns item.

x = [4,6,8,9]
y = x.pop()
print(x)
print(y)


# # Remove the Last Item from a List.

# In[23]:


# remove() removes the first instance of an item in a list.
x = [4,6,8,9]
y = x.remove(8)
print(x)
print(y)


# # Reverse the Items in a List.

# In[24]:


# reverse() reverses the order of the list.
# It is an in-place sort, which means the original list is changed.
x = [4,6,8,9]
x.reverse()
print(x)


# # Sort the Items in a List in Ascending order. In-Place.

# In[25]:


# Note: sorted(x) creates a new list and does not change the original list x.
# x.sort() sorts the list in-place.

x = [11,18,8,13]
x.sort()
print(x)


# # Sort the Items in a List in Descending Order. In-Place.

# In[26]:


x = [11,18,8,13]
x.sort(reverse=True)
print(x)


# # Tuples.

# 1. Immutable. (Can't add/change).
# 2. Useful for fixed data.
# 3. Faster to access than lists.
# 4. A sequence type.

# # Constructors - Creating new Tuples.

# In[27]:


x = ()
y = (1, 2, 3)
z = 4, 5, 6
u = (9,) # Tuple of size 1.
v = 5, # Tuple of size 1.
print(x)
print(y)
print(z)
print(u)
print(v)


# # Tuples are Immutable, but items may be mutable!

# In[28]:


y = ([1, 2], 3)
# y.remove(3) Can't do this.

del(y[0][1]) # Delete the 2.
print(y)

# Note that concatenating tuples works though!
y = y + (4, 5, 6)
print(y)


# # Sets.

# 1. Store non duplicate items.
# 2. Very fast access compared to lists.
# 3. Mathematical set operations (e.g. union, intersect etc).
# 4. Sets are unordered.

# # Constructors. Creating new sets.

# In[29]:


x = {3, 5, 3, 5}
print(x)

y = set()
print(y)

list1 = [2, 3, 4]
z = set(list1)
print(z)


# # Set Operations.

# In[30]:


x = {3, 5, 7, 9}

x.add(11)
print(x)

x.remove(9)
print(x)

# Determine the length of a set.
print(len(x))

# Check membership in a set.
print(5 in x)
print(4 in x)

# Pop a random element from a set (remember that sets are unordered).
print(x.pop(), x)

# Delete all items from a set.
x.clear()
print(x)

# Delete the set x.
del(x)


# # Mathematical Set Operations.

# 1. Intersection (AND): set1 & set2
# 2. Union (OR): set1 | set2
# 3. Symmetric difference (XOR): set1 ^ set2
# 4. Difference (in set1 but not set2): set1 - set2
# 5. Subset (set1 is contained in set2): set1 <= set2
# 6. Superset (set1 contains set2): set1 >= set2

# In[31]:


s1 = {1, 2, 3, 4}
s2 = {0, 1, 3, 5}
print(s1 & s2)
print(s1 | s2)
print(s1 ^ s2)
print(s1 - s2)
print(s1 <= s2)
print(s1 >= s2)


# # Dictionaries (dict).

# 1. Key/Value pairs.
# 2. Associative array (like Java HashMap).
# 3. Dictionaries are unordered.

# # Constructors of dictionaries.

# In[32]:


x = {'pork':25.4, 'beef': 33.7, 'chicken':22.1}
print(x)

y = dict([('pork', 25.4), ('beef', 33.7), ('chicken', 22.1)])
print(y)

z = dict(pork = 25.4, beef = 33.7, chicken = 22.1)
print(z)


# # dict operations.

# In[33]:


x = {'pork':25.4, 'beef': 33.7, 'chicken':22.1}

# Add or update an item.
x['shrimp'] = 38.2
print(x)

#Delete an item.
del(x['shrimp'])
print(x)

# Find the length of the dictionary.
print(len(x))

# Delete all items from dictionary x.
x.clear()
print(x)

# Delete dictionary x.
del(x)


# # Accessing keys and values in the dictionary.

# In[34]:


y = {'pork':25.4, 'beef': 33.7, 'chicken':22.1}
print(y.keys())
print(y.values())
print(y.items())

# Check membership in y_keys (only look in keys not values).
print('beef' in y)
print('beef' in y.keys()) # This also works the same as above.

# Check membership in y_values.
print('clams' in y.values())
print(22.1 in y.values())


# In[35]:


# Iterating a dictionary (note that items are in a random order).
for key in y:
    print(key, y[key])
    
for key, value in y.items():
    print(key, value)


# # Python List Comprehensions.

# Basic format: new_list = [transform sequence [filter]]

# In[36]:


import random


# In[37]:


# Get values within a range.
under_10 = [x for x in range(10)]
print('under_10: '+ str(under_10))


# In[39]:


# Get squared values.
squares = [i**2 for i in under_10]
print('squares: ' + str(squares))


# In[41]:


# Get odd numbers using mod (%).
odd_numbers = [i for i in under_10 if i%2 == 1]
print('odd_numbers: ' + str(odd_numbers))


# In[45]:


# Get multiples of 10.
multiple_10 = [10*x for x in range(10)]
print('multiple_10: ' + str(multiple_10))


# In[50]:


# Get all numbers from a string.
my_string = 'I love t0 go 2 the store 7 times a week.'
numbers_only = [n for n in my_string if n.isnumeric()]
print('numbers_only: ' + str(numbers_only)) # Prints the numbers in a  list.
print('numbers_only: ', ''.join(numbers_only)) # Prints the numbers as a string


# In[58]:


# Get indices of a list item.
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 6]
item = 5
indices = [ind for ind, val in enumerate(my_list) if val == item]
print('indices: ', indices)


# In[60]:


# Delete an item from a list.
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 6]
item = 5
my_list_del_item = [n for n in my_list if n != item]
print('my_list_del_item: ', my_list_del_item)

# String example.
letters = [x for x in 'ABCDEF']
random.shuffle(letters)
letrs = [a for a in letters if a != 'C']
print('letters: ', ''.join(letters))
print('letrs: ', ''.join(letrs))


# In[63]:


# if-else in a list comprehension.
# Must come before the iteration.

nums = [5, 3, 6 , 8 , 10, 7]
new_list = [x if x%2 == 0 else 10*x for x in nums] # If x is even return x, else if x is odd return 10*x.
print('new_list: ' + str(new_list))


# # Stacks, Queues and Heaps.

# # Stack using a Python List.

# 1. Stack is a last in first out data structure (LIFO).
# 2. Use append() to push and item on the stack.
# 3. Use pop() to remove an item from the stack.

# In[65]:


my_stack = list()
my_stack.append(8)
my_stack.append(12)
my_stack.append(4)
my_stack.append(13)
my_stack.append(11)
print(my_stack)


# In[66]:


print(my_stack.pop())
print(my_stack.pop())
print(my_stack)


# # Stack Using a List with a Wrapper Class.
# # (Additional Functionality Beyond That of a List).
# # Create a Stack Class and a Full Set of Class Methods.
# # (The Underlying Data Structure is Really a Python List).
# # For pop() and peek() Methods, Check Whether the
# # Stack is Empty to Avoid Exceptions.

# In[100]:


class Stack:
    def __init__(self):
        self.stack = list()
        
    def __str__(self):
        return(str(self.stack))
        
    def push(self, item):
        self.stack.append(item)
        
    def pop(self):
        if self.stack != []:
            item = self.stack.pop()
            return item
        else:
            return None
    
    def peek(self):
        if self.stack != []:
            item = self.stack[-1] # The last item.
            return item
        else:
            return None


# # Test Code for Stack Wrapper Class.

# In[102]:


st = Stack()
st.push(3)
st.push(4)
st.push(6)
print(st)
print('Pop():', st.pop())
print('Peek(): ', st.peek())
print(st)
print('Pop():',st.pop())
print('Peek():', st.peek())
print(st)
print('Pop():',st.pop())
print('Peek():', st.peek())
print(st)
print('Pop():',st.pop())
print('Peek():', st.peek())
print(st)
print('Peek(): ', st.peek())


# # Queues.

# # Queue Using a Python Deque.
# # Queue is a first in first out data structure (FIFO).
# # The Python Deque is a double ended queue, but can be used for the Queue.
# # Python append() method used to enqueue an item (add to the end of the Queue - the right hand end).
# # Python popleft() method used to dequeue an item (remove the front of the Queue - the left hand end).
# # (See Python documentation for deque).

# In[108]:


from collections import deque

my_queue = deque()
my_queue.append(9)
my_queue.append(5)
my_queue.append(10)
my_queue.append(2)
print(my_queue)
print(my_queue.popleft())
print(my_queue)
print(my_queue.popleft())
print(my_queue)


# # Queue Using a List With a Wrapper Class.

# In[127]:


class Queue:
    def __init__(self):
        self.queue = deque()
        
    def __str__(self):
        return(str(self.queue))
        
    def enqueue(self, item):
        self.queue.append(item)
        
    def dequeue(self):
        if self.queue != deque([]):
            item = self.queue.popleft()
            return item
        else:
            return None
    
    def getsize(self):
        return len(self.queue)


# # Test Code for Queue Wrapper Class.¶

# In[128]:


my_queue = Queue()
my_queue.enqueue(9)
my_queue.enqueue(5)
my_queue.enqueue(10)
my_queue.enqueue(2)
print(my_queue)
print('getsize: ', my_queue.getsize())
print('dequeue: ', my_queue.dequeue())
print(my_queue)
print('getsize: ', my_queue.getsize())
print('dequeue: ', my_queue.dequeue())
print(my_queue)
print('getsize: ', my_queue.getsize())
print('dequeue: ', my_queue.dequeue())
print('getsize: ', my_queue.getsize())
print('dequeue: ', my_queue.dequeue())
print('getsize: ', my_queue.getsize())
print('dequeue: ', my_queue.dequeue())
print('getsize: ', my_queue.getsize())


# # MaxHeap.

# 1. A complete binary tree.
# 2. Each node <= its parent.

# MaxHeap Operations:-
# 1. Push (insert).
# 2. Peek (get Max).
# 3. Pop (remove Max).

# # MaxHeap Using a Python List.

# A MaxHeap always bubbles the largest value to the top, so that it can be removed instantly.
# 
# Public functions: push, peek and pop.
# 
# Private functions: __swap, __floatUp, __bubbleDown and __str__.

# In[115]:


class MaxHeap:
    def __init__(self, items=[]):
        #super().__init__()
        self.heap = [0] # The 0th element of the list (not used).
        for item in items:
            self.heap.append(item)
            self.__floatUp(len(self.heap)-1)
            
    def push(self, data):
            self.heap.append(data)
            self.__floatUp(len(self.heap)-1)
            
    def peek(self):
        if self.heap[1]:
            return self.heap[1]
        else:
            return False
        
    def pop(self):
        if len(self.heap) >2:
            self.__swap(1, len(self.heap)-1) # Swap the 'first' and last list elements.
            maxval = self.heap.pop() # Pop off the maximum.
            self.__bubbleDown(1) # Bubble down the element at the top of the heap.
        elif len(self.heap) == 2:
            maxval = self.heap.pop() # Pop off the maximum.
        else:
            maxval = False
        return maxval
    
    def __swap(self, i, j):
        self.heap[i], self.heap[j] =  self.heap[j], self.heap[i]
        
    def __floatUp(self, index):
        parent = index//2
        if index <=1:
            return
        elif self.heap[index] > self.heap[parent]:
            self.__swap(index, parent)
            self.__floatUp(parent) # Recursive call.
            
    def __bubbleDown(self, index):
        left = index*2 # left child.
        right = left + 1 # Right child.
        largest = index
        if len(self.heap) > left and self.heap[largest] < self.heap[left]:
            largest = left
        if len(self.heap) > right and self.heap[largest] < self.heap[right]:
            largest = right
        if largest != index:
            self.__swap(index, largest)
            self.__bubbleDown(largest)
            
    def __str__(self):
        return str(self.heap)


# # MaxHeap Test Code.

# In[118]:


m = MaxHeap([24, 33, 41, 16])
print(m)

m.push(37)
print(m)
print(m.pop())
print(m)
print(m.peek())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Singly Linked List (My Attempt).

# Note that the data in each node can be quite general.

# In[4]:



# This Data type holds a number only.
class NumericData:
    def __init__(self, num = 0):
        self.number = num
        
    def setdata(self, num):
        self.number = num
        
    def getdata(self):
        tup = (self.number,)
        return(tup)
        
    def __str__(self):
        return('NumericData class:\n' + 'Data Numeric value = ' + str(self.number)+ '\n')


# This Data type holds a number, string and list.
class MultiTypeData:
    def __init__(self, num = 0, strng = '', lis = []):
        self.number = num
        self.string = strng
        self.list = lis
        
    def setdata(self, num, strng, lis):
        self.number = num
        self.string = strng
        self.list = lis
        
    def getdata(self):
        tup = (self.number, self.string, self.list)
        return(tup)
        
    def __str__(self):
        return('MultiTypeData class:\n' + 'Data Numeric value = '+ str(self.number) + 
               '\nData String value = ' + self.string +'\nData List value = ' + str(self.list) + '\n')
        
# The node holds a general data type and a pointer to the next node.
class Node:
    def __init__(self, d = None):
        self.data = d
        self.next = None
        
    def __str__(self):
        return('Node class:\n' + self.data.__str__())
        
        
# Note: Types such as list, dictionary, class, and objects, etc in Python behave like pointers.
# Note: Types such as int, str, float, bool, etc do not behave like a pointer.
class SinglyLinkedList:
    def __init__(self, r = None):
        #self.root = r  # CHECK THIS
        #self.this = r # CHECK THIS
        #self.next = None # CHECK THIS
        self.root = r
        
    # Add a new node to the head of the linked list.
    def addnode(self, n):
        n.next = self.root # Let the new node point to the root node.
        self.root = n # Update the root node. 
               
    def printSLL(self):
        this = self.root
        while this != None:
            print(this)
            this = this.next
     
    def find(self, d):
        this = self.root
        while this != None:   
            if this.data.getdata() == d.getdata():
                return(True, d)
            else:
                this = this.next
        return(False, None)


# # Test Code for the Singly Linked List.

# In[5]:


d = NumericData(7)
print(d)
#d = NumericData()
#d.setdata(7)
#print(d.getdata())
n = Node(d)
print(n)

d2 = MultiTypeData(34.6, 'Joey', [3, 1, 4, 2])
print(d2)
#d2 = MultiTypeData()
#d2.setdata(34.6, 'Joey', [3, 1, 4, 2])
#print(d2.getdata())
n2 = Node(d2)
print(n2)


# In[6]:


print('SinglyLinkedList with NumericData:\n')
sll = SinglyLinkedList()
sll.addnode(Node(NumericData(7)))
sll.addnode(Node(NumericData(9)))
sll.addnode(Node(NumericData(11)))
sll.addnode(Node(NumericData(13)))

sll.printSLL()

found, data = sll.find(NumericData(7))
print('found = ', found)
print('data:')
print(data)

found, data = sll.find(NumericData(555))
print('found = ', found)
print('data:')
print(data)

print('')
print('SinglyLinkedList with MultiTypeData:\n')
sll2 = SinglyLinkedList()
sll2.addnode(Node(MultiTypeData(34.6, 'Joey', [3, 1, 4, 2])))
sll2.addnode(Node(MultiTypeData(22.6, 'Billy', [2, 2, 2, 8])))
sll2.addnode(Node(MultiTypeData(1.5, 'Koko', [8, 8, 9, 9])))
sll2.addnode(Node(MultiTypeData(0.7, 'Philby', [7, 1])))

sll2.printSLL()

found, data = sll2.find(MultiTypeData(1.5, 'Koko', [8, 8, 9, 9]))
print('found = ', found)
print('data:')
print(data)

found, data = sll2.find(MultiTypeData(1.1, 'Atomic', [9, 8, 7, 6]))
print('found = ', found)
print('data:')
print(data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class Tree:
    def __init__(self):
        pass
    def addNode(self, val):
        pass

class Node:
    def __init__(self):
        self.value = 0.0
        self.parentNode = None
        self.leftNode = None
        self.rightNode = None
    def set value(self, val):
        self.value = val
        
        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[107]:


len([])


# In[126]:


x=deque()
x.append(4)
print(x)
#x.popleft()
print(x)
print(x == deque([]))


# In[ ]:





# In[ ]:





# In[56]:


4 != 5


# In[44]:


type(5.7)


# In[48]:


v = ''.join(numbers_only)
print(v)
print(type(v))


# In[ ]:




