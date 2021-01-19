#!/usr/bin/env python
# coding: utf-8

# # Think Python 2 code. 
# # David Brookes October 2020.

# In[20]:


# Please reference the book below for more details.
#
# Think Python
# How to Think Like a Computer Scientist
# 2nd Edition, Version 2.4.0
# Allen Downey
# Green Tea Press
# Needham, Massachusetts


# # Chapter 1
# # Programming

# In[1]:


# Printing.
print('Hello')


# In[2]:


# Simple mathematical operations.
print(1+1) 
print(10-2)
print(6*7)
print(6.0*7)
print(21/5)


# In[3]:


# print the type of a value.
print(type(2))
print(type(2.0))
print(type('Hello'))


# In[4]:


print(type('2'))
print(type('2.0'))


# # Chapter 2
# # Variables, expressions and statements

# In[5]:


# Variables.
x = 1
y = 67.2
z = 'Hello'
print(x)
print(y)
print(z)


# In[46]:


# Keywords (appear in green).
False
class
not 
return
# ...and so on.


# In[47]:


# Expressions.
x = 6
print(x+3)


# In[48]:


# Order of operations (PEDMAS).
x= 1+2**3
y = 2*3**2
z = (1+1)**(5-2)
print(x)
print(y)
print(z)


# In[49]:


x = 5+4/2
y=2*3-1
print(x)
print(y)
print(z)


# In[50]:


# String operations.
string1 = 'first'
string2 = 'second'

string3 = string1 + string2
string4 = string1*4
print(string3)
print(string4)


# In[52]:


x=y=1
print(x)
print(y)


# # Chapter 3
# # Functions

# In[53]:


print(int(2.333))
print(int(2.666))
print(int(-2.333))
print(int(-2.666))


# In[8]:


print(float(32))
print(float('3.142'))


# In[54]:


print(str(32))
print(str(3.142))


# In[55]:


import math

print(math.sqrt(2))
print(math.pi)
print(math.e)


# In[56]:


def myfunction():
    print('This is my function.')
    
myfunction()


# In[57]:


def myfunction(myargument):
    print('This is my function that I call ' + myargument + '.')
    
myfunction('Joey')


# In[58]:


import traceback

def twice(word):
    #traceback.print_stack()
    return(word*2)

def cat_twice(part1, part2):
    cat = part1 + part2
    print(twice(cat))

line1 = 'Bing tiddle '
line2 = 'tiddle bang. '

cat = 'Benny Hill'

cat_twice(line1, line2)
print(cat)


# In[59]:


# Fruitful function.
def my_square(myarg):
    return (myarg**2)

# Void function.
def print_my_square(myarg):
    print(myarg**2)
    
x = my_square(4.2)
print(x)
print(type(x))
y = print_my_square(3)
print(y)
print(type(y))


# In[60]:


def right_justify(word):
    column = 70
    word_length = len(word)
    new_word = ' '*(column-word_length) + word
    return(new_word)

print(right_justify('Kojak'))


# # Chapter 4
# # Case Study: Turtle Design.

# In[61]:


# Need more in this section!

#import turtle
#bob = turtle.Turtle()

# Iteration using a for loop.

for i in range(5):
    print(i)


# # Chapter 5
# # Conditionals and Recursion

# In[62]:


# Floor division and modulus.

minutes = 105
hours = minutes // 60
print(hours)

remainder = minutes % 60
print(remainder)


# In[18]:


# Boolean expressions. Relational operators.
x = 5
y = 6

print(x==y)
print(x!=y)
print(x>y)
print(x<y)
print(x>=y)
print(x<=y)


# In[19]:


# Logical operators.
x = 4
y = 7
print((x>0) and (y<10))
print((x>0) or (y<5))
print(not(x < y))

# Note nonzero numbers are interpreted as 'True' in Python.

print(42 and True)


# In[20]:


# Conditional execution.
x = 2
if x>0:
    print('x is positive')
    
# 'pass' is used as a place holder. There must be one statement in the body of the if statement.
x = -3
if x<0:
    pass # TODO: need to handle negative values!


# In[21]:


# Alternative execution.
x=4
if x>4:
    print('x>4')
else:
    print('x<=4')      


# In[22]:


# Chained conditionals.
x = 3
y = 3

if x < y:
    print('x is less than y')
elif x > y:
    print('x is greater than y')
else:
    print('x and y are equal')


# In[23]:


# Nested conditionals.

x = 3
y = 3

if x == y:
    print('x and y are equal')
else:
    if x < y:
        print('x is less than y')
    else:
        print('x is greater than y')


# In[24]:


# Coding shortcut.

x = 5

if 0 < x and x < 10:
    print('x is a positive single-digit number.')
    
if 0 < x < 10:
    print('x is a positive single-digit number.')


# In[25]:


# Recursion.

# Countdown example.
def countdown(n):
    if n <= 0:
        print('Blastoff!')
    else:
        print(n)
        countdown(n-1)

countdown(5)


# In[26]:


# Calculate 5+4+3+2+1.

def recurse(n, s):
    if n == 0:
        print(s)
    else:
        recurse(n-1, n+s)
        
recurse(5, 0)


# In[5]:


# Recursive function to reverse a list.

def reverse_list(p):
    length = len(p)
    if length == 0 or length == 1:
        return p
    else:
        head = p[:1] # This is a list object.
        tail = p[1:] # This is a list object.
        rev = reverse_list(tail)
        rev.extend(head)
        return rev        
    
my_list = [1, 2, 3, 4 , 5]
my_list_reversed = reverse_list(my_list)
print(my_list)
print(my_list_reversed)


# In[27]:


# Keyboard input.
name = input('What...is your name?\n')
    
print('Good day', name)


# # Chapter 6 
# # Fruitful functions.

# In[28]:



import math
x = 2
y = math.sqrt(2)
print(y)

# Alternatively the code below works, but it is considered bad practice.
# It is important to know which module the function belongs to.
from math import sqrt
x = 2
y = sqrt(2)
print(y)


# In[29]:



def area(radius):
    a = math.pi * radius**2
    return a

print(area(5))


# In[30]:


def absolute_value(x):
    if x < 0:
        return -x
    else:
        return x

print(absolute_value(-6))

def absolute_value_ERROR(x): # x = 0 not processed correctly.
    if x < 0:
        return -x
    elif x > 0:
        return x
    
print(absolute_value_ERROR(0))


# In[63]:


# Boolean functions.

def is_divisible(x, y):
    if x % y == 0:
        return True
    else:
        return False

is_divisible(6, 4)


# In[115]:


# Fibonacci numbers. Inefficient algorithm.

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(8))


# In[33]:


# Type checking.

def factorial(n):
    if not isinstance(n, int):
        print('Factorial is only defined for integers.')
        return None
    elif n < 0:
        print('Factorial is not defined for negative integers.')
        return None
    elif n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(5))
print(factorial(5.5))
print(factorial(-4))
print(factorial(-3.5))


# In[34]:


# Ackerman function.

def ack(m, n):
    if m == 0:
        return n+1
    elif m > 0 and n == 0:
        return ack(m-1, 1)
    elif m > 0 and n > 0:
        return ack(m-1, ack(m, n-1))
    else:
        print('Error ocurred!')
        
ack(3, 4)
    


# In[35]:


# Palindrome exercise.

def first(word):
    return word[0]

def last(word):
    return word[-1]

def middle(word):
    return word[1:-1]

def palindrome(word):
    if len(word) == 0 or len(word) == 1:
        return True
    else:
        if first(word) != last(word):
            return False
        else:
            return palindrome(middle(word))
    
myword = 'ebdbe'
print(palindrome(myword))


# # Chapter 7
# # Iteration.

# In[36]:


# Countdown example using 'while'.

def countdown(n):
    while n > 0:
        print(n)
        n = n - 1
    print('Blastoff!')
    
countdown(5)


# In[37]:


# This sequence may or may not terminate!
# See Collatz conjecture.

def sequence(n):
    while n != 1:
        #print(n)
        print(int(n))
        if n % 2 == 0: # n is even
            n = n / 2
        else: # n is odd
            n = n*3 + 1
        
sequence(3)


# In[38]:


# Using 'break' to get out of a loop.

while True:
    line = input('> ')
    if line == 'done':
        break
        print(line)
print('Done!')


# In[15]:


# Iterative function to reverse a list.

def reverse_list(p):
    length = len(p)
    rev =[]
    if length > 0:
        for index in range(length):
            rev.append(p[length - index - 1])
    return rev
            
my_list = ['fish', 'and', 'chips'] 
print(my_list)

my_list_reversed = reverse_list(my_list)
print(my_list_reversed)


# # Chapter 8
# # Strings.

# In[39]:


# A string is a sequence of characters.

# Positive indices.

fruit = 'banana'
print(fruit[0])
print(fruit[1])
print(fruit[2])
# ...
print(fruit[5])
print(len('banana'))


# In[40]:


# Negative indices.

fruit = 'banana'
print(fruit[-1])
print(fruit[-2])
print(fruit[-3])
print(fruit[-4])
print(fruit[-5])
print(fruit[-6])


# In[41]:


fruit = 'banana'
length = len('banana')

index = 0
while index < len(fruit):
    letter = fruit[index]
    print(letter)
    index = index + 1


# In[42]:


# More condensed code.

fruit = 'banana'
for letter in fruit:
    print(letter)


# In[43]:


# String slices.

s = 'Data Science'
print(s[0:7])  # Note: indices 0,1,2,3,4,5 & 6 - not 7!
print(s[2:10]) # Note: indices 2,3,4,5,6,7,8 & 9 - not 10!
print(s[0:len(s)])
print(s[:4])
print(s[:12])
print(s[:0])
print(s[:-1])
print(s[:-2])
print(s[:26])


# In[44]:


print(s[2:])
print(s[11:])
print(s[26:])
print(s[0:])
print(s[-1:])
print(s[-2:])


# In[1]:


# Strings are immutable!

greeting = 'Hello, world!'

try:
    greeting[0] = 'J' # ERROR!
except:
    print('Something went wrong!')
    
        


# In[64]:


# Need to make a copy of the string.

greeting = 'Hello, world!'
new_greeting = 'J' + greeting[1:]
new_greeting


# In[65]:


# Word search.

def find(word, letter):
    index = 0
    while index < len(word):
        if word[index] == letter:
            return index
        index = index + 1
    return -1

print(find('banana','a'))
print(find('banana','b'))
print(find('banana','n'))
print(find('banana','z'))


# In[66]:


# Looping and counting.

def count(word, search_letter):
    count = 0
    for letter in word:
        if letter == search_letter:
            count = count + 1
    print(count)
    
count('banana','a')
count('banana','b')
count('banana','n')
count('banana','z')      


# In[67]:


# String methods.
# Note: methods are like function, but with a different syntax.

word = 'banana'
new_word = word.upper()
new_word


# In[68]:


word = 'banana'
index = word.find('a')
index


# In[69]:


# The 'in' operator.

print('an' in 'banana')
print('anan' in 'banana')
print('baa' in 'banana')
print('seed' in 'banana')


# In[70]:


# String comparison.

word1 = 'apple'
word2 = 'banana'

if (word1 == word2):
    print('Words match.')
else:
    print('Words DON\'T match.') # Note the use of \.
        
if (word1 < word2):
    print(word1, 'comes before', word2)
else:
    print((word1, 'comes after', word2)) 


# In[71]:




def is_reverse(word1, word2):
    if len(word1) != len(word2):
        return False
    i = 0
    j = len(word2)-1
    while j >= 0:
        if word1[i] != word2[j]:
            return False
        i = i+1
        j = j-1
    return True

print(is_reverse('baaa','caab'))
print(is_reverse('ba','ab'))
print(is_reverse('ba','aba'))
print(is_reverse('a','a'))
print(is_reverse('',''))
print(is_reverse('',' '))


# # Chapter 9
# # Case Study: Word Play.

# In[72]:


# Load in a file containing words that are valid for crossword puzzles.
filename = 'D:\My Documents\Python Code\Think Python 2 Code\words.txt'

fin = open(filename)

line = fin.readline()
word = line.strip() # Strip the newline character.
print(word)
line = fin.readline()
word = line.strip() # Strip the newline character.
print(word)
# ...


# In[73]:


wordcount = 0
for line in fin: # The fin object appears to consist of line objects.
    word = line.strip()
    print(word)
    wordcount = wordcount + 1
    if wordcount == 5:
        break


# In[74]:


# Checking to see if a word is arranged alphabetically.
# Using a for loop.

def is_abecedarian(word):
    previous = word[0]
    for c in word:
        if c < previous:
            return False
        previous = c
    return True

print(is_abecedarian('been'))
print(is_abecedarian('sleep'))


# In[75]:


# Using recursion.

def is_abecedarian(word):
    if len(word) <= 1:
        return True
    if word[0] > word[1]:
        return False
    return is_abecedarian(word[1:])

print(is_abecedarian('been'))
print(is_abecedarian('sleep'))


# In[76]:


# Using a while loop.

def is_abecedarian(word):
    i = 0
    while i < len(word)-1:
        if word[i+1] < word[i]:
            return False
        i = i+1
    return True

print(is_abecedarian('been'))
print(is_abecedarian('sleep'))


# # Chapter 10
# # Lists.

# In[77]:


# A list is a sequence of values that can be of any type.

my_integers = [2, 4, 6, 8]
my_empty_list = []
my_strings = ['happy', 'go', 'lucky']
my_mixed_type =[42, 'word1', [45.0, 'word2']]

print(my_integers)
print(my_empty_list)
print(my_strings)
print(my_mixed_type)


# In[78]:


# Lists are mutable.

my_float = [22.4, 99.6]
my_float[0] = 6.9
print(my_float)


# In[79]:


# The 'in' operator.

cheeses = ['Cheddar', 'Edam', 'Gouda']

print('Edam' in cheeses)
print('Brie' in cheeses)

for cheese in cheeses:
    print(cheese)
    
my_numbers = [1, 2.3, 19]
for i in range(len(my_numbers)):
    my_numbers[i] = my_numbers[i]*2
    print(my_numbers[i])
    
for x in []:
    print('This never happens.')


# In[80]:


# List operators.

a = [1, 2, 3]
b = [4, 5, 6]
c = a + b
print(c)

d = [0]*4
e = [7,8,2]*4
print(d)
print(e)


# In[81]:


# List slices.

t = ['a', 'b', 'c', 'd', 'e', 'f']
print(t[1:3])
print(t[:4])
print(t[3:])
print(t[:])
print(t)


# In[82]:


t = ['a', 'b', 'c', 'd', 'e', 'f']
t[1:3] = ['x', 'y']
print(t)


# In[83]:


# List methods.

# Append an element to a list.
t = ['a', 'b', 'c']
t.append('d')
print(t)

# Extend a list by appending the elements of another list.
t1 = ['a', 'b', 'c']
t2 = ['d', 'e']
t1.extend(t2)
print(t1)

# Sort 'in order'.
t = ['d', 'c', 'e', 'b', 'a']
x = t.sort()
print(t)
print(x)

# Note: Most list methods are void and return 'None'.


# In[84]:


# Map, filter and reduce.

# Reduce.
def add_all(t):
    total = 0 # Accumulator.
    for x in t:
        total += x # Note: this is short for 'total = total + x'.
    return total

u = [1, 2, 3]
print(add_all(u))

# Can also use the Python function 'sum'.
my_sum = sum(u) # The elements of u are 'reduced' to a single value.
print(my_sum)


# In[85]:


# Map.
def capitalize_all(t):
    result = [] # Accumulator.
    for s in t:
        result.append(s.capitalize()) # capitalize() is a string method.
    return result

u = ['a', 'b', 'c', 'd', 'e', 'f']
capitalize_all(u)
# Note: capitalize_all() is a map.


# In[86]:


# Filter.
def only_upper(t):
    res = []
    for s in t:
        if s.isupper():
            res.append(s)
    return res
    
u = ['When', 'HARRY', 'met', 'SALLY','']
print(only_upper(u))
# Note: only_upper() is a filter.


# In[87]:


# Deleting elements.
t = ['a', 'b', 'c']
x = t.pop(1)
print(t)
print(x)

# If the removed value is not need then use the 'del' operator.
t = ['a', 'b', 'c']
del(t[2])
print(t)

# If the value to be removed is known, then use the 'remove' method.
t = ['a', 'b', 'c']
t.remove('b')
print(t)

t = ['a', 'b', 'c', 'd', 'e', 'f']
del t[1:5]
print(t)


# In[88]:


# Lists and strings.

s = 'hello'
t = ['h', 'e', 'l', 'l', 'o']
print(s == t)

# Split a string into letters.
u = list(s)
print(u)
print(t == u)


# In[89]:


# Split a string into words.
s = 'pining for the fjords'
t = s.split()
print(t)


# In[90]:


# Joining words to form a string.
t = ['pining', 'for', 'the', 'fjords']
delimiter = ' '
s = delimiter.join(t)
print(s)

delimiter = '' # Empty string.
s = delimiter.join(t)
print(s)


# In[91]:


# Objects and values.

string_a = 'banana'
string_b = 'banana'

string_a is string_b
# Note: a and b both refer to the same object.


# In[33]:


list_a = [1, 2, 3]
list_b = [1, 2, 3]

print(list_a is list_b)
# Note: a and b both refer to different object.
list_a[0] = 9
print(list_a)
print(list_b)


# In[1]:


# Aliasing.
list_a = [1, 2, 3]
list_b = list_a
print(list_b is list_a)

list_a[0] = 9 # Hence b[0] = 0 also.
print(list_a)
print(list_b)

# Note: Try to avoid aliasing with mutable objects!


# In[94]:


# List arguments.
def delete_head(t): # ''letters' and 't' are aliases.
    del t[0]

letters = ['a', 'b', 'c']
delete_head(letters)
letters


# In[95]:


# Operations on a list can; modify the list or create new lists.
# e.g. append() modifies a list.
t1 = [1, 2]
t2 = t1.append(3)
print(t1)
print(t2)

# The + operator creates a new list.
t3 = t1 + [4]
print(t1)
print(t3)

# Incorrect code here.
def bad_delete_head(t):
    t = t[1:] # WRONG!

t4 = [5, 6, 7]
bad_delete_head(t4)
print(t4)

# Note: The slice operator creates a new list and the assignment makes t refer to it, but that doesn’t
# affect the caller.

# Correct code here.
def tail(t):
    return t[1:]
# This function leaves the original list unmodified. Here’s how it is used:

letters = ['a', 'b', 'c']
rest = tail(letters)
print(rest)


# # Chapter 11
# # Dictionaries.

# In[96]:


# A dictinary is a mapping.

eng2sp = dict()
print(eng2sp) # {} is the empty dictionary.

eng2sp = {'one': 'uno', 'two': 'dos', 'three': 'tres'} # Note: 'key:value' pairs.
print(eng2sp)
print(eng2sp['two'])
# print(eng2sp['four']) # Error generated, 'four' not in dictionary.
print(len(eng2sp))
print('two' in eng2sp)
print('four' in eng2sp)


# In[97]:


# To see whether something appears as a value in a dictionary, you can use the method
# values, which returns a collection of values, and then use the in operator:

eng2sp = {'one': 'uno', 'two': 'dos', 'three': 'tres'}
vals = eng2sp.values()
print(vals)
print(type(vals))
print('uno' in vals)


# In[98]:


# Note: The in operator uses different algorithms for lists and dictionaries.
# For lists, as the list gets longer,the search time gets longer in direct proportion.
# Python dictionaries use a data structure called a hashtable that has a remarkable property:
# the in operator takes about the same amount of time no matter how many items are in the
# dictionary.


# In[99]:


def histogram(s):
    d = dict()
    for c in s:
        if c not in d:
            d[c] = 1
        else:
            d[c] += 1
    return d

count = histogram([6, 7, 7, 8, 7, 6, 0, 8])
print(count)
count = histogram('supercalifragilisticexplialidocious')
print(count)


# In[100]:


# The 'get' method: get takes a key and a default value.
# If the key appears in the dictionary, get returns the corresponding value; otherwise it returns the
# default value.

abc ={'a': 10, 'b': 20, 'c':30}

x = abc.get('a', 0)
y = abc.get('z', 0)

print(x)
print(y)


# In[4]:


# The histogram function may therefore be written more concisely.
def histogram(s):
    d = dict()
    for c in s:
        d[c] = d.get(c, 0)+1
    return d

count = histogram([6, 7, 7, 8, 7, 6, 0, 8])
print(count)
count = histogram('supercalifragilisticexplialidocious')
print(count)


# In[5]:


# Looping and dictionaries.
def print_hist(h):
    for c in h:
        print(c, h[c])

h = histogram('parrot')
print(h)
print_hist(h)

# Print out a sorted histogram.
sorted_h = sorted(h)
print(sorted_h)
for key in sorted(h):
    print(key, h[key])


# In[6]:


# Reverse lookup: Given a value v, what is key k? This is much slower than forward lookup.
def reverse_lookup(d, v):
    for k in d:
        if d[k] == v:
            return k
    raise LookupError('Value does not appear in the dictionary.') # Raise an exception.

h = histogram('parrot')
print(h)
key = reverse_lookup(h, 2)
print(key)
key = reverse_lookup(h, 17)
print(key)


# In[120]:


# Dictionaries and lists.

# Here is a function that inverts a dictionary:
def invert_dict(d):
    inverse = dict()
    for key in d:
        val = d[key]
        if val not in inverse:
            inverse[val] = [key]
        else:
            inverse[val].append(key)
    return inverse

hist = histogram('supercalifragilisticexplialidocious')
print(hist)

inverse = invert_dict(hist)
print(inverse)

# Note: Lists can be values in a dictionary, but they cannot be keys.
# Reason: keys must be hashable, and so must be immutable. (Lists are mutable).


# In[34]:


# A more efficient version of the function fibonacci() using a dictionary.

known = {0:0, 1:1}
def fibonacci(n):
    if n in known:
        return known[n]
    res = fibonacci(n-1) + fibonacci(n-2)
    known[n] = res
    return res

print(fibonacci(1001))


# In[2]:


# Global variables.

global_count = 0

def increment_WRONG():
    global_count = global_count + 1 # A new local variable global_count is created here.
       
try:
    increment_WRONG()
    print(global_count)
except:
    print('This code does not change the global variable.')


# In[124]:


global_count = 0
   
def increment_RIGHT():
    global global_count # Use the global version of the variable, not a new locally created one.
    global_count = global_count + 1

increment_RIGHT()
print(global_count)


# In[130]:


# List global variables.

global_list = [1, 2, 3]
print(global_list)

def change_list1():
    global global_list
    global_list[0] = 6
    
# Shorter version. No need to declare a mutable object.
def change_list2():
    global_list[0] = -1
    
change_list1()
print(global_list)

change_list2()
print(global_list)


# # Chapter 12
# # Tuples.

# In[143]:


# Tuples are immutable.
t = 'a', 'b', 'c', 'd', 'e'
print(t)
print(type(t))

# Although it is not necessary, it is common to enclose tuples in parentheses:
t = ('a', 'b', 'c', 'd', 'e')
print(t)
print(type(t))

# To create a tuple with a single element, include a final comma:
t = 'a',
print(t)
print(type(t))

t = ('a',)
print(t)
print(type(t))


# In[147]:


# The tuple() function.
# Create the empty tuple.

t = tuple()
print(t)
print(type(t))

t = tuple('Essex') # String as argument.
print(t)
print(type(t))

t = tuple([1, 2, 3, 4]) # List as argument.
print(t)
print(type(t))

t = tuple(('a', 'b', 'c', 'd')) # Tuple as argument.
print(t)
print(type(t))


# In[148]:


# Creating a new tuple with the same name. 

t = ('a', 'b', 'c', 'd', 'e')
t = ('A',) + t[1:]
t


# In[153]:


# Tuple assignment.
a = ('a', 'b', 'c', 'd', 'e')
b = ('f', 'g', 'h', 'i', 'j')
# Efficient way to swap a and b.
a, b = b, a

print(a)
print(type(a))
print(b)
print(type(b))


# In[157]:


# Function returning a tuple (multiple values).

def my_divmod(x, y):
    return x//y, x%y

t = my_divmod(7, 3)
print(t)

f, g = my_divmod(7, 3)
print(f)
print(g)


# In[3]:


# Variable length argument tuples. Use '*'.

def printall(*args): # '*' gathers arguments into a tuple.
    print(args)

printall(1, 2.0, '3')


# In[2]:


def my_divmod(x, y):
    return x//y, x%y

t = (6, 4)

print(len(t))

my_divmod(*t) # '*' is a scatter operator.


# In[5]:


# Function max() can take a variable number of arguments

print(max(3,4,7,5))


# In[10]:


# Function sum_all() can take a variable number of arguments.
# Note that this is a recursive function.

def sum_all(*args):
    if len(args) == 1:
        return args[0]
    else:
        return args[0] + sum_all(*args[1:])

print(sum_all(6))
print(sum_all(6,7))
print(sum_all(6,7,10))
print(sum_all(6,7,10,15))


# In[2]:


# Alternatively, can use the Python function sum().

def sum_all2(*args):
    return sum(args)

print(sum_all2(100, 200, 300))

# Note the Python function sum() can accept a tuple as an argument.
t = (4, 5, 6)
print(sum(t))


# In[14]:


# Lists and tuples.

s = 'abc'
t = [0, 1, 2]
z = zip(s, t)
print(z) # A zip object is a kind of iterator.
print(type(z)) # zip objects cannot be indexed, e.g. z[0] gives an ERROR.

for pair in zip(s, t):
        print(pair)
        
zip_to_list = list(z) # A list of tuples.
print(zip_to_list)
print(zip_to_list[1])


# In[15]:


t = [('a', 0), ('b', 1), ('c', 2)]
for letter, number in t: # Uses tuple assignment.
    print(number, letter)


# In[1]:


# Traversing two sequences at the same time.

def has_match(t1, t2):
    for x, y in zip(t1, t2):
        if x == y:
            return True
    return False

x = (1,2,3,4)
y = (9,4,3,8)
print(has_match(x, y))

x = (1,2,3,4)
y = (9,10,11,12)
print(has_match(x, y))


# In[23]:


# The 'enumerate' function.

for index, element in enumerate('abc'):
    print(index, element)
    
print(enumerate)
print(type(enumerate))

for t in enumerate('abc'):
    print(t)


# In[28]:


# Dictionaries and tuples.

# Dictionary items() method.
d = {'a':0, 'b':1, 'c':2}
t = d.items() # 'dict_items' object produced.
print(t)

for key, value in d.items():
    print(key, value)
    
for tup in d.items():
    print(tup)


# In[31]:


# Lists of tuples can be converted to a dictionary.

t = [('a', 0), ('c', 2), ('b', 1)]
d = dict(t)
print(d)


# In[32]:


# Combining 'dict and 'zip' to create a dictionary.

d = dict(zip('abc', range(3)))
print(d)


# In[25]:


# Using tuples as keys in dictionaries.

directory = dict()

directory['Brookes', 'David'] = 1234
directory['Simon', 'Moses'] = 5678
directory['Kumari', 'Sushila'] = 9012
directory['Rao', 'Dilpa'] = 8888

for last, first in directory:
    print(first, last, directory[last,first])


# # Chapter 13
# # Case study: data structure selection.

# In[ ]:


# Need something in this section!


# # Chapter 14
# # Files.

# In[1]:


# Print current working directory (folder).
import os

curr_dir=os.getcwd()
print(curr_dir)

# Change working directory.
# Note: 'r' allows backslashes (and forward slashes) in the file path name.

curr_dir = os.chdir(r"D:\My Documents\Python Code\Think Python 2 Code")
curr_dir=os.getcwd()
print(curr_dir)


# In[2]:


# Reading and writing files.

# To write a file, to open it with mode 'w' as a second parameter:
f = open('myfile.txt', 'w')

# If the file already exists, opening it in write mode clears out the old data and starts fresh,
# so be careful! If the file doesn’t exist, a new one is created.

line1 = "This is the first line,\n"
f.write(line1) # argument must be a string.

line2 = "and this is the second line.\n"
f.write(line2)

# Close the file after writing to it.
f.close()


# In[3]:


f = open('myfile.txt', 'r')

line = f.readline()
word = line.strip() # Strip the newline character.
print(word)
line = f.readline()
word = line.strip() # Strip the newline character.
print(word)


# In[4]:


# Format operator %.

print('I have %d chickens' % 5) # Integer.

print('I have %d chickens, pi is %g, and I like eating %s.' % (5, 3.142, 'pasta')) 
# Integer, float, string.
# Note these must be combined into a tuple.


# In[7]:


# Filenames and paths.

import os

curr_dir=os.getcwd()
print(curr_dir)

# Change working directory.
# Note: 'r' allows backslashes (and forward slashes) in the file path name.

curr_dir = os.chdir(r"D:\My Documents\Python Code")
curr_dir=os.getcwd()

# Function to print out files and folders (directories) recursively.

def my_walk(dirname):
    for name in os.listdir(dirname):
        path = os.path.join(dirname, name)
        if os.path.isfile(path):
            print(path)
        else:
            my_walk(path)
            
my_walk(curr_dir)


# In[9]:


# Catching exceptions (errors).

try:
    fin = open('bad_file') # 'badfile' does not exist.
except:
    print('Something went wrong.')
    
# Note: this is not the best use of 'try ' and 'except'.


# In[11]:


# Databases.
# These are like dictionaries, but data is stored permanently (like a file).

import dbm
db = dbm.open('captions', 'c') # 'c' means create the database if it does not exist already.

db['key1.png'] = 'value1.'

db['key1.png'] # Note below output: 'b' means a byte object.


# In[12]:


db['key2.png'] = 'value2.'
db['key3.png'] = 'value3.'

for key in db:
    print(key, db[key])
    
db.close() # Close the file after use.


# In[14]:


# Pickling.
# A limitation of 'dbm' is that the keys and values have to be strings or bytes. If you try to use
# any other type, you get an error.

# pickle.dumps takes an object as a parameter and returns a string representation (dumps is
# short for “dump string”).

import pickle
t = [1, 2, 3]
pickle.dumps(t)


# In[17]:


# pickle.loads (“load string”) reconstitutes the object.

t1 = [1, 2, 3]
s = pickle.dumps(t1)
t2 = pickle.loads(s)
print(t2)
# Although the new object has the same value as the old, it is not (in general) the same object:
print(t1 == t2)
print(t1 is t2)


# In[13]:


# Pipes.
# Most operating systems provide a command-line interface, also known as a shell.
# Any program that you can launch from the shell can also be launched from Python using
# a pipe object, which represents a running program.
# For example, the Unix command ls -l normally displays the contents of the current directory in long
# format. You can launch ls with os.popen.
import os

cmd = 'ls -l'
fp = os.popen(cmd) # Return object behaves like and open file.
res = fp.read() # Reads the whole of the file.
print(res)


# In[1]:



import os

# Listing the contents in a directory using 'os'
cwd = os.getcwd()
print(cwd)
print(os.listdir(cwd))
curr_dir = os.chdir(r"D:\My Documents\Python Code\Think Python 2 Code")
curr_dir=os.getcwd()
print(curr_dir)

# I'm not sure about pipes!!! 
cmd = 'ls -l'
fp = os.popen(cmd) # Return object behaves like an open file.
text = fp.read()
print(text)
stat = fp.close()
print(stat) # 'None' means it has worked, otherwise it hasn't! 


# In[3]:


# Writing modules.
# Note: my_square.py includes a function called number_squared() in the 
# current working directory (folder). The code is shown directly below:
# 
#!/usr/bin/env python
# coding: utf-8
# My_square function Python file.
#
# def number_squared(x):
#    return (x*x)

import my_square

curr_dir = os.chdir(r"D:\My Documents\Python Code\Think Python 2 Code")
curr_dir=os.getcwd()
print(curr_dir)

print(my_square)
print(my_square.__name__)

y = my_square.number_squared(3)
print(y)


# # Chapter 15
# # Classes and Objects.

# In[1]:


# Programmer defined types (classes).
# Note: this section uses ideas from later chapters! (e.g the use of the method __init__()).

from math import sqrt 

class Point:
    """Represents a point"""
    def __init__(self, x=0, y=0): # x=0 and y=0 are default values.
        self.x = x
        self.y = y
    
class Rectangle:
    """Represents a rectangle.
    attributes:width, height, corner. """
    def __init__(self, width, height, corner):
        self.width = width
        self.height = height
        self.corner = corner
        
def print_point(p):
    print('(',p.x,',',p.y,')')


# In[3]:


blank1 = Point(3.124, 2.718)
# blank1 object created. An object is an instance of a class.
# blank1.x = 3.124
# blank1.y = 2.718

blank2 = Point(3, 2)
# blank2.x = 3
# blank2.y = 2

blank3 = Point()

print_point(blank1)
print_point(blank2)
print_point(blank3)


# In[4]:


def dist(p1,p2):
    return sqrt((p1.x - p2.x)**2+(p1.y - p2.y)**2)

distance = dist(blank1, blank2)
distance


# In[5]:


# Create a Rectangle object.

rect1 = Rectangle(44, 26, Point(100, 150))

def find_centre(rect): # Takes in a Rectangle object and returns a Point object.
    p = Point(rect.corner.x + 0.5*rect.width, rect.corner.y + 0.5*rect.height)
    return(p)

# Find the centre of the Rectangle object.
centre_of_rect = find_centre(rect1)
print_point(centre_of_rect)


# In[13]:


# Objects are mutable.

box = Rectangle(100, 200, Point(10,10))

print(box.width)
box.width = box.width + 50
print(box.width)

# Function to modify a Rectangle object.
def grow_rectangle(rect, dwidth, dheight):
    rect.width += dwidth
    rect.height += dheight
    
grow_rectangle(box, 25, 75)   
print(box.width, box.height)
grow_rectangle(box, 25, 25)   
print(box.width, box.height)

# Inside the function grow_rectangle(), 'rect' is an alias for 'box', so when the function
# modifies 'rect', box changes.


# In[16]:


# Copying.
# Aliasing can make a program difficult to read because changes in one place might have
# unexpected effects in another place.
# Copying an object is often an alternative to aliasing.

p1 = Point()
p1.x = 3.0
p1.y = 4.0

import copy
p2 = copy.copy(p1)
# p1 and p2 contain the same data, but they are not the same Point.

print_point(p1)
print_point(p2)

print(p1 is p2)
print(p1 == p2) # Surprising this is False! So watch out for this.


# In[18]:


# Using copy.copy to duplicate a Rectangle, you will find that it copies the Rectangle
# object but not the embedded Point.
# This operation is called a shallow copy because it copies the object and any references
# it contains, but not the embedded objects.

box2 = copy.copy(box)
print(box2 is box)
print(box2.corner is box.corner)


# In[19]:


# The copy module provides a method named deepcopy that copies not only the
# object but also the objects it refers to, and the objects they refer to, and so on.

box3 = copy.deepcopy(box)
print(box3 is box)
print(box3.corner is box.corner)

# box3 and box are completely separate objects.


# In[24]:


p = Point()
p.x = 3
p.y = 4

# p.z
# AttributeError: Point instance has no attribute 'z'.

# If you are not sure what type an object is, you can ask:
print(type(p))

# You can also use isinstance to check whether an object is an instance of a class:
print(isinstance(p, Point))
# If you are not sure whether an object has a particular attribute, you can use the built-in
# function hasattr:
print(hasattr(p, 'x'))
print(hasattr(p, 'z'))


# In[30]:


# Can use 'try' and 'except'.
p = Point()
p.x = 3
p.y = 4

try:
    value = p.x
except AttributeError:
    value  = 0
    
print(value )

try:
    value = p.z
except AttributeError:
    value  = 0
    
print(value )


# # Chapter 16
# # Classes and Functions

# In[4]:


# Time class.

class Time:
    """Represents the time of day.
    attributes: hour, minute, second
    """
# We can create a new Time object and assign attributes for hours, minutes, and seconds:
time = Time()
time.hour = 11
time.minute = 59
time.second = 30
    
def print_time(t):
    print(t.hour, ':', t.minute,':',t.second)
    
print_time(time)


# In[5]:


# Pure functions.

def add_time(t1, t2):
    sum = Time()
    sum.hour = t1.hour + t2.hour
    sum.minute = t1.minute + t2.minute
    sum.second = t1.second + t2.second
    return sum

# This is called a pure function because it does not modify any of the objects passed to it 
# as arguments.

start = Time()
start.hour = 9
start.minute = 45
start.second = 0

duration = Time()
duration.hour = 1
duration.minute = 35
duration.second = 0

done = add_time(start, duration)
print_time(done) # Note 80 minutes generated.


# In[6]:


# Improved version - but getting a bit long!

def add_time2(t1, t2):
    sum = Time()
    sum.hour = t1.hour + t2.hour
    sum.minute = t1.minute + t2.minute
    sum.second = t1.second + t2.second
    if sum.second >= 60:
        sum.second -= 60
        sum.minute += 1
    if sum.minute >= 60:
        sum.minute -= 60
        sum.hour += 1
    return sum

done = add_time2(start, duration)
print_time(done)


# In[8]:


# Modifiers.

def increment(time, seconds):
    time.second += seconds
    if time.second >= 60:
        time.second -= 60
        time.minute += 1
    if time.minute >= 60:
        time.minute -= 60
        time.hour += 1

print_time(time)
increment(time, 25) # Note that 'time' is modified.
print_time(time)

# In general, it is recommended that you write pure functions whenever it is reasonable and resort
# to modifiers only if there is a compelling advantage. This approach might be called a
# functional programming style.


# # Chapter 17
# # Classes and Methods.

# In[ ]:


# Object-oriented features
# Python is an object-oriented programming language:
# • Programs include class and method definitions.
# • Most of the computation is expressed in terms of operations on objects.
# • Objects often represent things in the real world, and methods often correspond to the
# ways things in the real world interact.

# Methods.
# A method is a function that is associated with a particular class.
# Methods are semantically the same as functions, but there are two syntactic differences:
# • Methods are defined inside a class definition in order to make the relationship between the class 
# and the method explicit.
# • The syntax for invoking a method is different from the syntax for calling a function.


# In[3]:


# Printing objects.

class Time:
    def print_time(self): # Note that print_time() is a method.
                          # By convention, the first parameter of a method is called 'self'.
        print('%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second))
        
start = Time()
start.hour = 9
start.minute = 45
start.second = 00

# Two ways to print the time.

Time.print_time(start)

start.print_time() # This is the more common way.


# In[20]:


# The __init__() method. This sets the parameters of the object.
# Also 'self' and 'other' used below.

import math

class my_point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def dist_from_origin(self):
        return (math.sqrt(self.x**2 + self.y**2))
    def greater_dist_from_origin(self, other):
        return (self.dist_from_origin() > other.dist_from_origin())
    def __str__(self): # See below.
        return '%.2g : %.2g' % (self.x, self.y)
    
point1 = my_point(3, 4)
print(point1.dist_from_origin())
point2 = my_point(2.1, 4.3)
print(point2.dist_from_origin())

print(point1.greater_dist_from_origin(point2))


# In[21]:


# The __str__() method. This returns a string representation of the object.
# This is useful for debugging.
print(point1.__str__())
print(point2.__str__())


# In[24]:


# Operator overloading.
# For example, __add__() method allows the '+' operator to be overloaded.

class my_point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def dist_from_origin(self):
        return (math.sqrt(self.x**2 + self.y**2))
    def greater_dist_from_origin(self, other):
        return (self.dist_from_origin() > other.dist_from_origin())
    def __str__(self): # See below.
        return '%.2g : %.2g' % (self.x, self.y)
    def __add__(self, other): # This is basically vector addition. 
        yet_another = my_point() # This creates a new point (it does not modify).
        yet_another.x = self.x + other.x
        yet_another.y = self.y + other.y
        return (yet_another)
    
point1 = my_point(5.5, 4.5)
point2 = my_point(2.3, 1.1)

point3 = point1.__add__(point2) # Can use the __add__() method...
point4 = point1 + point2 # ...but can use the '+' operator instead.

print(point3.__str__())
print(point4.__str__())


# In[6]:


# Type-based dispatch.
# The method __add_() is modified to allow for different types of object.

class my_point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def dist_from_origin(self):
        return (math.sqrt(self.x**2 + self.y**2))
    def greater_dist_from_origin(self, other):
        return (self.dist_from_origin() > other.dist_from_origin())
    def __str__(self): # See below.
        return '%.2g : %.2g' % (self.x, self.y)  
    def add_point_dx(self, delta_x): 
        new_point = my_point()
        new_point.x = self.x + delta_x
        new_point.y = self.y
        return(new_point)
    def __add__(self, other): # This is basically vector addition. 
        if isinstance(other, my_point): # If 'other' is a 'my_point' object...
            yet_another = my_point() # This creates a new point (it does not modify).
            yet_another.x = self.x + other.x
            yet_another.y = self.y + other.y
        else: # It is assumed that 'other' is a number.
            yet_another = my_point()
            yet_another = self.add_point_dx(other)
        return (yet_another)
    def __radd__(self, other): # Note: __radd__() stands for “right-side add”.
        return self.__add__(other)
    
point1 = my_point(5.5, 4.5)
point2 = my_point(2.0, 1.0)

point3 = point1 + point2 # Vector addition.
point4 = point1 + 10.0 # Adds 10.0 to the x coordinate of point1 to create point4.

print(point3.__str__())
print(point4.__str__())


# In[58]:


# Polymorphism.
# Functions that work with several types are called polymorphic. 
# Polymorphism can facilitate code reuse.

def histogram(s):
    d = dict()
    for c in s:
        if c not in d:
            d[c] = 1
        else:
            d[c] = d[c]+1
    return d

my_string = 'asparagus'
my_tuple = (3, 4, 5, 6, 5, 4)
my_list = ['s','u','s','p','e','n','s','e']

print(histogram(my_string))
print(histogram(my_tuple))
print(histogram(my_list))


# In[59]:


# The built-in function 'sum', which adds the elements of a
# sequence, works as long as the elements of the sequence support addition.

point1 = my_point(1, 2)
point2 = my_point(5, 6)
point3 = my_point(6, 10)

print(sum((point1, point2, point3)))


# In[61]:


# Another way to access attributes is the built-in function vars(), which takes an object and
# returns a dictionary that maps from attribute names (as strings) to their values.

point1 = my_point(9, 10)

print(vars(point1))

# This function is useful for purposes of debugging.
def print_attributes(obj):
    for attr in vars(obj):
        print(attr, getattr(obj, attr))
        
# print_attributes() traverses the dictionary and prints each attribute name and its corresponding value.
# The built-in function getattr() takes an object and an attribute name (as a string) and returns
# the attribute’s value.

point1 = my_point(16, 32)
print_attributes(point1)


# In[ ]:


# Important note on Interface and implementation.
# One of the goals of object-oriented design is to make software more maintainable, which
# means that you can keep the program working when other parts of the system change, and
# modify the program to meet new requirements.
# A design principle that helps achieve that goal is to keep interfaces separate from implementations. 
# For objects, that means that the methods a class provides should not depend
# on how the attributes are represented.


# # Chapter 18
# # Inheritance.

# In[ ]:


# Inheritance is the ability to define a new class that is a modified version of an existing class.


# In[1]:


# Examples using playing cards.
# Suits and ranks encoded as integers.
# Suits:   Clubs -> 0, Diamonds -> 1, Hearts -> 2, Spades -> 3.
# Ranks:   Ace -> 1, 2 ->2, 3 -> 3,..., 10 -> 10, Jack -> 11, Queen -> 12, King -> 13.

class Card:
    """Represents a standard playing card."""
    # suit_names and rank_names are class attributes.
    
    suit_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades'] 
    rank_names = [None, 'Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    
    def __init__(self, suit=0, rank=2): # Default card is the 2 of Clubs.
        self.suit = suit
        self.rank = rank
        
    def __str__(self):
        return '%s of %s' % (Card.rank_names[self.rank], Card.suit_names[self.suit])
    
    def __lt__(self, other): # The method __lt__() allows the '<' operator to be overloaded.
        # It is assumed that 'suits are more important than rank', hence the code below.
        # Check the suits.
        if self.suit < other.suit: return True
        if self.suit > other.suit: return False
        # Suits are the same... check ranks.
        return self.rank < other.rank

    


# In[2]:


# Printing cards.

card1 = Card(2, 11)
print(card1)


# In[4]:


# Comparing cards.
card1 = Card(2, 11)
print(card1)
card2 = Card(2, 12)
print(card1)
card3 = Card(1, 4)
print(card3)

print(card1 < card2)
print(card1 < card3)


# In[56]:


# Define the class Deck.

import random # Used for shuffling the deck.

class Deck:
    def __init__(self):
        self.cards = [] # Attribute 'cards' is created. Initially an empty list.
        for suit in range(4):
            for rank in range(1, 14):
                card = Card(suit, rank)
                self.cards.append(card)
                
    def __str__(self):
        res = []
        for card in self.cards:
            res.append(str(card)) # Create a list of strings.
        # return '\n'.join(res) # Return one string.
        return ' : '.join(res) # Return one string. (This takes up less space in the output!).
    
    def pop_card(self): # Remove the last card in the deck.
        return self.cards.pop()
        
    def add_card(self, card): # Add a card to the deck.
        self.cards.append(card)
        
    def shuffle(self):
        random.shuffle(self.cards)
        
    def sort(self): # Note that the __lt__() method from the class Card is being used here.
        self.cards.sort()

    def move_cards(self, hand, num):
        for i in range(num):
            hand.add_card(self.pop_card()) # Move acrd from the Deck to the Hand.


# In[39]:


# Print the deck.

deck = Deck()
print(deck)


# In[40]:


# Add, remove , shuffle and sort a deck of cards. See the class Deck's methods.

deck.pop_card() # Remove three cards from the end of the list.
deck.pop_card()
deck.pop_card()

card1 = Card(3, 12) # Put the Queen of spades back in the deck.
deck.add_card(card1)

print(deck)


# In[41]:


# Shuffle.

deck.shuffle()
print(deck)


# In[42]:


deck.sort()
print(deck)


# In[44]:


# Inheritance.
# Inheritance is the ability to define a new class that is a modified version of an existing class.
# The relationship between classes — similar, but different — lends itself to inheritance.

# E.g. A Hand is like a Deck in some ways.

class Hand(Deck): # Hand inherits from Deck.
    """Represents a hand of playing cards."""

    def __init__(self, label=''): # Hand must overide Decks's version of __init__().
        self.cards = []
        self.label = label # Create an attribute label for the Hand.


# In[45]:


hand = Hand('new hand')
print(hand.cards)
print(hand.label)


# In[55]:


# Shuffle the Deck, pop 5 cards and add them to a Hand.
deck = Deck()
hand = Hand('new hand')
deck.shuffle()
for i in range(5):
    card = deck.pop_card()
    hand.add_card(card)
print(hand)


# In[67]:


# A natural next step is to encapsulate this code in a method called move_cards() inside class Deck.
# See class Deck above.
# 'def move_cards(self, hand, num):'
# Note that this function modifies Deck and Hand, and therefore returns 'None'.

deck = Deck()
hand = Hand('new hand')
deck.shuffle()

deck.move_cards(hand, 4) # Move 4 cards from the Deck to the Hand.
print(hand)


# In[ ]:


# Data Encapsulation.

# In object-oriented programming (OOP), encapsulation refers to the bundling of data with the methods
# that operate on that data, or the restricting of direct access to some of an object's components.


# # Chapter 19
# # The Goodies.

# In[ ]:


# These are useful features that are not really necessary — you can write good code without them —
# but with them you can sometimes write code that’s more concise, readable or efficient, and
# sometimes all three.


# In[2]:


# Conditional expressions.
x = 1 
if x > 0: 
    y = 100
else:
    y = -50
print(y)

# This can be written more concisely as...
x = 1
y = 100 if x > 0 else -50
print(y)


# In[23]:


# Recursive factorial function in conditional expression format.

def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)

print(factorial(6))


# In[35]:


# List comprehensions.

# Covert a list of strings to upper case.
def upper_all(t):
    res = []
    for s in t:
        res.append(s.upper())
    return res
names = ['David Brookes','Moses Simon','Sushila Kumari', 'Dilpa Rao']
print(names)
print(upper_all(names))


# In[36]:


# This can write this more concisely using a list comprehension:

def upper_all(t):
    return [s.upper() for s in t]
#The bracket operator indicates that a new list is being constructed.

names = ['David Brookes','Moses Simon','Sushila Kumari', 'Dilpa Rao']
print(names)
print(upper_all(names))


# In[41]:


# List comprehensions can also be used for filtering. For example, this function selects only
# the elements of t that are upper case, and returns a new list:

def only_upper(t):
    res = []
    for s in t:
        if s.isupper():
            res.append(s)
    return res

names = ['David Brookes','MOSES SIMON','Sushila KUMARI', 'DILPA RAO']
print(names)
print(only_upper(names))

# In list comprehension form.
def only_upper(t):
    return [s for s in t if s.isupper()]

# List comprehensions can be  harder to debug (you can’t put a print statement inside the loop)
# and therefore should only be used by experienced programmers.


# In[51]:


# Generator expressions.
# Generator expressions are similar to list comprehensions, but with parentheses instead of
# square brackets.

g = (x**2 for x in range(5))
print(g)

# Unlike a list comprehension, it does not compute the values all at once; it waits to be asked.
print('Using next().')
print(next(g))
print(next(g))
print(next(g))

# A for loop can be used to iterate through the values:
print('Using a for loop.')
for val in g:
    print(val)

# The generator object keeps track of where it is in the sequence, so the for loop picks up
# where next left off. 

# Once the generator is exhausted, it continues to raise StopIteration.
try:
    print(next(g))
except:
    print('Generator is exhausted.')


# In[52]:


# Generator expressions are often used with functions like sum, max, and min:
print(sum(x**2 for x in range(5)))


# In[53]:


# 'any()' and 'all()'.

# Python provides a built-in function, any(), that takes a sequence of boolean values and returns True
# if any of the values are True. It works on lists:

print(any([False, False, True]))

# Also it can be used with generator expressions:

print(any(letter == 'u' for letter in 'David'))
# Using any with a generator expression is efficient because it stops immediately if it finds a
# True value, so it doesn’t have to evaluate the whole sequence.


# In[56]:


# all().

print(all([True, True, False]))
print(any(letter == 'z' for letter in 'zzz'))


# In[63]:


# Sets.
# Python provides another built-in type, called a set, that behaves like a collection of dictionary
# keys with no values. Adding elements to a set is fast; so is checking membership.
# Sets provide methods and operators to compute common set operations.

my_list = [1,1,2,2,3,3,3]
my_set1 = set(my_list)
my_tuple = (1,1,2,2,3,3,3)
my_set2 = set(my_tuple)
my_string = 'abracadabra'
my_set3 = set(my_string)


print(my_list)
print(my_set1)
print(my_tuple)
print(my_set2)
print(my_string)
print(my_set3)

# Note that elements of a set are unique.


# In[70]:


# Sets are mutable. However, since they are unordered, indexing has no meaning.
# To make a set without any elements, we use the set() function without any argument.

my_set = set()
print(my_set)
my_set.add(2) # Add an element
print(my_set)
my_set.add(1)
print(my_set)
my_set.add(3)
print(my_set)
my_set.add(4)
print(my_set)
my_set.remove(4) # Removing an element. Also can use discard().
print(my_set)


# In[77]:


# Python Set Operations.

# Union.
# Union of A and B is a set of all elements from both sets.
A = {1,2,3,4}
B = {2,4,6,8}
union = A|B
print(union)

# Intersection.
# Intersection of A and B is a set of elements that are common in both the sets.
interscetion = A&B
print(interscetion)

# Difference.
# Difference of the set B from set A, (A - B), is a set of elements that are only in A but not in B.
difference = A-B
print(difference)

# Symmetric difference. Symmetric Difference of A and B is a set of elements 
# in A and B but not in both.
sym_diff = A^B
print(sym_diff)

# Set Membership Test. Use the 'in'operator.
print('a' in 'apple')
print('z' in 'apple')


# In[3]:


# Counters.
# A Counter is like a set, except that if an element appears more than once, the Counter
# keeps track of how many times it appears.
# 'Counter' is defined in a standard module called 'collections', so you have to import it.

from collections import Counter
count = Counter('parrot')
print(count)

#Counters behave like dictionaries in many ways; they map from each key to the number of
# times it appears.

print(count['p']) # 'p' occurs once.
print(count['r']) # 'r' occurs twice.
print(count['z']) # 'z' occurs zero times. (Note: No error is generated, as in a dictionary).


# In[8]:


# Example. Checking for anagrams.

def is_anagram(word1, word2):
    return Counter(word1) == Counter(word2)

print(is_anagram('conversation', 'conservation'))
print(is_anagram('section', 'notices'))
print(is_anagram('shower', 'whores'))
print(is_anagram('Moses', 'David'))

# Counters provide methods and operators to perform set-like operations, including addition,
# subtraction, union and intersection.


# In[15]:


# The method most_common() returns a list of value-frequency pairs, sorted from most common to least.

my_counter = Counter('abracadabra')
print(my_counter)
length = len(my_counter)
print(length, '\n')

for val, freq in my_counter.most_common(length):
    print(val, freq)


# In[16]:


#  defaultdict.
#The collections module also provides defaultdict, which is like a dictionary except that
# if you access a key that doesn’t exist, it can generate a new value on the fly.

from collections import defaultdict
d = defaultdict(list)

#Notice that the argument is list, which is a class object, not list(), which is a new list.
#The function you provide doesn’t get called unless you access a key that doesn’t exist.
t = d['new key']

print(t)

# The new list, which we’re calling t, is also added to the dictionary. So if we modify t, the
# change appears in d:
t.append('new value')
print(d)


# In[31]:


# Named tuples.
# Many simple objects are basically collections of related values. For example, the Point
# object defined in Chapter 15 contains two numbers, x and y. When you define a class like
# this, you usually start with an init method and a str method:

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return '(%g, %g)' % (self.x, self.y)
    
# This is a lot of code to convey a small amount of information. Python provides a more
# concise way to say the same thing:

from collections import namedtuple
Point_nt = namedtuple('Point_nt', ['x', 'y'])
# Point_nt automatically provides methods like __init__ and __str__.
print(Point_nt)

p = Point_nt(1, 2) # __init_() used.
print(p)  # __str_() used.
print(p.x)
print(p.y)


# In[25]:


# Gathering keyword args.
# In Chapter 12, we saw how to write a function that gathers its arguments into a tuple:

# Variable length argument tuples. Use '*'.
def printall(*args): # '*' gathers arguments into a tuple.
    print(args)

printall(1, 2.0, '3')
# You can call this function with any number of positional arguments (that is, arguments that
# don’t have keywords):

try:
    printall(1, 2.0, third='3')
except:
    print('This does not work with keywords')


# In[26]:


# To gather keyword arguments, you can use the ** operator:
def printall(*args, **kwargs):
    print(args, kwargs)
    
# You can call the keyword gathering parameter anything you want, but kwargs is a common
# choice. The result is a dictionary that maps keywords to values:

printall(1, 2.0, third='3')


# In[32]:


# If you have a dictionary of keywords and values, you can use the scatter operator, ** to
# call a function:
d = dict(x=1, y=2)
p = Point_nt(**d) # Class Point_nt used from above.
print(p)

# Without the scatter operator, the function would treat d as a single positional argument, so
# it would assign d to x and complain because there’s nothing to assign to y.


# In[ ]:


# When you are working with functions that have a large number of parameters, it is often
# useful to create and pass around dictionaries that specify frequently used options.

