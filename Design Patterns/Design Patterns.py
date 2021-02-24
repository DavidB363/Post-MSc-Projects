#!/usr/bin/env python
# coding: utf-8

# # Design Patterns

# In[1]:


# A software design pattern is a general reusable solution
# to a commonly occuring problem.


# In[2]:


# Types of design patterns:
# 1. Creational.
# 2. Structural.
# 3. Behavioural.


# In[3]:


# Types Creational Design Patterns.
# (Creation of objects)
#
# Singleton.
# Factory.
# Builder.
# Prototype.


# In[4]:


# Types Structural Design Patterns.
# (Simplify the structure by identifying relationships between objects)
#
# Adaptor.
# Decorator.
# Facade.
# Proxy.


# In[5]:


# Types Behavioural Design Patterns.
# (How objects behave)
#
# Chain of responsibilty.
# Strategy.
# Observer.
# State.
# Template.
# Flyweight.


# In[6]:


# The rest of this notebook is based on the Youtube video by Ariel Ortiz:
# 'Design Patterns in Python for the Untrained Eye'.
# Material for tutorial at : bit.ly/2V819eq


# # Design Principles

# In[7]:


# 1. Separate out the things that change from those that stay the same.
# 2. Program to an interface, not an implementation.
# 3. Prefer composition over inheritance.
# 4. Delegation.


# In[8]:


# Anatomy of a design pattern.
#
# 1. Intent.
# 2. Motivation.
# 3. Structure.
# 4. Implementation.


# # Singleton Design Pattern

# In[9]:


# Singleton is a creation design pattern that ensures a class
# has only one instance, while providing a global access
# point to this instance.


# In[10]:


# Example: Tigger.

# The code in this cell could be saved in a module called tigger.py
class _Tigger: # Class Tigger is made private by a leading underscore character.

    def __str__(self):
        return "I'm the only one!"

    def roar(self):
        return 'Grrr!'
    
_instance = None # private module-scoped variable.

def Tigger():
    global _instance
    if _instance is None:
        _instance = _Tigger()
    return _instance
        


# In[11]:


# Could import Tigger from tigger.py using the command:
# from tigger import Tigger

a = Tigger() # Note that the function Tigger is being called, not class _Tigger.
b = Tigger()

print(f'ID(a) = {id(a)}') # Print the unique id of object a.
print(f'ID(b) = {id(b)}') # Print the unique id of object b.
print(f'Are they the same object? {a is b}')


# In[12]:


a


# In[13]:


b


# In[14]:


print(a)
print(a.roar())
print(b)
print(b.roar())


# # Template Method Design Pattern

# In[15]:


#  The template method is a behavioural design pattern that defines
# the skeleton of an algorithm in the base class, but lets derived
# classes override specific stepes of the algorithm without changing its
# structure.


# In[16]:


# Example: An average calculator.

from abc import ABC # ABC is a Python Abstract Base class.
from abc import abstractmethod # An abstractmethod is a method that 
                                # must be implemented in the derived classes.

class AverageCalculator(ABC): 

    def average(self): 
        try:
            num_items = 0
            total_sum = 0
            while self.has_next():
                total_sum += self.next_item()
                num_items += 1
            if num_items == 0:
                raise RuntimeError("Can't compute the average of zero items.")
            return total_sum / num_items
        finally:
            self.dispose()

    @abstractmethod
    def has_next(self): 
        pass

    @abstractmethod
    def next_item(self): 
        pass
    
    # Note that dispose() is not an abstractmethod.
    def dispose(self): 
        pass
    


# In[17]:


# Create a derived class FileAverageCalculator to calculate
# the average of the elements in a file.

class FileAverageCalculator(AverageCalculator):

    def __init__(self, file): 
        self.file = file
        self.last_line = self.file.readline() 

    def has_next(self):
        return self.last_line != '' 

    def next_item(self):
        result = float(self.last_line)
        self.last_line = self.file.readline() 
        return result

    def dispose(self):
        self.file.close()


# In[18]:


# A data.txt file exists with the following data:
# 4
# 8
# 15
# 16
# 23
# 42


# In[19]:



# Calculate the average of the elements of data.txt.
fac = FileAverageCalculator(open('data.txt'))
print(fac.average()) # Call the template method.


# In[20]:


# Create a derived class MemoryAverageCalculator to calculate
# the average of the elements in a list.

class MemoryAverageCalculator(AverageCalculator):
    
    def __init__(self, lst): 
        self.lst = lst
        self.index = 0

    def has_next(self):
        return self.index < len(self.lst) 

    def next_item(self):    
        result = float(self.lst[self.index])
        self.index += 1           
        return result


# In[21]:



mac = MemoryAverageCalculator([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
print(mac.average()) # Call the template method.


# # Adapter Design Pattern

# In[22]:


# Adapter is a structural design pattern that converts the interface of a class 
# into another interface that clients expect. 
# Adapter lets classes work together that couldn’t otherwise 
# because of incompatible interfaces.
# (Also known as a 'Wrapper'.)


# In[23]:


# Example: An average calculator that takes a generator as input.
# Although the Template Method could be used, an Adapter is
# produced.
# Note : reading the next line of a file uses readline()
#      : getting the next generated number uses next()
# Therefore files and generators share similarities, so that
# an Adapter can be used.


# In[24]:


# Example of a generator.

g = (2 ** i for i in range(10))

# Elements of g are produced on demand, not all at once!

print(next(g)) # Get first power of 2.
print(next(g)) # Get second power of 2.
print(next(g)) # Get third power of 2.
print(list(g)) # Get, as a list, the remaining seven powers of 2.


# In[25]:


class GeneratorAdapter:

    def __init__(self, adaptee): # adaptee is a generator object.
        self.adaptee = adaptee

    def readline(self):
        try:
            return next(self.adaptee) 
        except StopIteration:
            return '' # Returning an empty string emulates file behaviour.

    def close(self): 
        pass # There is no need to free memory resources when using generators.


# In[26]:


# Create a generator that generates one million random numbers between 1 and 100.
from random import randint

g = (randint(1, 100) for i in range(1000000)) 
fac = FileAverageCalculator(GeneratorAdapter(g)) # Use the file class with the adapted generator.
print(fac.average()) # Call the template method. # Use the file method.


# In[27]:


# Exercise. Poultry.


# In[28]:


class Duck:

    def quack(self):
      print('Quack')

    def fly(self):
        print("I'm flying")


class Turkey:

    def gobble(self):
        print('Gobble gobble')

    def fly(self):
        print("I'm flying a short distance")


# In[29]:


class TurkeyAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee
        
    def quack(self):
        self.adaptee.gobble()
            
    def fly(self):
        for i in range(5):
            self.adaptee.fly()
    


# In[30]:


def duck_interaction(duck):
    duck.quack()
    duck.fly()


duck = Duck()
turkey = Turkey()
turkey_adapter = TurkeyAdapter(turkey)

print('The Turkey says...')
turkey.gobble()
turkey.fly()

print('\nThe Duck says...')
duck_interaction(duck)

print('\nThe TurkeyAdapter says...')
duck_interaction(turkey_adapter)


# # Observer Design Pattern

# In[31]:


# Observer is a behavioural design pattern that defines a one-to-many
# dependency between objects so that when one object changes state, 
# all its dependents are notified and updated automatically.


# In[32]:


# The cases when certain objects need to be informed about the changes occured in other objects are frequent. 
# To have a good design means to decouple as much as possible and to reduce the dependencies. 
# The Observer design pattern can be used whenever a subject (a publisher object) has to be observed by one or 
# more observers (the subscriber objects).


# In[33]:


# First, we’ll define the Observer and Observable classes. 
# These classes provide us the support we require to implement
# the pattern in most typical cases:


# In[34]:


from abc import ABC, abstractmethod


class Observer(ABC):

    @abstractmethod
    def update(self, observable, *args):
        pass


class Observable:

    def __init__(self):
        self.__observers = []

    def add_observer(self, observer):
        self.__observers.append(observer)

    def delete_observer(self, observer):
        self.__observers.remove(observer)

    def notify_observers(self, *args):
        for observer in self.__observers:
            observer.update(self, *args)


# In[35]:


# The following code shows how to use these classes. 
# An Employee instance is an observable object (publisher). 
# Every time its salary is modified all its registered observer objects (subscribers) get notified. 
# We provide two concrete observer classes for our demo:

# Payroll: A class responsible for paying the salary to an employee.

# TaxMan: A class responsible for collecting taxes from the employee.


# In[36]:


class Employee(Observable): 

    def __init__(self, name, salary):
        super().__init__() 
        self._name = name
        self._salary = salary

    @property
    def name(self):
        return self._name

    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, new_salary):
        self._salary = new_salary
        self.notify_observers(new_salary) 


class Payroll(Observer): 

    def update(self, changed_employee, new_salary):
        print(f'Cut a new check for {changed_employee.name}! '
            f'Her/his salary is now {new_salary}!')


class TaxMan(Observer): 

    def update(self, changed_employee, new_salary):
        print(f'Send {changed_employee.name} a new tax bill!')


# In[37]:


e = Employee('Amy Fowler Fawcett', 50000)
p = Payroll()
t = TaxMan()

e.add_observer(p)
e.add_observer(t)

print('Update 1')
e.salary = 60000

e.delete_observer(t)

print('\nUpdate 2')
e.salary = 65000


# In[ ]:




