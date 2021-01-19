#!/usr/bin/env python
# coding: utf-8

# 
# # Simulation of a Queue using Discrete Event Simulation.
# # (Work in progress!!  Needs comments and improvement in the structure of the program. Also analysis of generated data required e.g. average waiting times. Also graphics to be added possibly).
# # David Brookes December 2020.

# In[1]:


# Simulation of a queue.
# There is one teller who services one queue.


# In[2]:


import numpy as np
np.random.seed(1)

import random
random.seed(11)


# In[3]:


class State:
    def __init__(self, queue_len=0, teller_in_serv=False):
        self.queue_length = queue_len
        self.teller_in_service = teller_in_serv 
        # self.teller_break_time = teller_break_tim # The amount of time the teller is on a break from serving customers.
        self.waiting_time = [] # A list of waiting times for each person that has had to queue.
        
        
class Person:
    def __init__(self, start_wait = 0.0, end_wait = 0.0):
        self.start_waiting = start_wait
        self.end_waiting = end_wait

class Person_list:
    def __init__(self):
        self.person_list = []
        self.serve_index = 0 # The index of the next person to be served.
        
    def add_person(self, person):
        self.person_list.append(person)
        
#    def remove_person(self):
#        if self.person_list != []:
#            del self.person_list[0]

#    def print_list(self):
#        print('Printing the Person List...')
#        num_people = len(self.person_list)
#        for i in range(num_people):
#            print(self.person_list[i].start_waiting)
#            print(self.person_list[i].end_waiting)
#            print('')
            
    def print_list(self):
        print('Printing the Person List...')
        print('')
        for p in self.person_list:
            print(p.start_waiting)
            print(p.end_waiting)
            print('')
        if self.person_list == []:
            print('Person list is EMPTY') 
            print('')
    
    
class Event:
    """An event in the simulation."""
    # 'event_names' is a class attribute.
    event_names = ['Start', 'Customer_Arrival', 'Customer_Departure', 'Teller_Starts_Service', 'Teller_Ends_Service', 'End']
    
    def __init__(self, name, time):
        self.event_name = name
        self.event_time = time
        

# Set the parameters for the queuing model. (Experiment with different values!).

StartToEndTime = 100 # Time in seconds.

TellerStartsServiceTime_mu = 10 # Average time in seconds for a teller to start service (assuming normal distribution).
TellerStartsServiceTime_sd = 1  # The standard deviation of the normal distribution.

TellerServiceTime_mu = 30 # Average time in seconds for a teller to do a work session (assuming normal distribution).
TellerServiceTime_sd = 1   # The standard deviation of the normal distribution.

TellerBreakTime_mu = 10 # Average time in seconds for a teller to take a break (assuming normal distribution).
TellerBreakTime_sd = 1   # The standard deviation of the normal distribution.
        
CustomerArrivalTime_mu = 5 #  Average time in seconds for a customer arrival (assuming Poisson distribution).

CustomerServiceTime_mu = 4 #  Average time in seconds for a teller to serve a customer (assuming Poisson distribution).
    
class Events_list: 
    
    def __init__(self):
        self.event_list = []

#    def print_list(self):
#        print('Printing the Events List...')
#        num_events = len(self.event_list)
#        for i in range(num_events):
#            print(self.event_list[i].event_name)
#            print(self.event_list[i].event_time)
#            print('')
            
    def print_list(self):
        print('Printing the Events List...')
        print('')
        for e in self.event_list:
            print(e.event_name)
            print(e.event_time)
            print('')
        if self.event_list == []:
            print('Events list is EMPTY')  
            print('')

        
    def process_event(self):
        t0 = self.event_list[0].event_time
        e_name = self.event_list[0].event_name
        print('Processing (and deleting) the event -- ', e_name, '-- at time ', t0)
        
        if e_name == 'Start':
            
            # Create the first 'Customer_Arrival' event.
            t_customer_arrival = t0 + random.expovariate(1.0/CustomerArrivalTime_mu)
            e_customer_arrival = Event('Customer_Arrival', t_customer_arrival)
            self.insert_event(e_customer_arrival) 
            print('Schedule the event -- Customer_Arrival -- at time ', t_customer_arrival)
                 
            # Create a 'Teller_Starts_Service' event.
            t_teller_starts_service = t0 + np.random.normal(TellerStartsServiceTime_mu, TellerStartsServiceTime_sd)
            e_teller_starts_service = Event('Teller_Starts_Service', t_teller_starts_service)
            self.insert_event(e_teller_starts_service)
            print('Schedule the event -- Teller_Starts_Service -- at time ', t_teller_starts_service)
                  
            # Create an 'End' event.
            t_end = t0 + StartToEndTime
            e_end = Event('End', t_end)
            self.insert_event(e_end)
            print('Schedule the event -- End -- at time ', t_end)
                 
        elif  e_name == 'Customer_Arrival':
            # Increment the queue length.
            state.queue_length += 1
            print('Queue incremented to ', state.queue_length)

            # Create a Person and add the person to the Person list.            
            p = Person(t0) # Set start_waiting time to t0.
            
            print('Adding the new arrival to the person list...')
            p_list.add_person(p)
            p_list.print_list()
            
            # Create another 'Customer_Arrival' event.
            t_customer_arrival = t0 + random.expovariate(1.0/CustomerArrivalTime_mu)
            e_customer_arrival = Event('Customer_Arrival', t_customer_arrival)
            self.insert_event(e_customer_arrival) 
            print('Schedule the event -- Customer_Arrival -- at time ', t_customer_arrival)

            # Create a 'Customer_Departure' event if the Teller is serving and there is exactly one person in the queue.
            # (The departure time of a person joining an empty queue can be determined - statistically!).
            if state.teller_in_service == True and state.queue_length == 1:
                t_customer_departure = t0 + random.expovariate(1.0/CustomerServiceTime_mu)
                e_customer_departure = Event('Customer_Departure', t_customer_departure)
                self.insert_event(e_customer_departure) 
                print('Schedule the event -- Customer_Departure -- at time ', t_customer_departure)
            

            # PUT MORE CODE IN HERE WHEN state.teller_in_service == False???? No, I suspect!
                                 

        elif  e_name == 'Customer_Departure':
            if state.teller_in_service == True:
                if state.queue_length > 0:
                    print('state.teller_in_service', state.teller_in_service)
                    # Decrement the queue length.
                    state.queue_length -= 1
                    print('Queue decremented to ', state.queue_length)
                    # Process the next person from the Person list.

                    print ('For Person index ', p_list.serve_index ,'setting end_waiting time to ', t0)
                    p_list.person_list[p_list.serve_index].end_waiting = t0
                    #print('p_list.person_list[p_list.serve_index].start_waiting = ', p_list.person_list[p_list.serve_index].start_waiting)
                    #print('p_list.person_list[p_list.serve_index].end_waiting = ', p_list.person_list[p_list.serve_index].end_waiting)               
                    p_list.print_list()
                    # Increment the Serve_index (to service the next person in the queue).
                    p_list.serve_index += 1
                    # If there is still a queue of people then schedule the next departure event.
                    if state.queue_length > 0:
                        t_customer_departure = t0 + random.expovariate(1.0/CustomerServiceTime_mu)
                        e_customer_departure = Event('Customer_Departure', t_customer_departure)
                        self.insert_event(e_customer_departure) 
                        print('Schedule the event -- Customer_Departure -- at time ', t_customer_departure)
                
            # I DONT THINK THIS CODE BELOW IS NEEDED AT ALL!                   
            #else: # state.teller_in_service == False
                # The person will served by the teller at a delayed time determined by teller_break_time.
                # CHECK THIS!!!!!!
             #   print('state.teller_in_service', state.teller_in_service)
             #   e_customer_departure = Event('Customer_Departure', t0 + state.teller_break_time)
             #   self.insert_event(e_customer_departure)                
                
                

        
        elif  e_name == 'Teller_Starts_Service':
            state.teller_in_service = True   
            
            # Create a 'Customer_Departure' event if there is anyone in the queue.
            if state.queue_length > 0:
                t_customer_departure = t0 + random.expovariate(1.0/CustomerServiceTime_mu)
                e_customer_departure = Event('Customer_Departure', t_customer_departure)
                self.insert_event(e_customer_departure) 
                print('Schedule the event -- Customer_Departure -- at time ', t_customer_departure)
            
            # Create a 'Teller_Ends_Service' event.
            t_teller_ends_service = t0 + np.random.normal(TellerServiceTime_mu, TellerServiceTime_sd)
            e_teller_ends_service = Event('Teller_Ends_Service', t_teller_ends_service)
            self.insert_event(e_teller_ends_service)
            print('Schedule the event -- Teller_Ends_Service -- at time ', t_teller_ends_service)
        
        elif  e_name == 'Teller_Ends_Service':
            state.teller_in_service = False
            
            # Create a 'Teller_Starts_Service' event.
            t_teller_starts_service = t0 + np.random.normal(TellerBreakTime_mu, TellerBreakTime_sd)
            e_teller_starts_service = Event('Teller_Starts_Service', t_teller_starts_service)
            self.insert_event(e_teller_starts_service)  
            print('Schedule the event -- Teller_Starts_Service -- at time ', t_teller_starts_service)
                       
        if  e_name == 'End':    
            print('End of simulation')
            # Clear all the events list.
            self.event_list.clear() 
        else:
            # Delete the first event.
            # The event that has been processed is deleted from the events list in order to
            # to save computer memory. This is important for long simulations.
            del self.event_list[0]

    
    def insert_event(self, event):
        num_events = len(self.event_list)
        if num_events == 0:
            if event.event_time >= 0.0: # Want the time of the first event to be >=0.0 (start time will be set to 0.0).
                #print('appending an empty list')
                self.event_list.append(event)
        elif num_events == 1:
            if event.event_time >= self.event_list[0].event_time:
                #print('appending a list with one element in it')
                self.event_list.append(event)         
        else:
            before_first_event = False
            if event.event_time < self.event_list[0].event_time:
                before_first_event = True 
            #print('before_first_event', before_first_event)
            if not before_first_event:
                index = 1 # Start from 1 not 0. 
                inserted = False
                while index < num_events and inserted == False:
                    if event.event_time < self.event_list[index].event_time:
                        #print('insertion')
                        self.event_list.insert(index, event)
                        inserted = True
                    index = index + 1
                if inserted == False:
                    #print('appending')
                    self.event_list.append(event)    
                    
        
    def remove_event():
        pass


# In[4]:


# Initialise the State of the system.
state = State()
#print('Initial queue length = ', state.queue_length)
#print('teller_in_service = ', state.teller_in_service)
#print('')

# Create a Person list (All of the people involved during the length of the simulation).
p_list = Person_list()

# Create an Events list.
e_list = Events_list()

# Add the Start event to the Events list.
# Create the first Event.
e_start = Event('Start', 0.0)
e_list.insert_event(e_start)
e_list.print_list()

# print('Into the WHILE loop....')
# print('')

while e_list.event_list != []:
    print('')
    e_list.process_event()
    e_list.print_list()


# In[ ]:





# In[5]:


#my_list = ['a','b','c','d','e','f']
my_list = []

for letter in my_list:
    print(letter)
if my_list == []:
    print('EMPTY')


# In[ ]:




