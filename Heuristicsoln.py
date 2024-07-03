from math import inf
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, copy, itertools, time
import gurobipy as gp
from gurobipy import GRB
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from haversine import haversine, Unit
import itertools
from multiprocessing import Pool, cpu_count
from collections import deque
#INITIALIZATION/DATA READING===================================================================

# Delivery ID, Order Created at, Food Ready Time, Pickup Lat, Pickup Long, Dropoff Lat, Dropoff Long
deliverydata_file = open("./DasherOrders.csv", mode='r', encoding='utf-8', errors='ignore')
deliverydata = list(csv.reader(deliverydata_file))[1:]

#Dasher ID, Dasher Lat, Dasher Long
dasher_file = open("./DasherLocations.csv", mode='r', encoding='utf-8', errors='ignore')
dasherdata = list(csv.reader(dasher_file))[1:]

# Load the CSV files, skipping the first row
delivery_data = pd.read_csv("./DasherOrders.csv")
dasher_locations = pd.read_csv("./DasherLocations.csv")

delivery_data = delivery_data.head(20)

# Convert datetimes
delivery_data['Order Created at'] = pd.to_datetime(delivery_data['Order Created at'])
delivery_data['Food Ready Time '] = pd.to_datetime(delivery_data['Food Ready Time '])

#converts time string like "17:38" to an int, 1058 minutes.
def getmins(timestr: str):
        hours = int(timestr[0:2])
        minutes = int(timestr[3:5])
        return hours*60+minutes


#converts all data values in both csvs to correct data types for our use
for delivery in deliverydata:
    delivery[0] = int(delivery[0])
    delivery[1] = getmins(delivery[1])
    delivery[2] = getmins(delivery[2])
    delivery[3] = float(delivery[3])
    delivery[4] = float(delivery[4])
    delivery[5] = float(delivery[5])
    delivery[6] = float(delivery[6])
#same thing
for person in dasherdata:
    person[0] = int(person[0])
    person[1] = float(person[1])
    person[2] = float(person[2])


# UTILITY FUNCTIONS
#finds nearest dasher in list of dashers to order pickup, and returns the dasher, the dasher's new availability time (time when next available), and list index of dasher
def findNearest(order:list, available: list):
    orderlat, orderlon = order[3], order[4]
    droplat, droplon = order[5], order[6]
    mindist = inf
    bestdasher = None
    index = 0
    iterations = -1
    for dasher in available:
        iterations += 1
        dasherlat, dasherlon = dasher[1], dasher[2]
        currentdist = haversine((orderlat, orderlon), (dasherlat, dasherlon))
        if currentdist <= mindist:
            mindist = currentdist
            bestdasher = dasher
            index = iterations
    if bestdasher == None:
        return False
    traveltosite = calcTime(mindist)
    pickuptime = order[2]
    
    ordertraveltime = calcTime(haversine((orderlat, orderlon),(droplat, droplon)))

    #if the dasher gets there before the food is ready, then the dasher will be next available after travel time + food ready time
    if bestdasher[4] + traveltosite < pickuptime:
        dasheravailabletime = pickuptime + ordertraveltime

    #if the dasher gets there at the same time or after, then the dasher will be next available when he get there + travel time
    else:
        dasheravailabletime = bestdasher[4] + traveltosite + ordertraveltime

    #return the best dasher's info, when they will be next available, their list index so we can modify their info, and the total time from order creation to delivery
    return [bestdasher, dasheravailabletime, index, dasheravailabletime - order[1], droplat, droplon]



#takes distances in KM, which haversine returns by default, and returns number of minutes
def calcTime(distance: float):
    return ((distance*1000)/4.5)/60

#checks the average time it takes to travel from pickup to dropoff, 155.602 minutes.
def findAverageDeliveryTime(ind):
    deliverytime = []
    for delivery in deliverydata:
        deliverytime.append([calcTime(haversine((delivery[3],delivery[4]),(delivery[5],delivery[6]))), delivery[0]])
    print(deliverytime[ind-1])
    # print([delivery[1] for delivery in deliverytime if delivery[0] <= 45])



#HEURISTIC METHODS ONE========================================================================


#This is a NAIVE solution. Optimal Solution towards the bottom.


#HEURISTIC METHODS ONE========================================================================
#assign orders to the nearest available dasher, base naive solution. 



def methodOne(deliveryinfolist: list[list], dasherslist: list[list]):

    dashers = copy.deepcopy(dasherslist)
    deliveryinfo = copy.deepcopy(deliveryinfolist)

    #initialize arrays to hold delivery durations and delivery dropoff times
    deliverydurations = []
    endingtimes = []

    #sort delivery orders by time created 
    createddeliveryinfo = deque(sorted(deliveryinfo, key=lambda x: x[1]))

    #add extra field to track how many orders each dasher has
    for dasher in dashers:

        #let index 3 represent the amount of orders they currently have
        dasher.append(0)

        #let index 4 represent the time at which they are done with all current orders
        dasher.append(0)

    #assign orders to dasher
    while createddeliveryinfo:

        #get next order and delete from sorted list of orders
        order = createddeliveryinfo.popleft()

        #get the nearest dasher that doesn't already have 4 deliveries
        dashers = [dasher for dasher in dashers if dasher[3] < 4]

        #get all available dashers at time of order creation
        availabledashers = [dasher for dasher in dashers if dasher[4] <= order[1]]

        #find the nearest dasher. If there are no available dashers, plug in all dashers. 
        if availabledashers:
            nearestdasher, nextavailable, ind, ordertraveltime, droplat, droplon = findNearest(order, availabledashers)
        else:
            nearestdasher, nextavailable, ind, ordertraveltime, droplat, droplon = findOptimalNaive(order, dashers)

        #track the order's delivery time
        deliverydurations.append(ordertraveltime)
        endingtimes.append(nextavailable)

        #update the dasher's number of deliveries, when they will be next available, and their new location after dropping off
        dashers[ind][1] = droplat
        dashers[ind][2] = droplon
        dashers[ind][3] += 1
        dashers[ind][4] = nextavailable
    
    #calculate time from first order creation to last delivery dropoff in minutes
    mindeliverytime = min([delivery[1] for delivery in deliverydata])
    maxdeliverytime = max(endingtimes)
    totalruntime = maxdeliverytime - mindeliverytime
    
    print("Average Delivery Duration:================")
    print(mean(deliverydurations))

    print("Average Deliveries per Hour:==============")
    print(len(deliverydata)/(totalruntime/60))

    print("Longest Delivery Duration:================")
    print(max(deliverydurations))





#HEURISTIC METHOD TWO:==============================================================
# same as method one, naive presolve by sorting order list AND supplying all 100 dashers to generate diagnostic data for delivery times using this method

def methodTwo(deliveryinfolist: list[list], dasherslist: list[list]):

    dashers = copy.deepcopy(dasherslist)
    deliveryinfo = copy.deepcopy(deliveryinfolist)
    
    #initialize arrays to hold delivery durations, deliveries made(by who), and delivery dropoff times
    deliverydurations = []
    endingtimes = []

    #sort delivery orders by time created 
    createddeliveryinfo = deque(sorted(deliveryinfo, key=lambda x: x[1]))

    #add extra field to track how many orders each dasher has
    for dasher in dashers:
        #let index 3 represent the amount of orders they currently have
        dasher.append(0)

        #let index 4 represent the time at which they are done with all current orders
        dasher.append(0)

    #assign orders to dasher
    while createddeliveryinfo:

        #get next order and delete from sorted list of orders
        order = createddeliveryinfo.popleft()

        #get all available dashers at time of order creation
        availabledashers = [dasher for dasher in dashers if dasher[4] <= order[1]]

        #find the nearest dasher. If there are no available dashers, plug in all dashers. 
        if availabledashers:
            nearestdasher, nextavailable, ind, ordertraveltime, droplat, droplon = findNearest(order, availabledashers)
        else:
            nearestdasher, nextavailable, ind, ordertraveltime, droplat, droplon = findOptimalNaive(order, dashers)

        #track the order's delivery time
        deliverydurations.append(ordertraveltime)
        endingtimes.append(nextavailable)

        #update the dasher's number of deliveries, when they will be next available, and their new location after dropping off
        dashers[ind][1] = droplat
        dashers[ind][2] = droplon
        dashers[ind][3] += 1
        dashers[ind][4] = nextavailable
    
    dasherIDs = [dasher[0] for dasher in dashers]
    dasherfrequencies = [dasher[3] for dasher in dashers]

    #calculate time from first order creation to last delivery dropoff in minutes
    mindeliverytime = min([delivery[1] for delivery in deliverydata])
    maxdeliverytime = max(endingtimes)
    totalruntime = maxdeliverytime - mindeliverytime
    
    print("Average Delivery Duration:================")
    print(mean(deliverydurations))

    print("Average Deliveries per Hour:==============")
    print(len(deliverydata)/(totalruntime/60))

    print("Longest Delivery Duration:================")
    print(max(deliverydurations))

    tick_label = [f"{dasherid}" for dasherid in dasherIDs]
    plt.bar(dasherIDs, dasherfrequencies, tick_label = tick_label,
        width = 0.8,)
    
    plt.xlabel('Dasher ID')
    plt.ylabel('# of Orders')
    plt.title('Distribution of Orders Across Dashers')

    plt.show()

    print(len(deliverydurations))


# print("METHOD ONE:\n")
# methodOne(deliverydata, dasherdata)
print("METHOD TWO:\n")
methodTwo(deliverydata, dasherdata)
print("done")