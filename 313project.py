from haversine import haversine, Unit
from collections import deque
import heapq
from math import radians, inf
import csv, json
import numpy as np
import pandas as pd

#INITIALIZATION/DATA READING===================================================================

# Delivery ID, Order Created at, Food Ready Time, Pickup Lat, Pickup Long, Dropoff Lat, Dropoff Long
deliverydata_file = open("./IEMS313_ProjectData.csv", mode='r', encoding='utf-8', errors='ignore')
deliverydata = list(csv.reader(deliverydata_file))[1:]

#Dasher ID, Dasher Lat, Dasher Long
location_file = open("./DasherLocations.csv", mode='r', encoding='utf-8', errors='ignore')
locationdata = list(csv.reader(location_file))[1:]



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
for location in locationdata:
    location[0] = int(location[0])
    location[1] = float(location[1])
    location[2] = float(location[2])

#PROBLEM ONE========================================================================


#finds nearest dasher in list of dashers to order pickup, and returns the dasher, time to travel, and list index of dasher
def findNearest(order:list, available: list):
    orderlat, orderlon = order[3], order[4]
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
    return [bestdasher, calcTime(mindist), index]

#takes distances in KM, which haversine returns by default, and returns number of minutes
def calcTime(distance: float):
    return ((distance*1000)/4.5)/60

def methodOne(deliveryinfo: list[list], dashers: list[list]):
    deliverydurations = []
    dasherdict = dict()
    dasherlist = sorted(dashers, key=lambda x: x[0])
    #sort delivery orders by time created 
    createddeliveryinfo = deque(sorted(deliveryinfo, key=lambda x: x[1]))

    #add extra field to track how many orders each dasher has
    for dasher in dashers:
        dasher.append(0)

    #assign orders to dasher
    while createddeliveryinfo:

        #get next order
        order = createddeliveryinfo.popleft()

        #get the nearest dasher that doesn't already have 4 deliveries
        dashers = [dasher for dasher in dashers if dasher[3] < 4]
        nearestdasher, traveltime, ind = findNearest(order, dashers)

        #update the dasher's number of deliveries
        dashers[ind][3] += 1

        #add it to the dasher's list of orders along with the amount of time it takes to get to pickup
        dasherdict.setdefault(nearestdasher[0], []).append([traveltime, order])
    
    for dasher in range(1,51):
        deliveries = deque(dasherdict[dasher])
        dasherinfo = dasherlist[dasher-1]
        while deliveries:
            currentorder = deliveries.popleft()


print(methodOne(deliverydata, locationdata))
# getUsableVals(deliverydata)
# print([delivery[1] for delivery in deliverydata])

print("done")