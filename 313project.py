from haversine import haversine, Unit
from collections import deque
import heapq
from math import radians, inf
from statistics import mean, median
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


#This dont work yet its assigning the same dasher 200 times to be the optimal dasher for some reason
def findOptimalNaive(order: list, available: list):
    #demarcate food ready time
    pickuptime = order[2]

    #find the time it takes to travel from pickup to dropoff
    orderlat, orderlon = order[3], order[4]
    droplat, droplon = order[5], order[6]
    ordertraveltime = calcTime(haversine((orderlat, orderlon),(droplat, droplon)))
    
    #initialize max variables to update the optimal dasher & relevant info
    minresult = inf
    optimalDasher = None
    index = 0
    iterations = -1
    deliverytime = 0

    for dasher in available:
        iterations += 1

        #get where the dasher is right now
        dasherlat, dasherlon = dasher[1], dasher[2]
        dasherstarttime = dasher[4]
        traveltopickup = calcTime(haversine((orderlat, orderlon), (dasherlat, dasherlon)))  
        arrivaltime = dasherstarttime + traveltopickup
        if arrivaltime < pickuptime:
            finishtime = pickuptime + ordertraveltime
        else:
            finishtime = arrivaltime + ordertraveltime
        if finishtime < minresult:
            minresult = finishtime
            optimalDasher = dasher
            index = iterations
            deliverytime = finishtime - order[1]
    return [optimalDasher, minresult, index, deliverytime, droplat, droplon]


#takes distances in KM, which haversine returns by default, and returns number of minutes
def calcTime(distance: float):
    return ((distance*1000)/4.5)/60

def methodOne(deliveryinfo: list[list], dashers: list[list]):
    deliverydurations = []
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
        if not availabledashers:
            availabledashers = dashers
        nearestdasher, nextavailable, ind, ordertraveltime, droplat, droplon = findNearest(order, availabledashers)

        #track the order's delivery time
        deliverydurations.append(ordertraveltime)

        #update the dasher's number of deliveries, when they will be next available, and their new location after dropping off
        dashers[ind][1] = droplat
        dashers[ind][2] = droplon
        dashers[ind][3] += 1
        dashers[ind][4] = nextavailable
    print("mean:================")
    print(mean(deliverydurations))


# getUsableVals(deliverydata)
# print([delivery[1] for delivery in deliverydata])

print("done")