from math import inf
from haversine import haversine, Unit
import csv
# Delivery ID, Order Created at, Food Ready Time, Pickup Lat, Pickup Long, Dropoff Lat, Dropoff Long
deliverydata_file = open("./IEMS313_ProjectData.csv", mode='r', encoding='utf-8', errors='ignore')
deliverydata = list(csv.reader(deliverydata_file))[1:]

#Dasher ID, Dasher Lat, Dasher Long
location_file = open("./DasherLocations.csv", mode='r', encoding='utf-8', errors='ignore')
locationdata = list(csv.reader(location_file))[1:]


#takes distances in KM, which haversine returns by default, and returns number of minutes
def calcTime(distance: float):
    return ((distance*1000)/4.5)/60

def findOptimalNaive(order: list, available: list):
    pickuptime = order[2]
    orderlat, orderlon = order[3], order[4]
    droplat, droplon = order[5], order[6]
    ordertraveltime = calcTime(haversine((orderlat, orderlon),(droplat, droplon)))
    
    minresult = inf
    optimalDasher = None
    index = 0
    iterations = -1
    deliverytime = 0

    for dasher in available:
        iterations += 1
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