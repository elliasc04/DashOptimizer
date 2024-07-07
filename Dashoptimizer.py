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


# OPTIMIZED METHOD ONE: =========================================================================================


# This is a suboptimal solution that utilizes presolution via parallelized route generation and individualized, 
# iterative route evaluation to generate the minimum delivery time.


# OPTIMIZED METHOD ONE: =========================================================================================


# Generate all possible routes with 1, 2, 3, or 4 deliveries
# if the minimum delivery time is > 45 (arbitrary number), prune. <-- pruning rules
def generate_routes(delivery_data):
    routes = []
    for r in range(1, 5):
        routes.extend(itertools.permutations(delivery_data.index, r))
        print("Generated routes with ", r, " deliveries.")
    return routes

# Function to evaluate a single route
def evaluate_route(route):
    max_delivery_time = 0
    total_time = 0
    previous_dropoff_time = None

    # for generated routes, calculate total delivery time for each route.
    for i, delivery_id in enumerate(route):
        delivery = delivery_data.loc[delivery_id]
        pickup_time = delivery['Order Created at']
        if previous_dropoff_time:
            travel_duration = calcTime(
                haversine(
                (delivery_data.loc[route[i-1], 'Dropoff Latitude'], 
                delivery_data.loc[route[i-1], 'Dropoff Longitude']),
                (delivery['Pickup Latitude'],
                delivery['Pickup Longitude']))
            )
            pickup_time = max(pickup_time, previous_dropoff_time + pd.Timedelta(minutes=travel_duration))
        dropoff_time = pickup_time + pd.Timedelta(minutes=calcTime(haversine(
            (delivery['Pickup Latitude'], delivery['Pickup Longitude']),
            (delivery['Dropoff Latitude'], delivery['Dropoff Longitude']))
        ))
        delivery_duration = (dropoff_time - delivery['Food Ready Time ']).total_seconds() / 60
        max_delivery_time = max(max_delivery_time, delivery_duration)
        previous_dropoff_time = dropoff_time

    #if delivery time is greater than 45, prune. 
    if max_delivery_time <= 45:
        total_time = (previous_dropoff_time - delivery_data.loc[route[0], 'Food Ready Time ']).total_seconds() / 60
        efficiency = len(route) / total_time
        print(max_delivery_time)
        return (route, efficiency)
    return None

# Function to evaluate all routes in parallel
def parallel_evaluate_routes(routes):
    with Pool(cpu_count()) as pool:
        results = pool.map(evaluate_route, routes)
    return [result for result in results if result]


# Iterative optimization model
def optimize_routes_iterative(valid_routes, dashers, delivery_data):
    epsilon = 1e-6
    max_iterations = 100
    prev_avg_deliveries_per_hour = 0
    converged = False

    for _ in range(max_iterations):
        model = LpProblem(name="route-assignment", sense=LpMaximize)
        x = LpVariable.dicts("x", ((s, i) for s in range(len(dashers)) for i in range(len(valid_routes))), cat="Binary")
        total_time_hours = LpVariable("total_time_hours", lowBound=0, cat="Continuous")

        # Total deliveries
        total_deliveries = lpSum(x[s, i] * valid_routes[i][1] for s in range(len(dashers)) for i in range(len(valid_routes)))

        # Define total_time_hours
        model += total_time_hours == lpSum(x[s, i] * valid_routes[i][1] for s in range(len(dashers)) for i in range(len(valid_routes))) / 60.0  # Convert total_time from minutes to hours

        # Objective function: Maximize total deliveries
        model += total_deliveries

        # Constraints
        for i in range(len(valid_routes)):
            model += lpSum(x[s, i] for s in range(len(dashers))) == 1  # Each route must be assigned to exactly one dasher

        # Constraint to ensure each delivery is covered once
        for delivery_id in delivery_data.index:
            model += lpSum(x[s, i] for s in range(len(dashers)) for i in range(len(valid_routes)) if delivery_id in valid_routes[i][0]) <= 1

        # Fix total time if not the first iteration
        if prev_avg_deliveries_per_hour > 0:
            model += total_time_hours == fixed_total_time_hours

        # Solve the model
        model.solve()

        # Calculate the average deliveries per hour
        total_deliveries_value = value(total_deliveries)
        total_time_hours_value = value(total_time_hours)
        if total_time_hours_value == 0:
            avg_deliveries_per_hour = 0
        else:
            avg_deliveries_per_hour = total_deliveries_value / total_time_hours_value

        # Check for convergence
        if abs(avg_deliveries_per_hour - prev_avg_deliveries_per_hour) < epsilon:
            converged = True
            break

        # Update for next iteration
        prev_avg_deliveries_per_hour = avg_deliveries_per_hour
        fixed_total_time_hours = total_time_hours_value

    return model, x, prev_avg_deliveries_per_hour, converged, _

def OptimalSubTour():
    if __name__ == '__main__':
        start_time = time.time()  # Record the start time

        # Generate routes
        routes = generate_routes(delivery_data)

        # Evaluate routes in parallel
        filtered_routes = parallel_evaluate_routes(routes)
        print(len(filtered_routes))
        print(filtered_routes)
        # Optimize routes iteratively
        model, x, avg_deliveries_per_hour, converged, iterations = optimize_routes_iterative(filtered_routes, dasher_locations, delivery_data)

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the total execution time

        # Output results
        assigned_routes = [(s, i) for s in range(len(dasher_locations)) for i in range(len(filtered_routes)) if value(x[s, i]) == 1]
        # for s, i in assigned_routes:
        #     print(f"Dasher {s} is assigned to route {filtered_routes[i][0]}")
        print('Filtering and Optimizing Delivery Routes')
        print("Valid routes:")
        for i in range(len(filtered_routes)):
            print("Route:", filtered_routes[i][0], "Efficiency (total deliveries per hour):", (abs(filtered_routes[i][1]*60)))
        print("Number of valid routes:", len(filtered_routes),"\n")

        print('Path-wise Formulation')
        print("Number of routes assigned: ", len(assigned_routes),"\n")  # Display the number of assigned routes

        # Print the average deliveries per hour
        print(f"Average deliveries per hour per dasher: {avg_deliveries_per_hour/60}\n")
        
        # Comparing to the heuristic assignments in Questions 1 and 2
        print("Comparing to the heuristic assignments in Questions 1 and 2")
        print('The average deliveries per hour per dasher for the path-wise formulation in MILP approach 1 = 1, is higher than that for heuristic approach 1 = 0.376, and heuristic approach 2 = 0.384.')

# OPTIMIZED METHOD TWO: ==========================================================================================
# attempts to summarize entire problem into one MILP formulation. Dataframes make calculations easier, and 
# constraints are summarized in comments below.
def OptimalMILP(numDeliveries: int, numDashers: int):
    deliveries = deliverydata[0:numDeliveries]
    dashers = dasherdata[0:numDashers]

    delivery_coords = [(d[3], d[4]) for d in deliveries]
    dasher_coords = [(d[1], d[2]) for d in dashers]
    all_coords = delivery_coords + dasher_coords

    # Lengths
    total_len = len(all_coords)

    # Initialize matrices
    time2d = np.zeros((total_len, total_len))

    # Fill distance and time matrices
    for i, (lat1, long1) in enumerate(all_coords):
        for j, (lat2, long2) in enumerate(all_coords):
            distance = haversine((lat1, long1), (lat2, long2))
            time2d[i, j] = calcTime(distance)

    model = gp.Model("Delivery_Dispatch_Optimization")

    # Constant Definition
    Dashlen = len(dashers)
    Delivlen = len(deliveries)

    # Binary variable definitions
    y1 = model.addVars(Dashlen, Dashlen+Delivlen, Dashlen+Delivlen, vtype=GRB.BINARY, name='y1')
    y2 = model.addVars(Dashlen, Delivlen, vtype=GRB.BINARY, name='y2')

    # Objective: minimize total delivery time
    model.setObjective(gp.quicksum(time2d[i][j] * y1[dashind, i, j] for dashind in range(Dashlen) for i in range(Dashlen+Delivlen) for j in range(Dashlen+Delivlen)), GRB.MINIMIZE)

    for dashind in range(Dashlen):
        # Delivery assignment constraint: dashers must start at home, and go to delivery pickup
        model.addConstr(gp.quicksum(y1[dashind, Delivlen + dashind, j] for j in [i for i in range(Delivlen)]) == gp.quicksum(y2[dashind, delivind] for delivind in range(Delivlen)))

        # Return constraint: Dashers must return to point of origin after deliveries
        model.addConstr(gp.quicksum(y1[dashind, i, Delivlen + dashind] for i in [i for i in range(Delivlen)]) == gp.quicksum(y2[dashind, delivind] for delivind in range(Delivlen)))

        for i in [i for i in range(Delivlen)]:
            # Flow balancing constraint for delivery nodes; flow in - flow out = 0
            model.addConstr(gp.quicksum(y1[dashind, i, j] for j in range(Dashlen+Delivlen) if j != i) - gp.quicksum(y1[dashind, j, i] for j in range(Dashlen+Delivlen) if j != i) == 0)

        for delivind in range(Delivlen):
            # Pickup constraint: pickup times must be at or after food ready times
            model.addConstr(gp.quicksum(y1[dashind, delivind, j] for j in range(Dashlen+Delivlen) if j != delivind) <= 1)

            # Binary constraint making binary vars y1 and y2 dependents
            model.addConstr(y2[dashind, delivind] <= gp.quicksum(y1[dashind, delivind, j] for j in range(Dashlen+Delivlen) if j != delivind))
            
    # Every delivery only happens once
    for delivind in range(Delivlen):
        model.addConstr(gp.quicksum(y2[dashind, delivind] for dashind in range(Dashlen)) == 1)

    # Optimize the model
    model.optimize()

    print(f"Objective value: {model.ObjVal}")

# print("METHOD ONE:\n")
# OptimalSubTour()
print("METHOD TWO:\n")
OptimalMILP(100,10)
print("done")

