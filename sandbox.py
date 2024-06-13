import pandas as pd 
import numpy as np 
from haversine import haversine, Unit 
import gurobipy as gp
from gurobipy import GRB

# Load data 
deliveries = pd.read_csv('./IEMS313_ProjectData.csv') 
dashers = pd.read_csv('./DasherLocations.csv') 

# Subset to smaller sizes
deliveries = deliveries.head(4) 
dashers = dashers.head(2) 

# Convert time columns to datetime 
deliveries['Order Created at'] = pd.to_datetime(deliveries['Order Created at'], errors='coerce')
deliveries['Food Ready Time'] = pd.to_datetime(deliveries['Food Ready Time'], errors='coerce') 

# Define nodes 
delivery_nodes = list(range(len(deliveries))) 
dasher_nodes = list(range(len(deliveries), len(deliveries) + len(dashers))) 

# Calculate distance using Haversine formula 
def calculate_distance(lat1, long1, lat2, long2): 
    return haversine((lat1, long1), (lat2, long2), unit=Unit.METERS) 

# Add distance column to deliveries DataFrame 
deliveries['Distance'] = deliveries.apply(lambda row: calculate_distance(row['Pickup Latitude'], row['Pickup Longitude'], row['Dropoff Latitude'], row['Dropoff Longitude']), axis=1) 
# Convert distance (meters) to time (seconds) 
deliveries['Time_Seconds'] = deliveries['Distance'] / 4.5 

# Create distance and time matrices including dasher locations 
all_nodes = delivery_nodes + dasher_nodes 
distance_matrix = np.zeros((len(all_nodes), len(all_nodes))) 
time_matrix = np.zeros((len(all_nodes), len(all_nodes))) 

# Fill distance and time matrices 
for i, node1 in enumerate(all_nodes): 
    if node1 < len(delivery_nodes): 
        lat1 = deliveries.loc[node1, 'Pickup Latitude'] 
        long1 = deliveries.loc[node1, 'Pickup Longitude'] 
    else: 
        lat1 = dashers.loc[node1 - len(delivery_nodes), 'Dasher Lat'] 
        long1 = dashers.loc[node1 - len(delivery_nodes), 'Dasher Long'] 
    for j, node2 in enumerate(all_nodes): 
        if node2 < len(delivery_nodes): 
            lat2 = deliveries.loc[node2, 'Pickup Latitude'] 
            long2 = deliveries.loc[node2, 'Pickup Longitude'] 
        else: 
            lat2 = dashers.loc[node2 - len(delivery_nodes), 'Dasher Lat'] 
            long2 = dashers.loc[node2 - len(delivery_nodes), 'Dasher Long'] 
        distance_matrix[i, j] = calculate_distance(lat1, long1, lat2, long2) 
        time_matrix[i, j] = distance_matrix[i, j] / 4.5 

model = gp.Model("Delivery_Dispatch_Optimization")

# Define constants
S = len(dashers)
K = len(deliveries)
all_nodes_len = len(all_nodes)
delivery_nodes_len = len(delivery_nodes)

# Define variables
x = model.addVars(S, all_nodes_len, all_nodes_len, vtype=GRB.CONTINUOUS, name='x')
z = model.addVars(S, K, vtype=GRB.CONTINUOUS, name='z')

# Objective function: minimize the total delivery time in seconds
model.setObjective(gp.quicksum(time_matrix[i][j] * x[s, i, j] for s in range(S) for i in range(all_nodes_len) for j in range(all_nodes_len)), GRB.MINIMIZE)

# Constraints

# Each delivery is served exactly once
for k in range(K):
    model.addConstr(gp.quicksum(z[s, k] for s in range(S)) == 1)

# Each dasher starts at their home location and goes to one delivery location if assigned any deliveries
for s in range(S):
    dasher_node = delivery_nodes_len + s
    model.addConstr(gp.quicksum(x[s, dasher_node, j] for j in delivery_nodes) == gp.quicksum(z[s, k] for k in range(K)))

# Each dasher ends at their home location after visiting delivery locations if assigned any deliveries
for s in range(S):
    dasher_node = delivery_nodes_len + s
    model.addConstr(gp.quicksum(x[s, i, dasher_node] for i in delivery_nodes) == gp.quicksum(z[s, k] for k in range(K)))

# Flow conservation for delivery nodes
for s in range(S):
    for i in delivery_nodes:
        model.addConstr(gp.quicksum(x[s, i, j] for j in range(all_nodes_len) if j != i) - gp.quicksum(x[s, j, i] for j in range(all_nodes_len) if j != i) == 0)

# Link x and z variables
for s in range(S):
    for k in range(K):
        pickup_node = k
        model.addConstr(z[s, k] <= gp.quicksum(x[s, pickup_node, j] for j in range(all_nodes_len) if j != pickup_node))

# Ensure pickup time is after the food is ready
for s in range(S):
    for k in range(K):
        ready_time = deliveries.loc[k, 'Food Ready Time']
        order_time = deliveries.loc[k, 'Order Created at']
        travel_time_seconds = deliveries.loc[k, 'Time_Seconds']
        # Add wait time if dasher arrives before food is ready
        pickup_node = k
        model.addConstr(gp.quicksum(x[s, pickup_node, j] for j in range(all_nodes_len) if j != pickup_node) <= 1)

# Optimize the model
model.optimize()

# Print the status of the solution
print(f"Status: {model.Status}")

# Print the objective value
print(f"Objective value: {model.ObjVal}")