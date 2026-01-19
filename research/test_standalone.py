"""
Standalone test to verify GA optimization works end-to-end.
"""

import sys
sys.path.insert(0, '/Users/iganarendra/blood_supply_chain_route_opt')

import logging
import numpy as np
from ga_optimizer.data_extractor import DataExtractor
from ga_optimizer.genetic_algorithm import GeneticAlgorithm

logging.basicConfig(level=logging.INFO)

print("\n=== TEST 1: Data Extraction ===")
extractor = DataExtractor()
summary = extractor.summarize_data()
print(f"Facilities: {summary['num_facilities']}")
print(f"Trips: {summary['num_trip_records']}")
print(f"Locations: {summary['num_unique_locations']}")

print("\n=== TEST 2: Mock Distance Matrix ===")
n_locs = min(summary['num_unique_locations'], 10)  # Use first 10 locations
# Create mock distance/duration matrices (haversine-like)
dist_matrix = np.random.uniform(5000, 50000, (n_locs, n_locs))
dur_matrix = dist_matrix / 40 * 3.6  # Convert to time at 40 km/h

# Make symmetric
for i in range(n_locs):
    for j in range(i+1, n_locs):
        dist_matrix[j, i] = dist_matrix[i, j]
        dur_matrix[j, i] = dur_matrix[i, j]
    dist_matrix[i, i] = 0
    dur_matrix[i, i] = 0

print(f"Matrix shape: {dist_matrix.shape}")

print("\n=== TEST 3: GA Optimization ===")
num_customers = n_locs - 1  # Exclude depot
ga = GeneticAlgorithm(
    num_customers=num_customers,
    num_vehicles=2,
    duration_matrix=dur_matrix,
    distance_matrix=dist_matrix,
    vehicle_capacity=100.0,
    population_size=20,
    generations=50,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elite_size=2,
)

print(f"Running GA: {num_customers} customers, 2 vehicles, 20 pop, 50 gen")
best_solution = ga.run()

print("\n=== TEST 4: Results ===")
results = ga.get_best_solution_details()
print(f"Best fitness: {results['fitness']:.2f}")
print(f"Makespan: {results['makespan_s']/3600:.2f} hours")
print(f"Total distance: {results['total_distance_km']:.1f} km")
print(f"Total cost: {results['total_cost_idr']:.0f} IDR")
print(f"Vehicle routes: {results['routes']}")

print("\n=== ALL TESTS PASSED ===")
