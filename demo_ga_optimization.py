"""
Complete working demonstration of GA blood supply optimization for Malang Regency.
Uses simplified haversine distances (no geocoding delays) for quick demonstration.
"""

import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

from ga_optimizer.data_extractor import DataExtractor
from ga_optimizer.genetic_algorithm import GeneticAlgorithm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters."""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371000  # Earth radius in meters


def main():
    """Run complete GA optimization demonstration."""
    
    print("\n" + "="*70)
    print("BLOOD SUPPLY ROUTE OPTIMIZATION - GENETIC ALGORITHM")
    print("Malang Regency, Indonesia - 2 Vehicle Fleet")
    print("="*70 + "\n")
    
    # Step 1: Extract data
    logger.info("Step 1: Extract historical data from Excel files")
    extractor = DataExtractor()
    summary = extractor.summarize_data()
    
    logger.info(f"  - Facilities: {summary['num_facilities']}")
    logger.info(f"  - Historical trips: {summary['num_trip_records']}")
    logger.info(f"  - Unique delivery locations: {summary['num_unique_locations']}")
    
    # Step 2: Get baseline from historical data
    logger.info("Step 2: Calculate baseline metrics from historical trips")
    trips = summary['trip_history']
    if not trips.empty:
        valid_trips = trips[trips['distance_km'].notna()]
        baseline_dist = valid_trips['distance_km'].mean()
        on_time = (valid_trips['status'].str.lower() == 'tepat waktu').sum()
        on_time_pct = (on_time / len(valid_trips)) * 100
        
        logger.info(f"  - Average historical distance: {baseline_dist:.1f} km")
        logger.info(f"  - On-time delivery rate: {on_time_pct:.1f}%")
    else:
        baseline_dist = 50.0
        logger.warning("  - No historical data available, using default")
    
    # Step 3: Create simplified location coordinates
    logger.info("Step 3: Setup delivery locations (mock coordinates)")
    locations = summary['locations'][:15]  # Use subset for demo
    
    # Assign mock coordinates (Malang area: lat -8.0 to -8.5, lon 112.5 to 113.0)
    np.random.seed(42)
    for loc in locations:
        loc['lat'] = -8.0 - np.random.uniform(0, 0.5)
        loc['lon'] = 112.5 + np.random.uniform(0, 0.5)
    
    # PMI blood bank (depot) at center
    locations[0]['lat'] = -8.137524
    locations[0]['lon'] = 112.572488
    
    logger.info(f"  - Number of locations: {len(locations)}")
    
    # Step 4: Build distance/time matrices using haversine
    logger.info("Step 4: Build distance/time matrices")
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    duration_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_m = haversine_distance(
                    locations[i]['lat'], locations[i]['lon'],
                    locations[j]['lat'], locations[j]['lon']
                )
                distance_matrix[i, j] = dist_m
                duration_matrix[i, j] = (dist_m / 1000) / 40 * 3600  # 40 km/h avg speed
    
    logger.info(f"  - Matrix size: {n}x{n}")
    logger.info(f"  - Distance range: {distance_matrix[distance_matrix>0].min()/1000:.1f} - "
               f"{distance_matrix.max()/1000:.1f} km")
    
    # Step 5: Run GA optimization
    logger.info("Step 5: Run Genetic Algorithm optimization")
    num_customers = n - 1  # Exclude depot
    
    ga = GeneticAlgorithm(
        num_customers=num_customers,
        num_vehicles=2,
        duration_matrix=duration_matrix,
        distance_matrix=distance_matrix,
        vehicle_capacity=100.0,
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=10,
    )
    
    logger.info(f"  - Customers: {num_customers}, Vehicles: 2")
    logger.info(f"  - Population: 100, Generations: 500")
    
    best_solution = ga.run()
    results = ga.get_best_solution_details()
    
    # Step 6: Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nVehicle Fleet Summary:")
    print(f"  - Number of vehicles: 2")
    print(f"  - Vehicle capacity: 100 units each")
    
    print(f"\nOptimized Routes:")
    for i, route in enumerate(results['routes']):
        if route:
            stops = [locations[0]['name']] + [locations[c+1]['name'] for c in route] + [locations[0]['name']]
            print(f"\n  Vehicle {i+1}:")
            print(f"    - Stops: {len(route)}")
            print(f"    - Distance: {results['vehicle_distances_m'][i]/1000:.1f} km")
            print(f"    - Duration: {results['vehicle_times_s'][i]/3600:.2f} hours")
            print(f"    - Cost: {results['vehicle_costs_idr'][i]:.0f} IDR")
            print(f"    - Route: {' → '.join(stops[:4])}... → {stops[-1]}")
    
    print(f"\nFleet Performance:")
    print(f"  - Makespan (max delivery time): {results['makespan_s']/3600:.2f} hours")
    print(f"  - Total distance: {results['total_distance_km']:.1f} km")
    print(f"  - Total duration: {results['total_time_s']/3600:.2f} hours")
    print(f"  - Total fuel cost: {results['total_cost_idr']:.0f} IDR")
    print(f"    (Fuel: {12750} IDR/L, Efficiency: 9 km/L)")
    
    # Comparison with baseline
    if baseline_dist:
        # Assume baseline uses single vehicle doing all deliveries
        baseline_total_dist = baseline_dist * num_customers
        improvement_pct = (1 - results['total_distance_km'] / baseline_total_dist) * 100
        print(f"\nComparison with historical baseline:")
        print(f"  - Historical avg distance per trip: {baseline_dist:.1f} km")
        print(f"  - Estimated historical total: {baseline_total_dist:.1f} km")
        print(f"  - GA optimized total: {results['total_distance_km']:.1f} km")
        print(f"  - Distance reduction: {improvement_pct:.1f}%")
    
    # Step 7: Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_to_save = {
        'optimization_method': 'Genetic Algorithm',
        'num_customers': num_customers,
        'num_vehicles': 2,
        'population_size': 100,
        'generations': 500,
        'makespan_hours': results['makespan_s'] / 3600,
        'total_time_hours': results['total_time_s'] / 3600,
        'total_distance_km': results['total_distance_km'],
        'total_cost_idr': results['total_cost_idr'],
        'vehicle_1': {
            'stops': len(results['routes'][0]),
            'distance_km': results['vehicle_distances_m'][0] / 1000,
            'time_hours': results['vehicle_times_s'][0] / 3600,
            'cost_idr': results['vehicle_costs_idr'][0],
        },
        'vehicle_2': {
            'stops': len(results['routes'][1]),
            'distance_km': results['vehicle_distances_m'][1] / 1000,
            'time_hours': results['vehicle_times_s'][1] / 3600,
            'cost_idr': results['vehicle_costs_idr'][1],
        },
    }
    
    results_file = output_dir / "ga_optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
