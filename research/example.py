"""
Simple example demonstrating the blood supply chain route optimization
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.models import Location, BloodProduct, Delivery, Vehicle, PriorityLevel
from src.routing.osrm_client import OSRMRouter
from src.optimization.genetic_algorithm import RouteOptimizer
from src.optimization.batching import DeliveryBatcher
from src.utils.visualization import plot_routes, plot_optimization_metrics, generate_report
from datetime import datetime


def create_simple_example():
    """Create a simple example with 5 deliveries"""
    
    print("=" * 80)
    print("SIMPLE BLOOD SUPPLY CHAIN ROUTE OPTIMIZATION EXAMPLE")
    print("=" * 80)
    
    # Create locations (using Malang area coordinates)
    blood_bank = Location(
        id="BB_001",
        name="PMI Malang Blood Bank",
        address="Jl. Ade Irma Suryani, Malang",
        latitude=-8.1706,
        longitude=112.6314,
        location_type="blood_bank"
    )
    
    hospitals = [
        Location(
            id=f"HOSP_{i:03d}",
            name=f"Hospital {i+1}",
            address=f"Hospital Address {i+1}",
            latitude=-8.1706 + (i % 3 - 1) * 0.03,
            longitude=112.6314 + (i // 3 - 0.5) * 0.04,
            location_type="hospital"
        )
        for i in range(5)
    ]
    
    # Create blood products
    blood_types = ['A+', 'O+', 'B+', 'AB+', 'A-']
    deliveries = []
    
    for i, hospital in enumerate(hospitals):
        products = [
            BloodProduct(
                product_id=f"BP_{i:03d}",
                blood_type=blood_types[i],
                component="PRC",
                volume_ml=350.0,
                collection_date=datetime.now()
            )
        ]
        
        priority = [PriorityLevel.EMERGENCY, PriorityLevel.URGENT, PriorityLevel.ROUTINE][i % 3]
        
        delivery = Delivery(
            delivery_id=f"DEL_{i:03d}",
            origin=blood_bank,
            destination=hospital,
            products=products,
            priority=priority,
            requested_time=datetime.now()
        )
        deliveries.append(delivery)
    
    # Create vehicles
    vehicles = [
        Vehicle(
            vehicle_id=f"VEH_{i:03d}",
            capacity_liters=100.0,
            avg_speed_kmh=40.0,
            fuel_consumption_per_km=0.12
        )
        for i in range(2)
    ]
    
    print(f"\nCreated {len(deliveries)} deliveries and {len(vehicles)} vehicles")
    
    # Run batching
    print("\nBatching deliveries...")
    batcher = DeliveryBatcher()
    batches = batcher.create_optimized_batches(deliveries)
    print(f"Created {len(batches)} batches")
    
    # Initialize router
    print("\nInitializing OSRM router...")
    router = OSRMRouter()
    
    # Run optimization
    print("\nRunning genetic algorithm optimization...")
    optimizer = RouteOptimizer(
        deliveries=deliveries,
        vehicles=vehicles,
        router=router
    )
    
    result = optimizer.optimize(verbose=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Total Distance: {result.total_distance_km:.2f} km")
    print(f"Total Time: {result.total_time_hours:.2f} hours")
    print(f"Total Cost: IDR {result.total_cost_idr:,.2f}")
    print(f"Fitness Score: {result.fitness_score:.4f}")
    print(f"\nNumber of Routes: {len(result.routes)}")
    
    for i, route in enumerate(result.routes, 1):
        print(f"\nRoute {i} ({route.vehicle.vehicle_id}):")
        print(f"  Distance: {route.total_distance_km:.2f} km")
        print(f"  Time: {route.total_time_hours:.2f} hours")
        print(f"  Cost: IDR {route.total_cost_idr:,.2f}")
        print(f"  Deliveries: {len(route.deliveries)}")
        for j, delivery in enumerate(route.deliveries, 1):
            print(f"    {j}. {delivery.destination.name} (Priority: {delivery.priority.name})")
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    plot_optimization_metrics(result, save_path="output/example_metrics.png")
    plot_routes(result.routes, save_path="output/example_routes.png")
    generate_report(result, output_path="output/example_report.txt")
    
    print("\nResults saved to output/ directory")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    create_simple_example()
