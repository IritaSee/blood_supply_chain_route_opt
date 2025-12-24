# Blood Supply Chain Route Optimization - Quick Start Guide

This guide demonstrates how to use the Blood Supply Chain Route Optimization system.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Example

The simplest way to get started is to run the example script:

```bash
python example.py
```

This will:
- Create sample deliveries for 5 hospitals
- Optimize routes for 2 vehicles
- Generate visualizations and reports

### 2. Full Application

To run the full application with real data:

```bash
python main.py
```

This will:
- Load data from Excel files (All Droping.xlsx, Data PMI.xlsx)
- Extract location information
- Create deliveries and vehicles
- Run batching and optimization
- Generate comprehensive reports

## Understanding the Output

After running, check the `output/` directory for:

1. **optimization_report.txt** - Detailed text report with:
   - Overall summary (distance, time, cost)
   - Route-by-route breakdown
   - Delivery sequences

2. **optimization_metrics.png** - Visual charts showing:
   - Distance per route
   - Time per route
   - Cost per route
   - Overall summary

3. **optimized_routes.png** - Map visualization of routes

## Customization

### Adjusting Vehicle Parameters

Edit `config/settings.py`:

```python
VEHICLE_CONFIG = {
    'capacity_liters': 100,        # Vehicle capacity
    'avg_speed_kmh': 40,           # Average speed
    'fuel_consumption_per_km': 0.12,  # Fuel efficiency
    'fuel_price_per_liter': 10000,    # Fuel price (IDR)
    'driver_cost_per_hour': 50000,    # Driver cost (IDR)
}
```

### Tuning the Genetic Algorithm

Adjust optimization parameters:

```python
OPTIMIZATION_CONFIG = {
    'population_size': 100,         # Number of individuals
    'num_generations': 200,         # Number of iterations
    'crossover_probability': 0.8,   # Crossover rate
    'mutation_probability': 0.2,    # Mutation rate
    'time_weight': 0.7,            # Time objective weight
    'cost_weight': 0.3,            # Cost objective weight
}
```

### Batching Strategy

Configure batching behavior:

```python
BATCHING_CONFIG = {
    'max_batch_size': 10,              # Max deliveries per batch
    'time_window_minutes': 120,        # Time window grouping
    'geographic_clustering': True,     # Enable geo clustering
    'priority_grouping': True,         # Group by priority
}
```

## Programmatic Usage

### Basic Usage in Python

```python
from src.data.models import Location, Delivery, Vehicle, PriorityLevel
from src.routing.osrm_client import OSRMRouter
from src.optimization.genetic_algorithm import RouteOptimizer

# Create locations
blood_bank = Location(
    id="BB_001",
    name="PMI Malang",
    address="Malang City",
    latitude=-8.1706,
    longitude=112.6314
)

# Create deliveries
deliveries = [...]  # Your deliveries

# Create vehicles
vehicles = [...]  # Your vehicles

# Initialize router
router = OSRMRouter()

# Run optimization
optimizer = RouteOptimizer(
    deliveries=deliveries,
    vehicles=vehicles,
    router=router
)

result = optimizer.optimize()

# Access results
print(f"Total Distance: {result.total_distance_km} km")
print(f"Total Time: {result.total_time_hours} hours")
print(f"Total Cost: IDR {result.total_cost_idr}")
```

### Loading Data from Excel

```python
from src.data.loader import DataLoader

# Initialize loader
loader = DataLoader(data_dir=".")

# Load Excel file
df = loader.load_all_droping_data("All Droping.xlsx")

# Extract locations
locations = loader.extract_locations_from_df(df)

# Create sample deliveries
deliveries = loader.create_sample_deliveries(num_deliveries=10)
vehicles = loader.create_sample_vehicles(num_vehicles=3)
```

### Using the Batching Strategy

```python
from src.optimization.batching import DeliveryBatcher

batcher = DeliveryBatcher()

# Create optimized batches
batches = batcher.create_optimized_batches(deliveries)

# Get batch statistics
stats = batcher.get_batch_statistics(batches)
print(f"Created {stats['num_batches']} batches")
```

### Geocoding Addresses

```python
from src.routing.geocoding import Geocoder

geocoder = Geocoder()

# Geocode a single location
location = Location(
    id="HOSP_001",
    name="Hospital A",
    address="Jl. Example St, Malang"
)

geocoder.geocode_location(location)

# Batch geocode multiple locations
geocoder.geocode_locations_batch(locations, delay=1.0)
```

### Visualization

```python
from src.utils.visualization import (
    plot_routes, 
    plot_optimization_metrics, 
    generate_report
)

# Plot route map
plot_routes(result.routes, save_path="my_routes.png")

# Plot metrics
plot_optimization_metrics(result, save_path="my_metrics.png")

# Generate text report
generate_report(result, output_path="my_report.txt")
```

## Performance Tips

1. **For large datasets (>50 deliveries)**:
   - Increase `num_generations` to 300-500
   - Consider increasing `population_size` to 150-200
   - Use batching to split into smaller problems

2. **For faster results**:
   - Reduce `num_generations` to 50-100
   - Use smaller `population_size` (50-75)

3. **For better solutions**:
   - Increase both population and generations
   - Run multiple times and pick best result
   - Adjust mutation rate (try 0.1-0.3)

## Troubleshooting

### OSRM Connection Issues

If OSRM server is unavailable, the system will automatically fall back to haversine distance estimation. To use a local OSRM server:

```python
router = OSRMRouter(server_url="http://localhost:5000")
```

### Geocoding Rate Limits

Nominatim has rate limits. If you have many locations to geocode:

```python
# Increase delay between requests
geocoder.geocode_locations_batch(locations, delay=2.0)

# Or use cached coordinates when available
```

### Memory Issues

For very large problems:
- Process in batches
- Reduce population size
- Use fewer generations

## Example Scenarios

### Scenario 1: Emergency Deliveries Only

```python
# Filter for emergency deliveries
emergency_deliveries = [
    d for d in deliveries 
    if d.priority == PriorityLevel.EMERGENCY
]

# Optimize with higher time weight
config = OPTIMIZATION_CONFIG.copy()
config['time_weight'] = 0.9
config['cost_weight'] = 0.1
```

### Scenario 2: Cost Optimization

```python
# Optimize primarily for cost
config = OPTIMIZATION_CONFIG.copy()
config['time_weight'] = 0.3
config['cost_weight'] = 0.7
```

### Scenario 3: Multiple Depots

```python
# Create separate optimizations per depot
results = []
for depot_location in depot_locations:
    depot_deliveries = [
        d for d in deliveries 
        if d.origin == depot_location
    ]
    result = optimize(depot_deliveries, vehicles)
    results.append(result)
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/IritaSee/blood_supply_chain_route_opt/issues
- Documentation: See README.md

## Next Steps

1. Customize configuration for your specific needs
2. Integrate with your data sources
3. Set up real-time geocoding
4. Deploy OSRM server for production use
5. Implement automated scheduling

---

**Happy Optimizing! 🚚💉**
