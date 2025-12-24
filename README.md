# Blood Supply Chain Route Optimization

AI-based route optimization system using Genetic Algorithms for optimizing blood supply distribution in Malang Regency, Indonesia.

## Overview

This project implements a sophisticated route optimization system designed to minimize delivery time and reduce costs for blood supply chain logistics. The system uses genetic algorithms combined with OSRM (Open Source Routing Machine) for real-world route calculations.

## Features

- **Genetic Algorithm Optimization**: Advanced GA implementation using DEAP library
- **Real-world Routing**: Integration with OSRM for accurate route calculations
- **Smart Batching**: Intelligent delivery batching based on priority, time windows, and geographic clustering
- **Multi-objective Optimization**: 
  - Primary objective: Minimize delivery time (70% weight)
  - Secondary objective: Minimize cost (30% weight)
- **Geocoding Support**: Automatic address-to-coordinate conversion using OpenStreetMap
- **Visualization**: Comprehensive route and metrics visualization
- **Detailed Reporting**: Text-based optimization reports

## Project Structure

```
blood_supply_chain_route_opt/
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py         # Data models (Location, Delivery, Vehicle, etc.)
│   │   └── loader.py         # Data loading from Excel files
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── osrm_client.py    # OSRM routing integration
│   │   └── geocoding.py      # OpenStreetMap geocoding
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── genetic_algorithm.py  # GA implementation
│   │   └── batching.py       # Delivery batching strategies
│   └── utils/
│       ├── __init__.py
│       └── visualization.py  # Plotting and reporting
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
├── .gitignore
├── All Droping.xlsx          # Sample delivery data
├── Data PMI.xlsx             # PMI (Blood Bank) data
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/IritaSee/blood_supply_chain_route_opt.git
cd blood_supply_chain_route_opt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

#### 1. Basic Example

The simplest way to get started is to run the example script:

```bash
python example.py
```

This will:
- Create sample deliveries for 5 hospitals
- Optimize routes for 2 vehicles
- Generate visualizations and reports

#### 2. Full Application

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

### Understanding the Output

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

### Customization

#### Adjusting Vehicle Parameters

Edit `config/settings.py`:

```python
VEHICLE_CONFIG = {
    'capacity_liters': 100,        # Vehicle capacity
    'avg_speed_kmh': 40,           # Average speed
    'fuel_consumption_per_km': 0.12,  # Fuel efficiency
    'fuel_price_per_liter': 12750,    # Fuel price (IDR)
    'driver_cost_per_hour': 50000,    # Driver cost (IDR)
}
```

#### Tuning the Genetic Algorithm

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

#### Batching Strategy

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

## Configuration

### Default Vehicle Parameters (Malang Regency)

```python
VEHICLE_CONFIG = {
    'capacity_liters': 100,
    'avg_speed_kmh': 40,
    'fuel_consumption_per_km': 0.12,
    'fuel_price_per_liter': 12750,  # IDR
    'driver_cost_per_hour': 50000,  # IDR
    'vehicle_maintenance_cost_per_km': 500,  # IDR
}
```

### Optimization Parameters

```python
OPTIMIZATION_CONFIG = {
    'population_size': 100,
    'num_generations': 200,
    'crossover_probability': 0.8,
    'mutation_probability': 0.2,
    'time_weight': 0.7,  # Primary objective
    'cost_weight': 0.3,  # Secondary objective
}
```

## Data Format

### Input Excel Files

**All Droping.xlsx**: Contains delivery location data
- Columns should include location names and addresses
- System automatically identifies relevant columns

**Data PMI.xlsx**: Contains PMI (Blood Bank) operational data

### Adding Coordinates

The system can automatically geocode addresses using OpenStreetMap. To enable:

```python
from src.routing.geocoding import Geocoder

geocoder = Geocoder()
geocoder.geocode_locations_batch(locations, delay=1.0)
```

## Algorithm Details

### Genetic Algorithm

The GA implementation optimizes delivery routes by:
1. **Chromosome Representation**: Each individual represents a sequence of deliveries
2. **Fitness Function**: Weighted combination of delivery time and cost
3. **Selection**: Tournament selection
4. **Crossover**: Ordered crossover (OX) to maintain valid routes
5. **Mutation**: Shuffle mutation to explore new solutions
6. **Elitism**: Preserves best solutions across generations

### Batching Strategy

Three-level batching approach:
1. **Priority Grouping**: Separate emergency, urgent, and routine deliveries
2. **Time Windows**: Group deliveries within configurable time windows
3. **Geographic Clustering**: Further split large batches by location proximity

## Dependencies

Core libraries:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `deap` - Genetic algorithm framework
- `geopy` - Geocoding
- `requests` - OSRM API calls
- `matplotlib`, `seaborn` - Visualization
- `openpyxl` - Excel file reading

## Performance

Typical performance on standard hardware:
- 10 deliveries, 3 vehicles: ~30 seconds
- 50 deliveries, 5 vehicles: ~2-3 minutes
- 100 deliveries, 10 vehicles: ~5-10 minutes

### Performance Tips

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

## Limitations

- OSRM public server has rate limits; consider hosting your own for production
- Geocoding via Nominatim is limited to 1 request/second
- Large-scale optimizations (>100 deliveries) may require parameter tuning

## Future Enhancements

- [ ] Real-time traffic integration
- [ ] Dynamic route adjustment
- [ ] Multi-depot optimization
- [ ] Temperature monitoring integration
- [ ] Mobile app for drivers
- [ ] Dashboard for real-time monitoring
- [ ] Historical data analysis and forecasting
- [ ] Integration with hospital management systems

## Research Context

This system is designed for research in blood supply chain optimization for Malang Regency, East Java, Indonesia. It addresses:
- Time-sensitive blood product delivery
- Multiple delivery locations
- Vehicle capacity constraints
- Operating cost optimization
- Emergency response capabilities

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is developed for research purposes at Malang Regency, Indonesia.

## Acknowledgments

- OSRM Project for routing engine
- OpenStreetMap for geocoding services
- DEAP library for genetic algorithm framework
- Blood supply chain research community

## Contact

For questions or collaboration:
- GitHub Issues: [Create an issue](https://github.com/IritaSee/blood_supply_chain_route_opt/issues)

---

**Developed with ❤️ for improving blood supply chain logistics in Indonesia**
