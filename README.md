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

### Basic Usage

Run the optimization with default settings:

```bash
python main.py
```

This will:
1. Load data from Excel files
2. Create sample deliveries and vehicles
3. Batch deliveries using the optimized batching strategy
4. Run genetic algorithm optimization
5. Generate reports and visualizations in the `output/` directory

### Output Files

After running, you'll find in the `output/` directory:
- `optimization_report.txt` - Detailed text report
- `optimization_metrics.png` - Visualization of distances, times, and costs
- `optimized_routes.png` - Map visualization of optimized routes

### Custom Configuration

Edit `config/settings.py` to customize:

**Vehicle Parameters:**
- Capacity (liters)
- Average speed
- Fuel consumption
- Operating costs

**Genetic Algorithm:**
- Population size
- Number of generations
- Crossover and mutation rates
- Objective weights

**Batching Strategy:**
- Maximum batch size
- Time window duration
- Geographic clustering options

## Configuration

### Default Vehicle Parameters (Malang Regency)

```python
VEHICLE_CONFIG = {
    'capacity_liters': 100,
    'avg_speed_kmh': 40,
    'fuel_consumption_per_km': 0.12,
    'fuel_price_per_liter': 10000,  # IDR
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
