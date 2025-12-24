"""
Configuration file for Blood Supply Chain Route Optimization
"""

# OSRM Configuration (using public OSRM server by default)
OSRM_SERVER = "http://router.project-osrm.org"

# Vehicle Parameters (standard fleet assumptions for Malang Regency, Indonesia)
VEHICLE_CONFIG = {
    'capacity_liters': 100,  # Blood product capacity in liters
    'avg_speed_kmh': 40,  # Average speed in urban areas (km/h)
    'fuel_consumption_per_km': 0.12,  # Liters per km
    'fuel_price_per_liter': 10000,  # IDR (Indonesian Rupiah)
    'driver_cost_per_hour': 50000,  # IDR
    'vehicle_maintenance_cost_per_km': 500,  # IDR
}

# Optimization Parameters
OPTIMIZATION_CONFIG = {
    # Genetic Algorithm parameters
    'population_size': 100,
    'num_generations': 200,
    'crossover_probability': 0.8,
    'mutation_probability': 0.2,
    'tournament_size': 3,
    'elite_size': 5,  # Number of best individuals to preserve
    
    # Objective weights (sum should be 1.0)
    'time_weight': 0.7,  # Primary objective: minimize delivery time
    'cost_weight': 0.3,  # Secondary objective: minimize cost
    
    # Constraints
    'max_route_duration_hours': 8,  # Maximum working hours per route
    'service_time_minutes': 15,  # Time spent at each delivery location
}

# Geocoding Configuration
GEOCODING_CONFIG = {
    'use_cache': True,
    'cache_file': 'data/geocode_cache.json',
    'default_region': 'Malang Regency, East Java, Indonesia',
    'nominatim_user_agent': 'blood_supply_chain_optimizer',
}

# Blood Supply Chain Parameters
BLOOD_SUPPLY_CONFIG = {
    'max_transport_time_hours': 4,  # Maximum time blood can be in transport
    'temperature_range': (2, 8),  # Celsius, for cold chain
    'priority_levels': {
        'emergency': 1,
        'urgent': 2,
        'routine': 3,
    }
}

# Batching Strategy Parameters
BATCHING_CONFIG = {
    'max_batch_size': 10,  # Maximum number of deliveries per batch
    'time_window_minutes': 120,  # Group deliveries within this time window
    'geographic_clustering': True,  # Cluster by geographic proximity
    'priority_grouping': True,  # Group by priority level
}
