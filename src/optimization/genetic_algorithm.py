"""
Genetic Algorithm implementation for Blood Supply Chain Route Optimization
"""
import random
import logging
from typing import List, Tuple, Callable
from dataclasses import dataclass
from deap import base, creator, tools, algorithms
import numpy as np

from ..data.models import Delivery, Vehicle, Route, OptimizationResult
from ..routing.osrm_client import OSRMRouter
from config.settings import OPTIMIZATION_CONFIG, VEHICLE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RouteOptimizer:
    """Genetic Algorithm-based route optimizer"""
    
    deliveries: List[Delivery]
    vehicles: List[Vehicle]
    router: OSRMRouter
    
    def __post_init__(self):
        """Initialize optimizer after dataclass initialization"""
        self.config = OPTIMIZATION_CONFIG
        self.vehicle_config = VEHICLE_CONFIG
        self.distance_matrix = None
        self.time_matrix = None
        self.setup_deap()
    
    def setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("indices", random.sample, 
                            range(len(self.deliveries)), 
                            len(self.deliveries))
        self.toolbox.register("individual", tools.initIterate, 
                            creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, 
                            list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.config['tournament_size'])
        self.toolbox.register("evaluate", self.evaluate_fitness)
    
    def compute_matrices(self):
        """Compute distance and time matrices for all delivery locations"""
        logger.info("Computing distance and time matrices...")
        
        # Extract all unique locations
        locations = []
        location_map = {}
        
        for delivery in self.deliveries:
            if delivery.origin.id not in location_map:
                location_map[delivery.origin.id] = len(locations)
                locations.append(delivery.origin)
            if delivery.destination.id not in location_map:
                location_map[delivery.destination.id] = len(locations)
                locations.append(delivery.destination)
        
        # Get matrices from router
        self.distance_matrix, self.time_matrix = self.router.get_distance_matrix(locations)
        self.location_map = location_map
        self.locations = locations
        
        logger.info(f"Computed matrices for {len(locations)} locations")
    
    def evaluate_fitness(self, individual: List[int]) -> Tuple[float]:
        """
        Evaluate fitness of a route sequence
        
        Args:
            individual: List of delivery indices representing route sequence
            
        Returns:
            Tuple with single fitness value (lower is better)
        """
        if self.distance_matrix is None:
            self.compute_matrices()
        
        total_time = 0.0
        total_distance = 0.0
        total_cost = 0.0
        
        # Split deliveries among available vehicles
        deliveries_per_vehicle = len(individual) // len(self.vehicles) + 1
        
        for vehicle_idx, vehicle in enumerate(self.vehicles):
            start_idx = vehicle_idx * deliveries_per_vehicle
            end_idx = min(start_idx + deliveries_per_vehicle, len(individual))
            
            if start_idx >= len(individual):
                break
            
            vehicle_deliveries = [individual[i] for i in range(start_idx, end_idx)]
            
            # Calculate route metrics
            route_distance, route_time, route_cost = self._calculate_route_metrics(
                vehicle_deliveries, vehicle
            )
            
            total_distance += route_distance
            total_time += route_time
            total_cost += route_cost
        
        # Weighted fitness (minimize time primarily, then cost)
        time_weight = self.config['time_weight']
        cost_weight = self.config['cost_weight']
        
        # Normalize metrics for fair comparison
        normalized_time = total_time / len(self.deliveries)  # hours per delivery
        normalized_cost = total_cost / 1000000  # in millions IDR
        
        fitness = (time_weight * normalized_time) + (cost_weight * normalized_cost)
        
        return (fitness,)
    
    def _calculate_route_metrics(
        self, 
        delivery_indices: List[int], 
        vehicle: Vehicle
    ) -> Tuple[float, float, float]:
        """
        Calculate distance, time, and cost for a route
        
        Args:
            delivery_indices: Indices of deliveries in sequence
            vehicle: Vehicle for this route
            
        Returns:
            Tuple of (distance_km, time_hours, cost_idr)
        """
        if not delivery_indices:
            return 0.0, 0.0, 0.0
        
        total_distance = 0.0
        total_time = 0.0
        
        # Service time per delivery
        service_time_hours = self.config['service_time_minutes'] / 60.0
        
        prev_location_idx = None
        
        for delivery_idx in delivery_indices:
            delivery = self.deliveries[delivery_idx]
            
            # Get location indices
            origin_idx = self.location_map[delivery.origin.id]
            dest_idx = self.location_map[delivery.destination.id]
            
            # Distance from previous location to origin
            if prev_location_idx is not None:
                total_distance += self.distance_matrix[prev_location_idx][origin_idx]
                total_time += self.time_matrix[prev_location_idx][origin_idx]
            
            # Distance from origin to destination
            total_distance += self.distance_matrix[origin_idx][dest_idx]
            total_time += self.time_matrix[origin_idx][dest_idx]
            
            # Add service time
            total_time += service_time_hours
            
            prev_location_idx = dest_idx
        
        # Calculate costs
        fuel_cost = (total_distance * vehicle.fuel_consumption_per_km * 
                    self.vehicle_config['fuel_price_per_liter'])
        driver_cost = total_time * self.vehicle_config['driver_cost_per_hour']
        maintenance_cost = total_distance * self.vehicle_config['vehicle_maintenance_cost_per_km']
        
        total_cost = fuel_cost + driver_cost + maintenance_cost
        
        return total_distance, total_time, total_cost
    
    def optimize(self, verbose: bool = True) -> OptimizationResult:
        """
        Run genetic algorithm optimization
        
        Args:
            verbose: Print progress information
            
        Returns:
            OptimizationResult with optimized routes
        """
        logger.info("Starting genetic algorithm optimization...")
        
        # Ensure matrices are computed
        if self.distance_matrix is None:
            self.compute_matrices()
        
        # Create initial population
        population = self.toolbox.population(n=self.config['population_size'])
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of Fame to keep best individuals
        hof = tools.HallOfFame(self.config['elite_size'])
        
        # Run genetic algorithm
        population, logbook = algorithms.eaSimple(
            population, 
            self.toolbox,
            cxpb=self.config['crossover_probability'],
            mutpb=self.config['mutation_probability'],
            ngen=self.config['num_generations'],
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Get best solution
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        
        logger.info(f"Optimization complete. Best fitness: {best_fitness:.4f}")
        
        # Convert best individual to routes
        routes = self._individual_to_routes(best_individual)
        
        # Calculate total metrics
        total_distance = sum(r.total_distance_km for r in routes)
        total_time = sum(r.total_time_hours for r in routes)
        total_cost = sum(r.total_cost_idr for r in routes)
        
        result = OptimizationResult(
            routes=routes,
            total_distance_km=total_distance,
            total_time_hours=total_time,
            total_cost_idr=total_cost,
            fitness_score=best_fitness,
            generation=self.config['num_generations']
        )
        
        return result
    
    def _individual_to_routes(self, individual: List[int]) -> List[Route]:
        """
        Convert an individual (sequence of delivery indices) to Route objects
        
        Args:
            individual: List of delivery indices
            
        Returns:
            List of Route objects
        """
        routes = []
        deliveries_per_vehicle = len(individual) // len(self.vehicles) + 1
        
        for vehicle_idx, vehicle in enumerate(self.vehicles):
            start_idx = vehicle_idx * deliveries_per_vehicle
            end_idx = min(start_idx + deliveries_per_vehicle, len(individual))
            
            if start_idx >= len(individual):
                break
            
            vehicle_delivery_indices = individual[start_idx:end_idx]
            vehicle_deliveries = [self.deliveries[i] for i in vehicle_delivery_indices]
            
            # Calculate metrics
            distance, time, cost = self._calculate_route_metrics(
                vehicle_delivery_indices, vehicle
            )
            
            route = Route(
                route_id=f"ROUTE_{vehicle_idx:03d}",
                vehicle=vehicle,
                deliveries=vehicle_deliveries,
                sequence=vehicle_delivery_indices,
                total_distance_km=distance,
                total_time_hours=time,
                total_cost_idr=cost
            )
            
            routes.append(route)
        
        return routes
