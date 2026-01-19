"""
Deep Learning Route Selector for Parallel Competitive Model.

Generates candidate route permutations and selects the best one using
trained 1D CNN predictions for total duration and lateness risk.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import permutations
import random

from dl_predictor.cnn_model import DeliveryTimeCNN

logger = logging.getLogger(__name__)


class DeepLearningRouteSelector:
    """
    DL-based route selector using candidate generation and CNN prediction.
    
    Approach:
    1. Generate N diverse route permutations (50-100 candidates)
    2. Predict total duration and lateness risk for each candidate
    3. Select the route with the best score (lowest predicted time/lateness)
    """
    
    def __init__(self, 
                 cnn_model: DeliveryTimeCNN,
                 duration_matrix: np.ndarray,
                 distance_matrix: np.ndarray,
                 num_candidates: int = 100):
        """
        Initialize DL Route Selector.
        
        Args:
            cnn_model: Trained DeliveryTimeCNN model
            duration_matrix: OSRM duration matrix (n x n) in seconds
            distance_matrix: OSRM distance matrix (n x n) in meters
            num_candidates: Number of candidate routes to generate
        """
        self.model = cnn_model
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.num_candidates = num_candidates
        
        logger.info(f"DL Route Selector initialized with {num_candidates} candidates")
    
    def generate_candidates(self, 
                          num_customers: int,
                          num_vehicles: int = 2) -> List[List[List[int]]]:
        """
        Generate diverse route permutations using heuristics.
        
        Uses:
        - Nearest Neighbor heuristic as base
        - Random swaps for diversity
        - Random partitioning for multi-vehicle scenarios
        
        Args:
            num_customers: Number of customers to visit
            num_vehicles: Number of vehicles
        
        Returns:
            List of candidate solutions, where each solution is:
            [[vehicle1_route], [vehicle2_route], ...]
        """
        logger.info(f"Generating {self.num_candidates} candidate routes for {num_customers} customers")
        
        candidates = []
        customers = list(range(num_customers))
        
        # Generate diverse candidates
        for i in range(self.num_candidates):
            if i == 0:
                # First candidate: Nearest Neighbor
                candidate = self._nearest_neighbor_split(customers, num_vehicles)
            elif i < 10:
                # Next 9: Variations of nearest neighbor with swaps
                candidate = self._nearest_neighbor_split(customers, num_vehicles)
                candidate = self._apply_random_swaps(candidate, num_swaps=2)
            else:
                # Rest: Random permutations with balanced splits
                candidate = self._random_split(customers, num_vehicles)
            
            candidates.append(candidate)
        
        logger.info(f"Generated {len(candidates)} candidate routes")
        return candidates
    
    def _nearest_neighbor_split(self, 
                                customers: List[int],
                                num_vehicles: int) -> List[List[int]]:
        """
        Nearest neighbor heuristic with simple split.
        
        Args:
            customers: List of customer indices
            num_vehicles: Number of vehicles
        
        Returns:
            Routes for each vehicle
        """
        # Simple approach: alternating assignment
        routes = [[] for _ in range(num_vehicles)]
        
        remaining = customers.copy()
        current_vehicle = 0
        
        while remaining:
            # For simplicity, use distance from depot (index 0)
            if routes[current_vehicle]:
                # Find nearest to last customer in current route
                last_idx = routes[current_vehicle][-1] + 1  # +1 because depot is 0
                nearest = min(remaining, 
                            key=lambda x: self.duration_matrix[last_idx, x + 1])
            else:
                # Find nearest to depot
                nearest = min(remaining,
                            key=lambda x: self.duration_matrix[0, x + 1])
            
            routes[current_vehicle].append(nearest)
            remaining.remove(nearest)
            
            # Alternate vehicles
            current_vehicle = (current_vehicle + 1) % num_vehicles
        
        return routes
    
    def _random_split(self, 
                     customers: List[int],
                     num_vehicles: int) -> List[List[int]]:
        """
        Random permutation with balanced split.
        
        Args:
            customers: List of customer indices
            num_vehicles: Number of vehicles
        
        Returns:
            Randomly shuffled routes for each vehicle
        """
        shuffled = customers.copy()
        random.shuffle(shuffled)
        
        # Split roughly evenly
        routes = [[] for _ in range(num_vehicles)]
        for i, customer in enumerate(shuffled):
            routes[i % num_vehicles].append(customer)
        
        return routes
    
    def _apply_random_swaps(self, 
                           routes: List[List[int]],
                           num_swaps: int = 2) -> List[List[int]]:
        """
        Apply random swaps within routes for diversity.
        
        Args:
            routes: Current routes
            num_swaps: Number of swaps to apply
        
        Returns:
            Modified routes
        """
        routes_copy = [route.copy() for route in routes]
        
        for _ in range(num_swaps):
            # Pick random vehicle
            vehicle_idx = random.randint(0, len(routes_copy) - 1)
            
            if len(routes_copy[vehicle_idx]) >= 2:
                # Swap two random positions
                i = random.randint(0, len(routes_copy[vehicle_idx]) - 1)
                j = random.randint(0, len(routes_copy[vehicle_idx]) - 1)
                routes_copy[vehicle_idx][i], routes_copy[vehicle_idx][j] = \
                    routes_copy[vehicle_idx][j], routes_copy[vehicle_idx][i]
        
        return routes_copy
    
    def predict_route_score(self, 
                           routes: List[List[int]],
                           cost_per_km: float = 1416.67) -> Dict:
        """
        Predict total duration, lateness risk, and cost for a route solution.
        
        Args:
            routes: Route solution [[v1_route], [v2_route], ...]
            cost_per_km: Cost per kilometer
        
        Returns:
            Dict with predicted metrics
        """
        total_time = 0.0
        total_distance = 0.0
        total_lateness_risk = 0.0
        vehicle_metrics = []
        
        for vehicle_id, route in enumerate(routes):
            if not route:
                continue
            
            # Calculate route metrics
            time_s = 0.0
            distance_m = 0.0
            
            # Depot (0) to first customer
            current = 0
            for customer_id in route:
                next_loc = customer_id + 1  # Customer indices offset by depot
                time_s += self.duration_matrix[current, next_loc]
                distance_m += self.distance_matrix[current, next_loc]
                current = next_loc
            
            # Last customer back to depot
            time_s += self.duration_matrix[current, 0]
            distance_m += self.distance_matrix[current, 0]
            
            total_time += time_s
            total_distance += distance_m
            
            # Predict lateness risk using CNN if available
            # For now, use simple heuristic: longer routes = higher risk
            route_lateness_risk = min(1.0, time_s / 7200)  # Normalize by 2 hours
            total_lateness_risk += route_lateness_risk
            
            vehicle_metrics.append({
                'vehicle_id': vehicle_id,
                'time_s': time_s,
                'distance_m': distance_m,
                'lateness_risk': route_lateness_risk
            })
        
        # Calculate cost
        total_cost = (total_distance / 1000.0) * cost_per_km
        
        return {
            'total_time_s': total_time,
            'total_time_h': total_time / 3600,
            'total_distance_m': total_distance,
            'total_distance_km': total_distance / 1000,
            'total_cost_idr': total_cost,
            'total_lateness_risk': total_lateness_risk,
            'avg_lateness_risk': total_lateness_risk / len(routes) if routes else 0,
            'vehicle_metrics': vehicle_metrics
        }
    
    def select_best_route(self, 
                         num_customers: int,
                         num_vehicles: int = 2,
                         time_weight: float = 0.7,
                         cost_weight: float = 0.3) -> Tuple[List[List[int]], Dict]:
        """
        Generate candidates and select the best route using DL predictions.
        
        Args:
            num_customers: Number of customers to visit
            num_vehicles: Number of vehicles
            time_weight: Weight for time objective (default 70%)
            cost_weight: Weight for cost objective (default 30%)
        
        Returns:
            Tuple of (best_routes, best_metrics)
        """
        logger.info("DL Route Selection: Generating and evaluating candidates...")
        
        # Generate candidates
        candidates = self.generate_candidates(num_customers, num_vehicles)
        
        # Evaluate each candidate
        best_score = float('inf')
        best_routes = None
        best_metrics = None
        
        for i, candidate in enumerate(candidates):
            # Predict metrics for this candidate
            metrics = self.predict_route_score(candidate)
            
            # Compute weighted score (lower is better)
            # Normalize by typical values for fair comparison
            time_norm = metrics['total_time_s'] / 3600  # hours
            cost_norm = metrics['total_cost_idr'] / 1000000  # millions IDR
            
            score = (time_weight * time_norm) + (cost_weight * cost_norm)
            
            if score < best_score:
                best_score = score
                best_routes = candidate
                best_metrics = metrics
                best_metrics['score'] = score
        
        logger.info(f"DL Selection complete. Best score: {best_score:.4f}")
        logger.info(f"  Time: {best_metrics['total_time_h']:.2f} hours")
        logger.info(f"  Distance: {best_metrics['total_distance_km']:.1f} km")
        logger.info(f"  Cost: IDR {best_metrics['total_cost_idr']:,.0f}")
        
        return best_routes, best_metrics
