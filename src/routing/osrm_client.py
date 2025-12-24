"""
OSRM (Open Source Routing Machine) integration for route optimization
"""
import requests
import logging
from typing import List, Tuple, Dict, Optional
import time

from ..data.models import Location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSRMRouter:
    """Interface to OSRM routing service"""
    
    def __init__(self, server_url: str = "http://router.project-osrm.org"):
        """
        Initialize OSRM router
        
        Args:
            server_url: URL of OSRM server
        """
        self.server_url = server_url.rstrip('/')
        self.cache = {}  # Cache for route queries
        logger.info(f"Initialized OSRM router with server: {self.server_url}")
    
    def get_route(
        self, 
        origin: Location, 
        destination: Location,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Get route between two locations
        
        Args:
            origin: Starting location
            destination: Ending location
            use_cache: Use cached results if available
            
        Returns:
            Dictionary with route information (distance, duration, geometry)
        """
        if not origin.has_coordinates() or not destination.has_coordinates():
            logger.warning(f"Missing coordinates for route query")
            return None
        
        # Check cache
        cache_key = f"{origin.id}_{destination.id}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Build OSRM query
        coords = f"{origin.longitude},{origin.latitude};{destination.longitude},{destination.latitude}"
        url = f"{self.server_url}/route/v1/driving/{coords}"
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'steps': 'false'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                result = {
                    'distance_m': route['distance'],
                    'duration_s': route['duration'],
                    'distance_km': route['distance'] / 1000.0,
                    'duration_hours': route['duration'] / 3600.0,
                    'geometry': route.get('geometry')
                }
                
                # Cache the result
                self.cache[cache_key] = result
                return result
            else:
                logger.warning(f"OSRM returned error: {data.get('code')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error querying OSRM: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_distance_matrix(
        self, 
        locations: List[Location],
        use_cache: bool = True
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Get distance and time matrices for a list of locations
        
        Args:
            locations: List of locations
            use_cache: Use cached results if available
            
        Returns:
            Tuple of (distance_matrix_km, time_matrix_hours)
        """
        n = len(locations)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        time_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Check if all locations have coordinates
        missing_coords = [loc for loc in locations if not loc.has_coordinates()]
        if missing_coords:
            logger.warning(f"{len(missing_coords)} locations missing coordinates")
            # Use fallback estimation
            return self._estimate_distance_matrix(locations)
        
        logger.info(f"Computing distance matrix for {n} locations")
        
        # Query OSRM for each pair
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                route = self.get_route(locations[i], locations[j], use_cache=use_cache)
                
                if route:
                    distance_matrix[i][j] = route['distance_km']
                    time_matrix[i][j] = route['duration_hours']
                else:
                    # Fallback to haversine distance
                    dist_km = self._haversine_distance(locations[i], locations[j])
                    distance_matrix[i][j] = dist_km
                    time_matrix[i][j] = dist_km / 40.0  # Assume 40 km/h average
                
                # Rate limiting
                if not use_cache:
                    time.sleep(0.1)  # Be nice to public OSRM server
        
        logger.info("Distance matrix computation complete")
        return distance_matrix, time_matrix
    
    def _haversine_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate haversine distance between two locations
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        if not loc1.has_coordinates() or not loc2.has_coordinates():
            return 0.0
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = radians(loc1.latitude), radians(loc1.longitude)
        lat2, lon2 = radians(loc2.latitude), radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _estimate_distance_matrix(
        self, 
        locations: List[Location]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Estimate distance matrix using haversine distance for locations without coordinates
        
        Args:
            locations: List of locations
            
        Returns:
            Tuple of (distance_matrix_km, time_matrix_hours)
        """
        n = len(locations)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        time_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        avg_speed = 40.0  # km/h
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if locations[i].has_coordinates() and locations[j].has_coordinates():
                        dist = self._haversine_distance(locations[i], locations[j])
                    else:
                        # Random estimation for locations without coordinates
                        dist = 10.0 + (abs(hash(locations[i].id) - hash(locations[j].id)) % 40)
                    
                    distance_matrix[i][j] = dist
                    time_matrix[i][j] = dist / avg_speed
        
        return distance_matrix, time_matrix
    
    def clear_cache(self):
        """Clear the route cache"""
        self.cache.clear()
        logger.info("Route cache cleared")
