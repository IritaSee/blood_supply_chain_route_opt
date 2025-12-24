"""
Geocoding utilities using OpenStreetMap Nominatim
"""
import time
import logging
from typing import Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from ..data.models import Location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Geocoder:
    """Geocoding service using OpenStreetMap Nominatim"""
    
    def __init__(self, user_agent: str = "blood_supply_chain_optimizer"):
        """
        Initialize geocoder
        
        Args:
            user_agent: User agent string for Nominatim
        """
        self.geocoder = Nominatim(user_agent=user_agent)
        self.default_region = "Malang Regency, East Java, Indonesia"
        logger.info("Initialized Nominatim geocoder")
    
    def geocode_address(
        self, 
        address: str, 
        region: Optional[str] = None,
        retry: int = 3
    ) -> Optional[Tuple[float, float]]:
        """
        Geocode an address to coordinates
        
        Args:
            address: Address string
            region: Region to append to address for better results
            retry: Number of retries on failure
            
        Returns:
            Tuple of (latitude, longitude) or None if failed
        """
        if region is None:
            region = self.default_region
        
        # Construct full address
        full_address = f"{address}, {region}" if region else address
        
        for attempt in range(retry):
            try:
                location = self.geocoder.geocode(full_address, timeout=10)
                
                if location:
                    logger.info(f"Geocoded: {address} -> ({location.latitude}, {location.longitude})")
                    return (location.latitude, location.longitude)
                else:
                    logger.warning(f"No results for address: {address}")
                    return None
                    
            except GeocoderTimedOut:
                logger.warning(f"Geocoding timeout (attempt {attempt + 1}/{retry})")
                time.sleep(1)
            except GeocoderServiceError as e:
                logger.error(f"Geocoding service error: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected geocoding error: {e}")
                return None
        
        logger.error(f"Failed to geocode after {retry} attempts: {address}")
        return None
    
    def geocode_location(self, location: Location, region: Optional[str] = None) -> bool:
        """
        Geocode a Location object and update its coordinates
        
        Args:
            location: Location object to geocode
            region: Optional region override
            
        Returns:
            True if successful, False otherwise
        """
        if location.has_coordinates():
            logger.debug(f"Location {location.id} already has coordinates")
            return True
        
        coords = self.geocode_address(location.address, region)
        
        if coords:
            location.latitude, location.longitude = coords
            return True
        else:
            logger.warning(f"Failed to geocode location: {location.name}")
            return False
    
    def geocode_locations_batch(
        self, 
        locations: list[Location],
        region: Optional[str] = None,
        delay: float = 1.0
    ) -> int:
        """
        Geocode multiple locations with rate limiting
        
        Args:
            locations: List of Location objects
            region: Optional region override
            delay: Delay between requests in seconds
            
        Returns:
            Number of successfully geocoded locations
        """
        success_count = 0
        
        for i, location in enumerate(locations):
            if location.has_coordinates():
                success_count += 1
                continue
            
            logger.info(f"Geocoding {i+1}/{len(locations)}: {location.name}")
            
            if self.geocode_location(location, region):
                success_count += 1
            
            # Rate limiting to be respectful to Nominatim
            if i < len(locations) - 1:
                time.sleep(delay)
        
        logger.info(f"Geocoded {success_count}/{len(locations)} locations")
        return success_count
    
    def reverse_geocode(
        self, 
        latitude: float, 
        longitude: float
    ) -> Optional[str]:
        """
        Reverse geocode coordinates to address
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Address string or None if failed
        """
        try:
            location = self.geocoder.reverse(f"{latitude}, {longitude}", timeout=10)
            
            if location:
                return location.address
            else:
                return None
                
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return None
