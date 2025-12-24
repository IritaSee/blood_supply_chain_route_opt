"""
Data loader for Blood Supply Chain Route Optimization
Reads Excel files and creates data models
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from .models import Location, BloodProduct, Delivery, Vehicle, PriorityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and process blood supply chain data from Excel files"""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing Excel data files
        """
        self.data_dir = Path(data_dir)
        self.locations: Dict[str, Location] = {}
        self.deliveries: List[Delivery] = []
        self.vehicles: List[Vehicle] = []
        self.geocode_cache: Dict[str, Tuple[float, float]] = {}
        self._load_geocode_cache()
    
    def _load_geocode_cache(self):
        """Load geocoding cache from file"""
        cache_file = self.data_dir / 'data' / 'geocode_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.geocode_cache = json.load(f)
                logger.info(f"Loaded {len(self.geocode_cache)} cached geocodes")
            except Exception as e:
                logger.warning(f"Failed to load geocode cache: {e}")
    
    def _save_geocode_cache(self):
        """Save geocoding cache to file"""
        cache_file = self.data_dir / 'data' / 'geocode_cache.json'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.geocode_cache, f, indent=2)
            logger.info(f"Saved {len(self.geocode_cache)} geocodes to cache")
        except Exception as e:
            logger.warning(f"Failed to save geocode cache: {e}")
    
    def load_all_droping_data(self, filename: str = "All Droping.xlsx") -> pd.DataFrame:
        """
        Load the 'All Droping' Excel file
        
        Args:
            filename: Name of the Excel file
            
        Returns:
            DataFrame with the data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading data from {filepath}")
        
        try:
            # Try to load all sheets
            xls = pd.ExcelFile(filepath)
            logger.info(f"Found sheets: {xls.sheet_names}")
            
            # Load first sheet or all sheets
            df = pd.read_excel(filepath, sheet_name=0)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def load_pmi_data(self, filename: str = "Data PMI.xlsx") -> pd.DataFrame:
        """
        Load the PMI (Blood Bank) data Excel file
        
        Args:
            filename: Name of the Excel file
            
        Returns:
            DataFrame with the data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading PMI data from {filepath}")
        
        try:
            xls = pd.ExcelFile(filepath)
            logger.info(f"Found sheets: {xls.sheet_names}")
            
            df = pd.read_excel(filepath, sheet_name=0)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def extract_locations_from_df(self, df: pd.DataFrame) -> List[Location]:
        """
        Extract location information from DataFrame
        
        Args:
            df: DataFrame containing location data
            
        Returns:
            List of Location objects
        """
        locations = []
        
        # Try to identify location-related columns
        potential_name_cols = [col for col in df.columns if any(
            keyword in str(col).lower() 
            for keyword in ['nama', 'name', 'lokasi', 'location', 'tempat', 'place', 'rs', 'hospital']
        )]
        
        potential_address_cols = [col for col in df.columns if any(
            keyword in str(col).lower() 
            for keyword in ['alamat', 'address', 'lokasi', 'location']
        )]
        
        if not potential_name_cols:
            logger.warning("No location name columns found")
            return locations
        
        name_col = potential_name_cols[0]
        address_col = potential_address_cols[0] if potential_address_cols else name_col
        
        logger.info(f"Using columns - Name: {name_col}, Address: {address_col}")
        
        for idx, row in df.iterrows():
            if pd.notna(row[name_col]):
                location_id = f"LOC_{idx:04d}"
                name = str(row[name_col]).strip()
                address = str(row[address_col]).strip() if pd.notna(row[address_col]) else name
                
                location = Location(
                    id=location_id,
                    name=name,
                    address=address,
                    location_type="hospital"
                )
                
                locations.append(location)
                self.locations[location_id] = location
        
        logger.info(f"Extracted {len(locations)} locations")
        return locations
    
    def create_sample_deliveries(
        self, 
        num_deliveries: int = 10,
        origin_location: Optional[Location] = None
    ) -> List[Delivery]:
        """
        Create sample delivery requests for testing
        
        Args:
            num_deliveries: Number of deliveries to create
            origin_location: Origin location (blood bank), if None uses first location
            
        Returns:
            List of Delivery objects
        """
        if not self.locations:
            logger.error("No locations available. Load location data first.")
            return []
        
        location_list = list(self.locations.values())
        
        if origin_location is None:
            origin_location = location_list[0]
        
        blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        components = ['WB', 'PRC', 'TC', 'FFP']  # Whole Blood, Packed Red Cells, Thrombocyte, Fresh Frozen Plasma
        priorities = [PriorityLevel.EMERGENCY, PriorityLevel.URGENT, PriorityLevel.ROUTINE]
        
        deliveries = []
        
        for i in range(min(num_deliveries, len(location_list) - 1)):
            # Select destination (skip origin)
            destination = location_list[(i + 1) % len(location_list)]
            
            # Create blood products for this delivery
            num_products = 1 + (i % 3)  # 1-3 products per delivery
            products = []
            
            for j in range(num_products):
                product = BloodProduct(
                    product_id=f"BP_{i:04d}_{j:02d}",
                    blood_type=blood_types[i % len(blood_types)],
                    component=components[j % len(components)],
                    volume_ml=350.0,  # Standard blood bag volume
                    collection_date=datetime.now()
                )
                products.append(product)
            
            # Create delivery
            delivery = Delivery(
                delivery_id=f"DEL_{i:04d}",
                origin=origin_location,
                destination=destination,
                products=products,
                priority=priorities[i % len(priorities)],
                requested_time=datetime.now()
            )
            
            deliveries.append(delivery)
        
        self.deliveries = deliveries
        logger.info(f"Created {len(deliveries)} sample deliveries")
        return deliveries
    
    def create_sample_vehicles(self, num_vehicles: int = 3) -> List[Vehicle]:
        """
        Create sample vehicles for testing
        
        Args:
            num_vehicles: Number of vehicles to create
            
        Returns:
            List of Vehicle objects
        """
        from config.settings import VEHICLE_CONFIG
        
        vehicles = []
        
        for i in range(num_vehicles):
            vehicle = Vehicle(
                vehicle_id=f"VEH_{i:03d}",
                capacity_liters=VEHICLE_CONFIG['capacity_liters'],
                fuel_consumption_per_km=VEHICLE_CONFIG['fuel_consumption_per_km'],
                avg_speed_kmh=VEHICLE_CONFIG['avg_speed_kmh'],
                available=True
            )
            vehicles.append(vehicle)
        
        self.vehicles = vehicles
        logger.info(f"Created {len(vehicles)} sample vehicles")
        return vehicles
    
    def get_summary(self) -> Dict:
        """Get summary of loaded data"""
        return {
            'num_locations': len(self.locations),
            'num_deliveries': len(self.deliveries),
            'num_vehicles': len(self.vehicles),
            'locations_with_coordinates': sum(1 for loc in self.locations.values() if loc.has_coordinates()),
        }
