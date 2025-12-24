"""
Data models for Blood Supply Chain Route Optimization
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum


class PriorityLevel(Enum):
    """Priority levels for blood deliveries"""
    EMERGENCY = 1
    URGENT = 2
    ROUTINE = 3


@dataclass
class Location:
    """Represents a location (hospital, blood bank, etc.)"""
    id: str
    name: str
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_type: str = "hospital"  # hospital, blood_bank, collection_center
    
    def has_coordinates(self) -> bool:
        """Check if location has valid coordinates"""
        return self.latitude is not None and self.longitude is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'address': self.address,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'location_type': self.location_type,
        }


@dataclass
class BloodProduct:
    """Represents a blood product"""
    product_id: str
    blood_type: str  # A+, A-, B+, B-, AB+, AB-, O+, O-
    component: str  # WB (Whole Blood), PRC (Packed Red Cells), etc.
    volume_ml: float
    collection_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'product_id': self.product_id,
            'blood_type': self.blood_type,
            'component': self.component,
            'volume_ml': self.volume_ml,
            'collection_date': self.collection_date.isoformat() if self.collection_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
        }


@dataclass
class Delivery:
    """Represents a blood delivery request"""
    delivery_id: str
    origin: Location
    destination: Location
    products: List[BloodProduct]
    priority: PriorityLevel
    requested_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    total_volume_ml: float = 0.0
    
    def __post_init__(self):
        """Calculate total volume after initialization"""
        if self.products:
            self.total_volume_ml = sum(p.volume_ml for p in self.products)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'delivery_id': self.delivery_id,
            'origin': self.origin.to_dict(),
            'destination': self.destination.to_dict(),
            'products': [p.to_dict() for p in self.products],
            'priority': self.priority.name,
            'requested_time': self.requested_time.isoformat() if self.requested_time else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'total_volume_ml': self.total_volume_ml,
        }


@dataclass
class Vehicle:
    """Represents a delivery vehicle"""
    vehicle_id: str
    capacity_liters: float
    current_location: Optional[Location] = None
    available: bool = True
    fuel_consumption_per_km: float = 0.12
    avg_speed_kmh: float = 40.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'vehicle_id': self.vehicle_id,
            'capacity_liters': self.capacity_liters,
            'current_location': self.current_location.to_dict() if self.current_location else None,
            'available': self.available,
            'fuel_consumption_per_km': self.fuel_consumption_per_km,
            'avg_speed_kmh': self.avg_speed_kmh,
        }


@dataclass
class Route:
    """Represents a delivery route"""
    route_id: str
    vehicle: Vehicle
    deliveries: List[Delivery]
    sequence: List[int] = field(default_factory=list)  # Indices of deliveries in order
    total_distance_km: float = 0.0
    total_time_hours: float = 0.0
    total_cost_idr: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'route_id': self.route_id,
            'vehicle': self.vehicle.to_dict(),
            'deliveries': [d.to_dict() for d in self.deliveries],
            'sequence': self.sequence,
            'total_distance_km': self.total_distance_km,
            'total_time_hours': self.total_time_hours,
            'total_cost_idr': self.total_cost_idr,
        }


@dataclass
class OptimizationResult:
    """Results from route optimization"""
    routes: List[Route]
    total_distance_km: float
    total_time_hours: float
    total_cost_idr: float
    fitness_score: float
    generation: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'routes': [r.to_dict() for r in self.routes],
            'total_distance_km': self.total_distance_km,
            'total_time_hours': self.total_time_hours,
            'total_cost_idr': self.total_cost_idr,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
        }
