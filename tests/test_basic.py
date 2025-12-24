"""
Basic tests for blood supply chain route optimization
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import Location, BloodProduct, Delivery, Vehicle, PriorityLevel
from src.routing.osrm_client import OSRMRouter
from src.optimization.batching import DeliveryBatcher
from src.optimization.genetic_algorithm import RouteOptimizer
from datetime import datetime


def test_location_creation():
    """Test Location model creation"""
    location = Location(
        id="TEST_001",
        name="Test Hospital",
        address="Test Address",
        latitude=-8.1706,
        longitude=112.6314
    )
    assert location.has_coordinates()
    assert location.name == "Test Hospital"
    print("✓ Location creation test passed")


def test_delivery_creation():
    """Test Delivery model creation"""
    origin = Location(
        id="ORIGIN",
        name="Blood Bank",
        address="Origin Address",
        latitude=-8.1706,
        longitude=112.6314
    )
    
    destination = Location(
        id="DEST",
        name="Hospital",
        address="Dest Address",
        latitude=-8.2,
        longitude=112.7
    )
    
    product = BloodProduct(
        product_id="BP_001",
        blood_type="A+",
        component="PRC",
        volume_ml=350.0
    )
    
    delivery = Delivery(
        delivery_id="DEL_001",
        origin=origin,
        destination=destination,
        products=[product],
        priority=PriorityLevel.EMERGENCY
    )
    
    assert delivery.total_volume_ml == 350.0
    assert delivery.priority == PriorityLevel.EMERGENCY
    print("✓ Delivery creation test passed")


def test_vehicle_creation():
    """Test Vehicle model creation"""
    vehicle = Vehicle(
        vehicle_id="VEH_001",
        capacity_liters=100.0,
        avg_speed_kmh=40.0
    )
    
    assert vehicle.available == True
    assert vehicle.capacity_liters == 100.0
    print("✓ Vehicle creation test passed")


def test_osrm_router():
    """Test OSRM router initialization"""
    router = OSRMRouter()
    assert router.server_url == "http://router.project-osrm.org"
    print("✓ OSRM router test passed")


def test_batching():
    """Test delivery batching"""
    # Create sample deliveries
    deliveries = []
    for i in range(5):
        origin = Location(f"O_{i}", "Origin", "Addr", -8.17, 112.63)
        dest = Location(f"D_{i}", f"Dest {i}", "Addr", -8.17 + i*0.01, 112.63)
        product = BloodProduct(f"BP_{i}", "A+", "PRC", 350.0)
        priority = [PriorityLevel.EMERGENCY, PriorityLevel.URGENT, 
                   PriorityLevel.ROUTINE][i % 3]
        
        delivery = Delivery(
            f"DEL_{i}", origin, dest, [product], priority, datetime.now()
        )
        deliveries.append(delivery)
    
    batcher = DeliveryBatcher()
    batches = batcher.create_optimized_batches(deliveries)
    
    assert len(batches) > 0
    assert sum(len(b) for b in batches) == len(deliveries)
    print(f"✓ Batching test passed - created {len(batches)} batches")


def test_optimization():
    """Test genetic algorithm optimization"""
    # Create simple test case
    origin = Location("O", "Origin", "Addr", -8.17, 112.63)
    
    deliveries = []
    for i in range(3):
        dest = Location(f"D_{i}", f"Dest {i}", "Addr", 
                       -8.17 + i*0.02, 112.63 + i*0.02)
        product = BloodProduct(f"BP_{i}", "A+", "PRC", 350.0)
        delivery = Delivery(
            f"DEL_{i}", origin, dest, [product], 
            PriorityLevel.ROUTINE, datetime.now()
        )
        deliveries.append(delivery)
    
    vehicles = [Vehicle(f"VEH_{i}", 100.0, 40.0) for i in range(2)]
    router = OSRMRouter()
    
    # Run optimization with reduced parameters for speed
    from config.settings import OPTIMIZATION_CONFIG
    original_config = OPTIMIZATION_CONFIG.copy()
    OPTIMIZATION_CONFIG['num_generations'] = 10
    OPTIMIZATION_CONFIG['population_size'] = 20
    
    optimizer = RouteOptimizer(
        deliveries=deliveries,
        vehicles=vehicles,
        router=router
    )
    
    result = optimizer.optimize(verbose=False)
    
    # Restore original config
    OPTIMIZATION_CONFIG.update(original_config)
    
    assert result is not None
    assert result.total_distance_km >= 0
    assert result.total_time_hours >= 0
    assert len(result.routes) > 0
    print(f"✓ Optimization test passed - {len(result.routes)} routes created")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Blood Supply Chain Route Optimization Tests")
    print("=" * 60)
    
    tests = [
        test_location_creation,
        test_delivery_creation,
        test_vehicle_creation,
        test_osrm_router,
        test_batching,
        test_optimization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
