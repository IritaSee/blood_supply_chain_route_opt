"""
Quick test of the GA optimization pipeline using haversine (no OSRM calls).
Uses simplified data for faster execution.
"""

import logging
from main import OptimizationPipeline
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with haversine (fast, no OSRM delays)
print("\n" + "="*70)
print("QUICK TEST: GA Optimization with Haversine Distance (No OSRM)")
print("="*70 + "\n")

pipeline = OptimizationPipeline(use_osrm=False)

try:
    # Run with small parameters for quick test
    results = pipeline.run_full_pipeline(population_size=50, generations=100)
    
    print("\n" + "="*70)
    print("TEST COMPLETE - Results saved to ./results/")
    print("="*70)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
