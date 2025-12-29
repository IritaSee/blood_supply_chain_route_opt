"""
Quick test of integrated pipeline (GA + DL).
Uses haversine distance to avoid OSRM delays.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import OptimizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run integrated pipeline with GA and DL."""
    logger.info("\n" + "="*80)
    logger.info("INTEGRATED PIPELINE TEST: GA + DL")
    logger.info("="*80 + "\n")
    
    # Initialize pipeline (use haversine for speed)
    pipeline = OptimizationPipeline(use_osrm=False)
    
    # Run with DL enabled
    results = pipeline.run_full_pipeline(
        population_size=50,   # Smaller for quick demo
        generations=100,      # Fewer generations
        train_dl=True,        # Enable DL predictor
        dl_epochs=20          # Quick DL training
    )
    
    logger.info("\n" + "="*80)
    logger.info("INTEGRATED RESULTS SUMMARY")
    logger.info("="*80)
    
    # GA Results
    ga = results.get('ga_results', {})
    logger.info("\nGA Optimization:")
    logger.info(f"  Makespan: {ga.get('makespan_s', 0) / 3600:.2f} hours")
    logger.info(f"  Total distance: {ga.get('total_distance_km', 0):.2f} km")
    logger.info(f"  Total cost: IDR {ga.get('total_cost_idr', 0):,.0f}")
    
    # DL Results
    dl = results.get('dl_metrics', {})
    if dl:
        logger.info("\nDL Time Predictor:")
        logger.info(f"  Test MAE: {dl.get('mae', 0):.2f} minutes")
        logger.info(f"  Test RMSE: {dl.get('rmse', 0):.2f} minutes")
        logger.info(f"  Test R²: {dl.get('r2_score', 0):.4f}")
    
    # Baseline
    baseline = results.get('baseline', {})
    if baseline:
        logger.info("\nHistorical Baseline:")
        logger.info(f"  Avg distance: {baseline.get('avg_distance_km', 0):.2f} km")
        logger.info(f"  On-time rate: {baseline.get('on_time_percentage', 0):.1f}%")
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE - Check results/ folder for outputs")
    logger.info("="*80)


if __name__ == '__main__':
    main()
