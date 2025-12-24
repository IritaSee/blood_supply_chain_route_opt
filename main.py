"""
Main application for Blood Supply Chain Route Optimization
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.models import Location
from src.routing.geocoding import Geocoder
from src.routing.osrm_client import OSRMRouter
from src.optimization.batching import DeliveryBatcher
from src.optimization.genetic_algorithm import RouteOptimizer
from src.utils.visualization import plot_routes, plot_optimization_metrics, generate_report
from config.settings import OSRM_SERVER

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    logger.info("=" * 80)
    logger.info("BLOOD SUPPLY CHAIN ROUTE OPTIMIZATION")
    logger.info("Genetic Algorithm-based Optimization for Malang Regency, Indonesia")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load data
        logger.info("\n[STEP 1] Loading data...")
        data_loader = DataLoader(data_dir=".")
        
        # Load Excel files
        df_droping = data_loader.load_all_droping_data("All Droping.xlsx")
        logger.info(f"Loaded {len(df_droping)} records from All Droping.xlsx")
        
        # Extract locations
        locations = data_loader.extract_locations_from_df(df_droping)
        logger.info(f"Extracted {len(locations)} unique locations")
        
        # Step 2: Geocode locations (if needed)
        logger.info("\n[STEP 2] Geocoding locations...")
        geocoder = Geocoder()
        
        # For demo purposes, we'll use sample coordinates
        # In production, uncomment the following to geocode real addresses:
        # geocoder.geocode_locations_batch(locations[:5], delay=1.0)  # Limit to first 5 for demo
        
        # Add sample coordinates for demonstration
        if locations:
            # Central Malang coordinates as base
            base_lat, base_lon = -8.1706, 112.6314
            for i, loc in enumerate(locations[:10]):  # Limit to 10 for demo
                # Add small random offsets to create a cluster
                loc.latitude = base_lat + (i % 4 - 1.5) * 0.05
                loc.longitude = base_lon + (i // 4 - 1) * 0.05
            logger.info(f"Added sample coordinates to {min(10, len(locations))} locations")
        
        # Step 3: Create sample deliveries and vehicles
        logger.info("\n[STEP 3] Creating deliveries and vehicles...")
        deliveries = data_loader.create_sample_deliveries(num_deliveries=10)
        vehicles = data_loader.create_sample_vehicles(num_vehicles=3)
        
        summary = data_loader.get_summary()
        logger.info(f"Data summary: {summary}")
        
        # Step 4: Batch deliveries
        logger.info("\n[STEP 4] Batching deliveries...")
        batcher = DeliveryBatcher()
        batches = batcher.create_optimized_batches(deliveries)
        batch_stats = batcher.get_batch_statistics(batches)
        logger.info(f"Batch statistics: {batch_stats}")
        
        # Step 5: Initialize router
        logger.info("\n[STEP 5] Initializing OSRM router...")
        router = OSRMRouter(server_url=OSRM_SERVER)
        
        # Step 6: Run optimization for first batch
        logger.info("\n[STEP 6] Running genetic algorithm optimization...")
        
        # Use first batch for demonstration
        batch_to_optimize = batches[0] if batches else deliveries
        
        optimizer = RouteOptimizer(
            deliveries=batch_to_optimize,
            vehicles=vehicles,
            router=router
        )
        
        result = optimizer.optimize(verbose=True)
        
        # Step 7: Display results
        logger.info("\n[STEP 7] Optimization complete!")
        logger.info(f"Total Distance: {result.total_distance_km:.2f} km")
        logger.info(f"Total Time: {result.total_time_hours:.2f} hours")
        logger.info(f"Total Cost: IDR {result.total_cost_idr:,.2f}")
        logger.info(f"Fitness Score: {result.fitness_score:.4f}")
        
        # Generate detailed report
        logger.info("\n[STEP 8] Generating reports and visualizations...")
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate text report
        report_path = output_dir / "optimization_report.txt"
        generate_report(result, output_path=str(report_path))
        
        # Generate visualizations
        metrics_path = output_dir / "optimization_metrics.png"
        plot_optimization_metrics(result, save_path=str(metrics_path))
        
        routes_path = output_dir / "optimized_routes.png"
        plot_routes(result.routes, save_path=str(routes_path))
        
        logger.info(f"\nResults saved to {output_dir}/")
        logger.info("  - optimization_report.txt")
        logger.info("  - optimization_metrics.png")
        logger.info("  - optimized_routes.png")
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
