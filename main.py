"""
Main optimization pipeline orchestrator.
Coordinates data extraction, geocoding, routing, and GA optimization.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ga_optimizer.data_extractor import DataExtractor
from ga_optimizer.geocoder import Geocoder
from ga_optimizer.routing import OSRMRouter
from ga_optimizer.genetic_algorithm import GeneticAlgorithm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Main orchestrator for blood supply routing optimization."""
    
    def __init__(self, 
                 pmi_file: str = "Data PMI.xlsx",
                 droping_file: str = "All Droping.xlsx",
                 output_dir: str = "results",
                 use_osrm: bool = True):
        """Initialize pipeline."""
        self.pmi_file = pmi_file
        self.droping_file = droping_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_osrm = use_osrm
        
        self.extractor = DataExtractor(pmi_file, droping_file)
        self.geocoder = Geocoder(cache_file=str(self.output_dir / "geocode_cache.db"))
        self.router = OSRMRouter(
            cache_file=str(self.output_dir / "routing_cache.db"),
            use_osrm=use_osrm
        )
        
        self.locations = None
        self.duration_matrix = None
        self.distance_matrix = None
        self.ga_results = None
        
        logger.info(f"Pipeline initialized (OSRM: {use_osrm})")
    
    def extract_data(self) -> Dict:
        """Extract data from Excel files."""
        logger.info("=== STEP 1: Extract Data ===")
        
        summary = self.extractor.summarize_data()
        
        logger.info(f"Facilities: {summary['num_facilities']}")
        logger.info(f"Trip records: {summary['num_trip_records']}")
        logger.info(f"Hospitals: {summary['num_hospitals']}")
        logger.info(f"Unique locations: {summary['num_unique_locations']}")
        logger.info(f"Distance range: {summary['distance_range_km']['min']:.1f} - "
                   f"{summary['distance_range_km']['max']:.1f} km "
                   f"(avg: {summary['distance_range_km']['mean']:.1f} km)")
        
        return summary
    
    def geocode_locations(self, locations: List[Dict]) -> List[Dict]:
        """Geocode facility locations."""
        logger.info("=== STEP 2: Geocode Locations ===")
        logger.info(f"Geocoding {len(locations)} locations...")
        
        geocoded = self.geocoder.geocode_batch(locations)
        
        # Count successes
        success_count = sum(1 for loc in geocoded if loc.get('lat') is not None)
        logger.info(f"Successfully geocoded: {success_count}/{len(geocoded)}")
        
        if success_count < len(geocoded):
            failed = [loc['name'] for loc in geocoded if loc.get('lat') is None]
            logger.warning(f"Failed to geocode: {failed}")
        
        cache_stats = self.geocoder.get_cache_stats()
        logger.info(f"Cache stats: {cache_stats}")
        
        self.locations = geocoded
        return geocoded
    
    def build_matrices(self) -> tuple:
        """Build distance and time matrices."""
        logger.info("=== STEP 3: Build Distance/Time Matrices ===")
        
        if not self.locations:
            raise ValueError("Locations not geocoded")
        
        logger.info(f"Building matrices for {len(self.locations)} locations "
                   f"({self.router.get_router_mode()})...")
        
        duration_matrix, distance_matrix = self.router.build_matrix(self.locations)
        
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        
        logger.info(f"Duration matrix shape: {duration_matrix.shape}")
        logger.info(f"Distance matrix shape: {distance_matrix.shape}")
        
        return duration_matrix, distance_matrix
    
    def extract_baseline(self, trip_data: pd.DataFrame) -> Dict:
        """Extract baseline metrics from historical trip data."""
        logger.info("=== STEP 4: Extract Baseline Metrics ===")
        
        if trip_data.empty:
            logger.warning("No trip history data available")
            return {}
        
        # Filter valid trips
        valid_trips = trip_data[trip_data['distance_km'].notna()].copy()
        
        if valid_trips.empty:
            return {}
        
        baseline = {
            'num_trips': len(valid_trips),
            'avg_distance_km': valid_trips['distance_km'].mean(),
            'avg_duration_hours': None,
            'avg_cost_idr': None,
            'on_time_percentage': 0,
        }
        
        # Calculate on-time percentage
        if 'status' in valid_trips.columns:
            on_time = (valid_trips['status'].str.lower() == 'tepat waktu').sum()
            baseline['on_time_percentage'] = (on_time / len(valid_trips)) * 100
        
        # Try to calculate duration if available
        try:
            valid_trips['trip_duration_hours'] = pd.to_timedelta(
                valid_trips['trip_duration']
            ).dt.total_seconds() / 3600
            baseline['avg_duration_hours'] = valid_trips['trip_duration_hours'].mean()
        except:
            pass
        
        # Calculate cost
        baseline['avg_cost_idr'] = (
            valid_trips['distance_km'].mean() * (12750 / 9)  # Cost per km
        )
        
        logger.info(f"Baseline metrics:")
        logger.info(f"  Trips: {baseline['num_trips']}")
        logger.info(f"  Avg distance: {baseline['avg_distance_km']:.1f} km")
        logger.info(f"  Avg cost: {baseline['avg_cost_idr']:.0f} IDR")
        logger.info(f"  On-time: {baseline['on_time_percentage']:.1f}%")
        
        return baseline
    
    def optimize(self, population_size: int = 150, 
                 generations: int = 800) -> Dict:
        """Run GA optimization."""
        logger.info("=== STEP 5: Genetic Algorithm Optimization ===")
        
        if self.duration_matrix is None or self.distance_matrix is None:
            raise ValueError("Matrices not built")
        
        # Exclude depot (index 0) from customer count
        num_customers = len(self.locations) - 1
        num_vehicles = 2
        
        logger.info(f"Starting GA: {num_customers} customers, {num_vehicles} vehicles")
        logger.info(f"Parameters: pop={population_size}, gen={generations}")
        
        # Slice matrices (exclude depot from customer matrix)
        # But keep full matrix for routing calculations
        ga = GeneticAlgorithm(
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            duration_matrix=self.duration_matrix,
            distance_matrix=self.distance_matrix,
            vehicle_capacity=100.0,
            population_size=population_size,
            generations=generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=int(population_size * 0.1),
        )
        
        best_solution = ga.run()
        self.ga_results = ga.get_best_solution_details()
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best makespan: {self.ga_results['makespan_s'] / 3600:.2f} hours")
        logger.info(f"Total cost: {self.ga_results['total_cost_idr']:.0f} IDR")
        logger.info(f"Total distance: {self.ga_results['total_distance_km']:.1f} km")
        
        return self.ga_results
    
    def save_results(self, baseline: Dict):
        """Save optimization results to JSON/CSV."""
        logger.info("=== STEP 6: Save Results ===")
        
        if not self.ga_results:
            logger.warning("No GA results to save")
            return
        
        # Save GA results
        results_file = self.output_dir / "ga_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy/non-serializable types
            results_to_save = {
                'makespan_hours': self.ga_results['makespan_s'] / 3600,
                'total_time_hours': self.ga_results['total_time_s'] / 3600,
                'total_distance_km': self.ga_results['total_distance_km'],
                'total_cost_idr': self.ga_results['total_cost_idr'],
                'vehicle_distances_km': [d / 1000 for d in self.ga_results['vehicle_distances_m']],
                'vehicle_times_hours': [t / 3600 for t in self.ga_results['vehicle_times_s']],
                'vehicle_costs_idr': self.ga_results['vehicle_costs_idr'],
                'num_routes': self.ga_results['num_routes'],
            }
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save comparison
        if baseline:
            comparison = {
                'baseline_distance_km': baseline.get('avg_distance_km'),
                'ga_distance_km': self.ga_results['total_distance_km'],
                'distance_reduction_pct': (
                    (1 - self.ga_results['total_distance_km'] / 
                     (baseline.get('avg_distance_km', 1) * self.ga_results['num_routes']))
                    * 100
                ) if baseline.get('avg_distance_km') else 0,
                'baseline_cost_idr': baseline.get('avg_cost_idr'),
                'ga_cost_idr': self.ga_results['total_cost_idr'],
                'cost_reduction_pct': (
                    (1 - self.ga_results['total_cost_idr'] / 
                     (baseline.get('avg_cost_idr', 1) * self.ga_results['num_routes']))
                    * 100
                ) if baseline.get('avg_cost_idr') else 0,
            }
            
            comp_file = self.output_dir / "comparison.json"
            with open(comp_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"Comparison saved to {comp_file}")
    
    def run_full_pipeline(self, population_size: int = 150, 
                         generations: int = 800) -> Dict:
        """Execute full optimization pipeline."""
        logger.info("\n" + "="*60)
        logger.info("BLOOD SUPPLY ROUTING OPTIMIZATION PIPELINE")
        logger.info("Malang Regency, Indonesia - 2-Vehicle GA Optimization")
        logger.info("="*60 + "\n")
        
        # Step 1: Extract data
        summary = self.extract_data()
        
        # Step 2: Geocode locations
        locations = self.extractor.get_all_locations()
        geocoded = self.geocode_locations(locations)
        
        # Step 3: Build matrices
        duration_matrix, distance_matrix = self.build_matrices()
        
        # Step 4: Extract baseline
        baseline = self.extract_baseline(summary['trip_history'])
        
        # Step 5: Optimize
        ga_results = self.optimize(population_size, generations)
        
        # Step 6: Save results
        self.save_results(baseline)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60 + "\n")
        
        return {
            'summary': summary,
            'locations': geocoded,
            'baseline': baseline,
            'ga_results': ga_results,
        }


if __name__ == '__main__':
    pipeline = OptimizationPipeline(use_osrm=True)
    results = pipeline.run_full_pipeline(population_size=150, generations=800)

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
