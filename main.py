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
from ga_optimizer import geocoder as ga_geocoder
from ga_optimizer import routing as ga_routing
from ga_optimizer.genetic_algorithm import GeneticAlgorithm
from dl_predictor.data_preprocessor import TripDataPreprocessor
from dl_predictor.cnn_model import DeliveryTimeCNN
from dl_predictor.route_selector import DeepLearningRouteSelector
from config.dl_config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG

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
        self.geocoder = ga_geocoder.Geocoder(cache_file=str(self.output_dir / "geocode_cache.db"))
        self.router = ga_routing.OSRMRouter(
            cache_file=str(self.output_dir / "routing_cache.db"),
            use_osrm=use_osrm
        )
        
        self.locations = None
        self.duration_matrix = None
        self.distance_matrix = None
        self.ga_results = None
        self.dl_model = None
        self.dl_metrics = None
        
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
        
        if not self.ga_results:
            logger.warning("Optimization finished but no feasible solution was recorded (ga_results empty)")
            return {}

        logger.info("Optimization complete!")
        logger.info(f"Best makespan: {self.ga_results.get('makespan_s', 0) / 3600:.2f} hours")
        logger.info(f"Total cost: {self.ga_results.get('total_cost_idr', 0):.0f} IDR")
        logger.info(f"Total distance: {self.ga_results.get('total_distance_km', 0):.1f} km")
        
        return self.ga_results
    
    def train_dl_predictor(self, train_dl: bool = True,
                          epochs: int = None) -> Dict:
        """Train or load DL time predictor."""
        logger.info("=== STEP 5B: Deep Learning Time Predictor ===")
        
        if not train_dl:
            logger.info("Skipping DL training (train_dl=False)")
            return {}
        
        epochs = epochs or TRAINING_CONFIG.get('epochs', 50)
        
        # Prepare data
        logger.info("Preparing data for DL training...")
        preprocessor = TripDataPreprocessor(file_path=self.droping_file)
        data = preprocessor.prepare_data(
            target_col='duration_minutes',
            sequence_length=DATA_CONFIG.get('sequence_length', 5),
            test_size=DATA_CONFIG.get('test_size', 0.2),
            random_seed=DATA_CONFIG.get('random_seed', 42)
        )
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Build model
        logger.info("Building 1D CNN model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        config = {
            **MODEL_CONFIG,
            **TRAINING_CONFIG,
            'model_dir': str(self.output_dir / 'dl_models'),
        }
        
        cnn = DeliveryTimeCNN(config=config)
        cnn.build_model(
            input_shape=input_shape,
            conv_filters=MODEL_CONFIG.get('conv_filters', [32, 64]),
            kernel_sizes=MODEL_CONFIG.get('kernel_sizes', [3, 3]),
            pool_sizes=MODEL_CONFIG.get('pool_sizes', [1, 1]),
            dense_units=MODEL_CONFIG.get('dense_units', [64, 32]),
            dropout_rate=MODEL_CONFIG.get('dropout_rate', 0.3),
        )
        
        # Compile
        cnn.compile_model(
            learning_rate=TRAINING_CONFIG.get('learning_rate', 0.001),
            optimizer=TRAINING_CONFIG.get('optimizer', 'adam'),
            loss=TRAINING_CONFIG.get('loss', 'mse'),
            metrics=TRAINING_CONFIG.get('metrics', ['mae', 'mse'])
        )
        
        # Train
        logger.info(f"Training DL model for {epochs} epochs...")
        cnn.train(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=TRAINING_CONFIG.get('batch_size', 32),
            validation_split=DATA_CONFIG.get('validation_split', 0.15),
            verbose=0
        )
        
        # Evaluate
        logger.info("Evaluating DL model...")
        metrics = cnn.evaluate(X_test, y_test)
        
        logger.info(f"DL Test MAE: {metrics['mae']:.2f} minutes")
        logger.info(f"DL Test RMSE: {metrics['rmse']:.2f} minutes")
        logger.info(f"DL Test R²: {metrics['r2_score']:.4f}")
        
        # Save model
        model_file = self.output_dir / 'dl_models' / 'trained_model.keras'
        cnn.save(str(model_file))
        logger.info(f"DL model saved to {model_file}")
        
        self.dl_model = cnn
        self.dl_metrics = metrics
        
        return metrics
    
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
        
        # Save DL metrics if available
        if self.dl_metrics:
            dl_results = {
                'test_mae_minutes': self.dl_metrics['mae'],
                'test_rmse_minutes': self.dl_metrics['rmse'],
                'test_mape_percent': self.dl_metrics['mape'],
                'test_r2_score': self.dl_metrics['r2_score'],
                'n_test_samples': self.dl_metrics['n_samples'],
            }
            
            dl_file = self.output_dir / "dl_results.json"
            with open(dl_file, 'w') as f:
                json.dump(dl_results, f, indent=2)
            
            logger.info(f"DL results saved to {dl_file}")
    
    def optimize_with_dl_selection(self, num_candidates: int = 100) -> Dict:
        """
        Run DL-based route selection (Approach B in Parallel Competitive Model).
        
        Args:
            num_candidates: Number of candidate routes to generate and evaluate
        
        Returns:
            Dict with DL selection results
        """
        logger.info("=== STEP 5C: DL Route Selection ===")
        
        if self.duration_matrix is None or self.distance_matrix is None:
            raise ValueError("Matrices not built")
        
        if self.dl_model is None:
            logger.warning("DL model not trained. Skipping DL route selection.")
            return {}
        
        num_customers = len(self.locations) - 1
        num_vehicles = 2
        
        logger.info(f"DL Route Selection: {num_customers} customers, {num_vehicles} vehicles")
        logger.info(f"Generating {num_candidates} candidate routes...")
        
        # Initialize DL Route Selector
        dl_selector = DeepLearningRouteSelector(
            cnn_model=self.dl_model,
            duration_matrix=self.duration_matrix,
            distance_matrix=self.distance_matrix,
            num_candidates=num_candidates
        )
        
        # Select best route from candidates
        best_routes, best_metrics = dl_selector.select_best_route(
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            time_weight=0.7,
            cost_weight=0.3
        )
        
        # Store results
        self.dl_selection_results = {
            'num_candidates': num_candidates,
            'best_routes': best_routes,
            'total_time_h': best_metrics['total_time_h'],
            'total_distance_km': best_metrics['total_distance_km'],
            'total_cost_idr': best_metrics['total_cost_idr'],
            'avg_lateness_risk': best_metrics['avg_lateness_risk'],
            'vehicle_metrics': best_metrics['vehicle_metrics']
        }
        
        logger.info("DL Selection complete!")
        logger.info(f"Best route found (from {num_candidates} candidates):")
        logger.info(f"  Time: {best_metrics['total_time_h']:.2f} hours")
        logger.info(f"  Distance: {best_metrics['total_distance_km']:.1f} km")
        logger.info(f"  Cost: IDR {best_metrics['total_cost_idr']:,.0f}")
        logger.info(f"  Lateness Risk: {best_metrics['avg_lateness_risk']:.3f}")
        
        return self.dl_selection_results
    
    def generate_comparison_report(self, baseline: Dict) -> str:
        """
        Generate comparison report: GA vs DL vs Baseline.
        
        Args:
            baseline: Historical baseline metrics
        
        Returns:
            Path to comparison report file
        """
        logger.info("=== STEP 6: Generate Comparison Report ===")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PARALLEL COMPETITIVE MODEL - COMPARISON REPORT")
        report_lines.append("Blood Supply Route Optimization - PMI Kabupaten Malang")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Baseline
        report_lines.append("HISTORICAL BASELINE:")
        report_lines.append(f"  Number of trips: {baseline.get('num_trips', 0)}")
        report_lines.append(f"  Average distance: {baseline.get('avg_distance_km', 0):.1f} km")
        report_lines.append(f"  Average cost: IDR {baseline.get('avg_cost_idr', 0):,.0f}")
        report_lines.append(f"  On-time percentage: {baseline.get('on_time_percentage', 0):.1f}%")
        report_lines.append("")
        
        # Approach A: Genetic Algorithm
        report_lines.append("APPROACH A: GENETIC ALGORITHM (GA)")
        report_lines.append("  Mechanism: Evolutionary optimization")
        report_lines.append("  Process: Population -> Selection -> Crossover -> Mutation")
        if self.ga_results:
            report_lines.append(f"  Total Distance: {self.ga_results.get('total_distance_km', 0):.1f} km")
            report_lines.append(f"  Total Time: {self.ga_results.get('makespan_s', 0) / 3600:.2f} hours")
            report_lines.append(f"  Total Cost: IDR {self.ga_results.get('total_cost_idr', 0):,.0f}")
            report_lines.append(f"  Number of routes: {self.ga_results.get('num_routes', 0)}")
        else:
            report_lines.append("  [NO RESULTS - GA not run]")
        report_lines.append("")
        
        # Approach B: DL Route Selector
        report_lines.append("APPROACH B: DEEP LEARNING ROUTE SELECTOR")
        report_lines.append("  Mechanism: Predictive scoring of candidate routes")
        report_lines.append("  Process: Candidate Generation -> Prediction -> Selection")
        if hasattr(self, 'dl_selection_results') and self.dl_selection_results:
            report_lines.append(f"  Candidates evaluated: {self.dl_selection_results.get('num_candidates', 0)}")
            report_lines.append(f"  Total Distance: {self.dl_selection_results.get('total_distance_km', 0):.1f} km")
            report_lines.append(f"  Predicted Time: {self.dl_selection_results.get('total_time_h', 0):.2f} hours")
            report_lines.append(f"  Total Cost: IDR {self.dl_selection_results.get('total_cost_idr', 0):,.0f}")
            report_lines.append(f"  Lateness Risk: {self.dl_selection_results.get('avg_lateness_risk', 0):.3f}")
        else:
            report_lines.append("  [NO RESULTS - DL Selection not run]")
        report_lines.append("")
        
        # Comparison
        report_lines.append("="*80)
        report_lines.append("WINNER DETERMINATION")
        report_lines.append("="*80)
        
        if self.ga_results and hasattr(self, 'dl_selection_results') and self.dl_selection_results:
            ga_score = (0.7 * (self.ga_results.get('makespan_s', 0) / 3600) + 
                       0.3 * (self.ga_results.get('total_cost_idr', 0) / 1000000))
            dl_score = (0.7 * self.dl_selection_results.get('total_time_h', 0) + 
                       0.3 * (self.dl_selection_results.get('total_cost_idr', 0) / 1000000))
            
            report_lines.append(f"GA Score (70% Time + 30% Cost): {ga_score:.4f}")
            report_lines.append(f"DL Score (70% Time + 30% Cost): {dl_score:.4f}")
            report_lines.append("")
            
            if ga_score < dl_score:
                winner = "GENETIC ALGORITHM (GA)"
                improvement = ((dl_score - ga_score) / dl_score) * 100
            else:
                winner = "DEEP LEARNING ROUTE SELECTOR"
                improvement = ((ga_score - dl_score) / ga_score) * 100
            
            report_lines.append(f"WINNER: {winner}")
            report_lines.append(f"Improvement over other method: {improvement:.1f}%")
        else:
            report_lines.append("Cannot determine winner - missing results from one or both approaches")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comparison report saved to {report_path}")
        
        # Also print to console
        for line in report_lines:
            logger.info(line)
        
        return str(report_path)
    
    def run_full_pipeline(self, population_size: int = 150, 
                         generations: int = 800,
                         train_dl: bool = True,
                         dl_epochs: int = 50,
                         num_dl_candidates: int = 100,
                         run_comparison: bool = True) -> Dict:
        """
        Execute full Parallel Competitive Model pipeline.
        
        Args:
            population_size: GA population size
            generations: GA generations
            train_dl: Train DL model
            dl_epochs: DL training epochs
            num_dl_candidates: Number of candidates for DL route selection
            run_comparison: Run both approaches and compare them
        """
        logger.info("\n" + "="*60)
        logger.info("PARALLEL COMPETITIVE MODEL - ROUTE OPTIMIZATION")
        logger.info("Malang Regency, Indonesia")
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
        
        # Step 5A: Optimize with GA (Approach A)
        ga_results = self.optimize(population_size, generations)
        
        # Step 5B: Train DL Predictor
        dl_metrics = {}
        if train_dl:
            try:
                dl_metrics = self.train_dl_predictor(
                    train_dl=True,
                    epochs=dl_epochs
                )
            except Exception as e:
                logger.warning(f"DL training failed: {e}")
                logger.warning("Continuing without DL predictions")
                run_comparison = False
        
        # Step 5C: DL Route Selection (Approach B)
        dl_selection_results = {}
        if run_comparison and self.dl_model:
            try:
                dl_selection_results = self.optimize_with_dl_selection(
                    num_candidates=num_dl_candidates
                )
            except Exception as e:
                logger.warning(f"DL route selection failed: {e}")
                logger.warning("Skipping comparison")
                run_comparison = False
        
        # Step 6: Generate comparison report
        if run_comparison:
            comparison_report_path = self.generate_comparison_report(baseline)
        else:
            comparison_report_path = None
        
        # Save individual results
        self.save_results(baseline)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60 + "\n")
        
        return {
            'summary': summary,
            'locations': geocoded,
            'baseline': baseline,
            'ga_results': ga_results,
            'dl_metrics': dl_metrics,
            'dl_selection_results': dl_selection_results,
            'comparison_report': comparison_report_path,
        }


if __name__ == '__main__':
    # Run Parallel Competitive Model with both GA and DL approaches
    pipeline = OptimizationPipeline(use_osrm=True)
    results = pipeline.run_full_pipeline(
        population_size=150,
        generations=800,
        train_dl=True,          # Enable DL training
        dl_epochs=50,           # DL training epochs
        num_dl_candidates=100,  # Number of candidates for DL route selection
        run_comparison=True     # Compare both approaches
    )

if __name__ == "__main__":
    # Run the full pipeline
    pipeline = OptimizationPipeline(use_osrm=True)
    
    try:
        results = pipeline.run_full_pipeline(
            population_size=150,
            generations=800,
            train_dl=True,
            dl_epochs=50,
            num_dl_candidates=100,
            run_comparison=True
        )
        
        if results:
            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 60)
            logger.info(f"GA Results: {results.get('ga_results_path', 'N/A')}")
            logger.info(f"DL Results: {results.get('dl_results_path', 'N/A')}")
            if 'comparison_report' in results:
                logger.info(f"Comparison Report: {results['comparison_report']}")
                logger.info("\n" + "=" * 60)
                logger.info("COMPARISON RESULTS:")
                logger.info("=" * 60)
                with open(results['comparison_report'], 'r') as f:
                    print(f.read())
        else:
            logger.error("Pipeline failed to complete.")
            
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise
