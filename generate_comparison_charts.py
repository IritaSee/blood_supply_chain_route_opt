"""
Generate comprehensive comparison charts for Baseline, GA, and DL approaches.
Visualizes performance metrics across distance, time, cost, and predictive accuracy.
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from ga_optimizer.baseline_extractor import BaselineExtractor
from dl_predictor.data_preprocessor import TripDataPreprocessor
from dl_predictor.cnn_model import DeliveryTimeCNN
from config.dl_config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Colors for the three approaches
COLORS = {
    'baseline': '#FF6B6B',      # Red
    'ga': '#4ECDC4',            # Teal
    'dl': '#45B7D1'             # Blue
}


def get_baseline_metrics():
    """Extract baseline metrics from historical trips."""
    logger.info("Extracting baseline metrics...")
    extractor = BaselineExtractor()
    baseline = extractor.get_overall_baseline()
    
    return {
        'name': 'Non-Optimized (Baseline)',
        'num_trips': baseline['num_trips'],
        'avg_distance_km': baseline['avg_distance_km'],
        'total_distance_km': baseline['total_distance_km'],
        'avg_duration_hours': baseline.get('avg_duration_hours', 0),
        'avg_duration_minutes': baseline.get('avg_duration_minutes', 
                                            baseline.get('avg_duration_hours', 0) * 60),
        'total_duration_hours': baseline['num_trips'] * baseline.get('avg_duration_hours', 0),
        'avg_cost_per_trip': baseline['avg_cost_per_trip_idr'],
        'total_cost_idr': baseline['total_cost_idr'],
        'on_time_percentage': baseline['on_time_percentage'],
        'avg_lateness_minutes': baseline['avg_lateness_minutes'],
    }


def get_ga_metrics():
    """Extract GA optimization results."""
    logger.info("Loading GA optimization results...")
    results_file = Path('results/comparison.json')
    
    if not results_file.exists():
        logger.warning("GA results file not found. Running GA optimization...")
        # Try to load from ga_optimization_results.json
        results_file = Path('results/ga_optimization_results.json')
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Handle both comparison.json and ga_optimization_results.json formats
        if 'ga_results' in data:
            ga_data = data['ga_results']
        else:
            ga_data = data
        
        return {
            'name': 'Genetic Algorithm (GA)',
            'num_routes': ga_data.get('num_routes', 2),
            'total_distance_km': ga_data['total_distance_km'],
            'avg_distance_km': ga_data['total_distance_km'] / ga_data.get('num_routes', 2),
            'makespan_hours': ga_data['makespan_hours'],
            'total_duration_hours': ga_data['total_time_hours'],
            'avg_duration_hours': ga_data['total_time_hours'] / ga_data.get('num_routes', 2),
            'avg_duration_minutes': (ga_data['total_time_hours'] / ga_data.get('num_routes', 2)) * 60,
            'total_cost_idr': ga_data['total_cost_idr'],
            'avg_cost_per_route': ga_data['total_cost_idr'] / ga_data.get('num_routes', 2),
        }
    else:
        logger.error("GA results file not found!")
        return None


def get_dl_metrics():
    """Extract DL metrics from saved results or train new model."""
    logger.info("Loading Deep Learning predictor results...")
    
    # First try to load from saved CSV
    metrics_csv = Path('results/dl_models/test_metrics.csv')
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if len(df) > 0:
                row = df.iloc[0]
                logger.info("Loaded DL metrics from saved CSV")
                return {
                    'name': 'Deep Learning (1D CNN)',
                    'test_samples': int(row.get('n_samples', 0)),
                    'train_samples': 0,  # Not stored in metrics CSV
                    'mae_minutes': float(row['mae']),
                    'rmse_minutes': float(row['rmse']),
                    'mape_percent': float(row.get('mape', 0)),
                    'r2_score': float(row['r2_score']),
                }
        except Exception as e:
            logger.warning(f"Could not load DL metrics from CSV: {e}")
    
    # If no saved results, train new model
    logger.info("No saved DL results found, training new model...")
    try:
        # Prepare data
        preprocessor = TripDataPreprocessor(file_path='All Droping.xlsx')
        data = preprocessor.prepare_data(
            target_col='duration_minutes',
            sequence_length=DATA_CONFIG.get('sequence_length', 5),
            test_size=DATA_CONFIG.get('test_size', 0.2),
            random_seed=DATA_CONFIG.get('random_seed', 42)
        )
        
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Build and train model
        config = {
            **MODEL_CONFIG,
            **TRAINING_CONFIG,
            'model_dir': 'results/dl_models',
        }
        
        cnn = DeliveryTimeCNN(config=config)
        cnn.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            conv_filters=MODEL_CONFIG.get('conv_filters', [32, 64, 32]),
            kernel_sizes=MODEL_CONFIG.get('kernel_sizes', [3, 3, 3]),
            pool_sizes=MODEL_CONFIG.get('pool_sizes', [1, 1, 1]),
            dense_units=MODEL_CONFIG.get('dense_units', [64, 32]),
            dropout_rate=MODEL_CONFIG.get('dropout_rate', 0.3)
        )
        
        cnn.compile_model(
            learning_rate=TRAINING_CONFIG.get('learning_rate', 0.001),
            optimizer=TRAINING_CONFIG.get('optimizer', 'adam'),
            loss=TRAINING_CONFIG.get('loss', 'mse'),
            metrics=TRAINING_CONFIG.get('metrics', ['mae', 'mse'])
        )
        
        # Train with reduced verbosity
        history = cnn.train(
            X_train, y_train,
            epochs=20,  # Quick training for chart generation
            batch_size=TRAINING_CONFIG.get('batch_size', 32),
            validation_split=DATA_CONFIG.get('validation_split', 0.15),
            verbose=0
        )
        
        # Evaluate
        metrics = cnn.evaluate(X_test, y_test)
        
        # Save metrics for future use
        save_dir = Path('results/dl_models')
        save_dir.mkdir(parents=True, exist_ok=True)
        cnn.export_metrics_to_csv(metrics, str(save_dir / 'test_metrics.csv'))
        
        return {
            'name': 'Deep Learning (1D CNN)',
            'test_samples': len(X_test),
            'train_samples': len(X_train),
            'mae_minutes': metrics['mae'],
            'rmse_minutes': metrics['rmse'],
            'mape_percent': metrics.get('mape', 0),
            'r2_score': metrics['r2_score'],
            'model': cnn,
            'history': history,
        }
    except Exception as e:
        logger.error(f"Error with DL model: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_distance_comparison(baseline, ga):
    """Create distance comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metric 1: Total Distance
    ax = axes[0]
    approaches = ['Baseline\n(287 trips)', 'GA\n(2 routes)']
    distances = [
        baseline['total_distance_km'],
        ga['total_distance_km']
    ]
    bars = ax.bar(approaches, distances, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Total Distance (km)', fontsize=11, fontweight='bold')
    ax.set_title('Total Distance Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,.0f} km',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Metric 2: Average Distance
    ax = axes[1]
    avg_distances = [
        baseline['avg_distance_km'],
        ga['avg_distance_km']
    ]
    bars = ax.bar(approaches, avg_distances, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Distance (km)', fontsize=11, fontweight='bold')
    ax.set_title('Average Distance per Trip/Route', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} km',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_time_comparison(baseline, ga):
    """Create time/duration comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metric 1: Total Duration
    ax = axes[0]
    approaches = ['Baseline\n(287 trips)', 'GA\n(2 routes)']
    durations = [
        baseline['total_duration_hours'],
        ga['total_duration_hours']
    ]
    bars = ax.bar(approaches, durations, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Total Duration (hours)', fontsize=11, fontweight='bold')
    ax.set_title('Total Delivery Duration Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, durations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} hrs',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Metric 2: Average Duration per Trip
    ax = axes[1]
    avg_durations = [
        baseline['avg_duration_hours'],
        ga['avg_duration_hours']
    ]
    bars = ax.bar(approaches, avg_durations, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Duration (hours)', fontsize=11, fontweight='bold')
    ax.set_title('Average Duration per Trip/Route', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_durations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} hrs',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_cost_comparison(baseline, ga):
    """Create cost comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['Baseline\n(287 trips)', 'GA\n(2 routes)']
    costs = [
        baseline['total_cost_idr'],
        ga['total_cost_idr']
    ]
    
    bars = ax.bar(approaches, costs, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Total Cost (IDR)', fontsize=11, fontweight='bold')
    ax.set_title('Total Cost Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis in millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'IDR {x/1e6:.1f}M'))
    
    for bar, val in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'IDR {val/1e6:.2f}M',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add percentage improvement
    improvement = ((costs[0] - costs[1]) / costs[0]) * 100
    ax.text(0.5, 0.95, f'GA Cost Reduction: {improvement:.1f}%',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_quality_comparison(baseline):
    """Create service quality comparison (on-time rate, lateness)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # On-time percentage
    ax = axes[0]
    on_time = baseline['on_time_percentage']
    late = 100 - on_time
    
    labels = ['On-Time', 'Late']
    sizes = [on_time, late]
    colors_pie = ['#90EE90', '#FFB6C6']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Baseline Delivery On-Time Performance', fontsize=12, fontweight='bold')
    
    # Average lateness
    ax = axes[1]
    categories = ['Baseline\nAverage Lateness']
    lateness = [baseline['avg_lateness_minutes']]
    
    bars = ax.bar(categories, lateness, color=[COLORS['baseline']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5, width=0.5)
    ax.set_ylabel('Average Lateness (minutes)', fontsize=11, fontweight='bold')
    ax.set_title('Average Delivery Lateness', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(lateness) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, lateness):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} min',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_dl_performance(dl_metrics):
    """Create Deep Learning model performance chart."""
    if not dl_metrics:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metric 1: Prediction Errors (MAE, RMSE)
    ax = axes[0]
    metrics_names = ['MAE\n(Mean Absolute Error)', 'RMSE\n(Root Mean Squared Error)']
    metrics_values = [
        dl_metrics['mae_minutes'],
        dl_metrics['rmse_minutes']
    ]
    
    bars = ax.bar(metrics_names, metrics_values, color=[COLORS['dl'], '#95E1D3'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Error (minutes)', fontsize=11, fontweight='bold')
    ax.set_title('1D CNN Time Prediction Errors (Test Set)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} min',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Metric 2: Model Information
    ax = axes[1]
    ax.axis('off')
    
    info_text = f"""
    1D CNN Time Prediction Model Performance
    
    Training Samples: {dl_metrics.get('train_samples', 'N/A')}
    Test Samples: {dl_metrics['test_samples']}
    
    Mean Absolute Error (MAE): {dl_metrics['mae_minutes']:.2f} minutes
    Root Mean Squared Error (RMSE): {dl_metrics['rmse_minutes']:.2f} minutes
    Mean Absolute Percentage Error: {dl_metrics['mape_percent']:.2f}%
    R² Score: {dl_metrics['r2_score']:.4f}
    
    Model Architecture:
    • Input: (sequence_length={DATA_CONFIG.get('sequence_length', 10)}, features=7)
    • Conv1D Filters: {MODEL_CONFIG.get('conv_filters', [32, 64, 32])}
    • Dense Units: {MODEL_CONFIG.get('dense_units', [64, 32])}
    • Dropout Rate: {MODEL_CONFIG.get('dropout_rate', 0.3)}
    • Training Epochs: {TRAINING_CONFIG.get('epochs', 100)}
    • Batch Size: {TRAINING_CONFIG.get('batch_size', 32)}
    """
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def create_integrated_summary(baseline, ga, dl_metrics):
    """Create an integrated summary comparing all three approaches."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Distance per unit (Trip/Route)
    ax = axes[0, 0]
    approaches = ['Baseline', 'GA']
    distances = [baseline['avg_distance_km'], ga['avg_distance_km']]
    bars = ax.bar(approaches, distances, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Distance (km)', fontsize=11, fontweight='bold')
    ax.set_title('Distance Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} km',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Duration per unit
    ax = axes[0, 1]
    durations = [baseline['avg_duration_hours'], ga['avg_duration_hours']]
    bars = ax.bar(approaches, durations, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Duration (hours)', fontsize=11, fontweight='bold')
    ax.set_title('Time Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, durations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} hrs',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Cost per km
    ax = axes[1, 0]
    cost_per_km = [
        baseline['total_cost_idr'] / baseline['total_distance_km'],
        ga['total_cost_idr'] / ga['total_distance_km']
    ]
    bars = ax.bar(approaches, cost_per_km, color=[COLORS['baseline'], COLORS['ga']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Cost per km (IDR)', fontsize=11, fontweight='bold')
    ax.set_title('Cost Efficiency (Cost/km)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, cost_per_km):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'IDR {val:,.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Summary metrics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Baseline', 'GA', 'Improvement %'],
        ['Total Distance', f'{baseline["total_distance_km"]:,.0f} km', 
         f'{ga["total_distance_km"]:,.0f} km',
         f'{((baseline["total_distance_km"]-ga["total_distance_km"])/baseline["total_distance_km"]*100):.1f}%'],
        ['Total Cost', f'IDR {baseline["total_cost_idr"]/1e6:.2f}M', 
         f'IDR {ga["total_cost_idr"]/1e6:.2f}M',
         f'{((baseline["total_cost_idr"]-ga["total_cost_idr"])/baseline["total_cost_idr"]*100):.1f}%'],
        ['Total Duration', f'{baseline["total_duration_hours"]:.1f} hrs', 
         f'{ga["total_duration_hours"]:.1f} hrs',
         f'{((baseline["total_duration_hours"]-ga["total_duration_hours"])/baseline["total_duration_hours"]*100):.1f}%'],
        ['Avg Duration/Unit', f'{baseline["avg_duration_hours"]:.2f} hrs', 
         f'{ga["avg_duration_hours"]:.2f} hrs',
         f'{((baseline["avg_duration_hours"]-ga["avg_duration_hours"])/baseline["avg_duration_hours"]*100):.1f}%'],
    ]
    
    if dl_metrics:
        summary_data.append(['DL MAE (Test)', 
                           f'{dl_metrics["mae_minutes"]:.2f} min',
                           '-', '-'])
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax.set_title('Comprehensive Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def main():
    """Generate all comparison charts."""
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPREHENSIVE COMPARISON CHARTS")
    logger.info("="*80 + "\n")
    
    # Create results directory
    results_dir = Path('results/comparison_charts')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    baseline = get_baseline_metrics()
    ga = get_ga_metrics()
    dl_metrics = get_dl_metrics()
    
    if not ga:
        logger.error("Cannot generate charts without GA results!")
        return
    
    # Generate charts
    charts = []
    
    logger.info("Creating distance comparison chart...")
    fig = create_distance_comparison(baseline, ga)
    if fig:
        fig.savefig(results_dir / '01_distance_comparison.png', dpi=300, bbox_inches='tight')
        charts.append('01_distance_comparison.png')
        plt.close(fig)
    
    logger.info("Creating time/duration comparison chart...")
    fig = create_time_comparison(baseline, ga)
    if fig:
        fig.savefig(results_dir / '02_time_comparison.png', dpi=300, bbox_inches='tight')
        charts.append('02_time_comparison.png')
        plt.close(fig)
    
    logger.info("Creating cost comparison chart...")
    fig = create_cost_comparison(baseline, ga)
    if fig:
        fig.savefig(results_dir / '03_cost_comparison.png', dpi=300, bbox_inches='tight')
        charts.append('03_cost_comparison.png')
        plt.close(fig)
    
    logger.info("Creating service quality chart...")
    fig = create_quality_comparison(baseline)
    if fig:
        fig.savefig(results_dir / '04_quality_comparison.png', dpi=300, bbox_inches='tight')
        charts.append('04_quality_comparison.png')
        plt.close(fig)
    
    if dl_metrics:
        logger.info("Creating DL performance chart...")
        fig = create_dl_performance(dl_metrics)
        if fig:
            fig.savefig(results_dir / '05_dl_performance.png', dpi=300, bbox_inches='tight')
            charts.append('05_dl_performance.png')
            plt.close(fig)
    
    logger.info("Creating integrated summary chart...")
    fig = create_integrated_summary(baseline, ga, dl_metrics)
    if fig:
        fig.savefig(results_dir / '06_integrated_summary.png', dpi=300, bbox_inches='tight')
        charts.append('06_integrated_summary.png')
        plt.close(fig)
    
    # Summary report
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80 + "\n")
    
    logger.info(f"Baseline (Non-Optimized):")
    logger.info(f"  • Total trips: {baseline['num_trips']}")
    logger.info(f"  • Total distance: {baseline['total_distance_km']:,.0f} km")
    logger.info(f"  • Average distance per trip: {baseline['avg_distance_km']:.2f} km")
    logger.info(f"  • Average duration: {baseline['avg_duration_hours']:.2f} hours")
    logger.info(f"  • Total cost: IDR {baseline['total_cost_idr']/1e6:.2f}M")
    logger.info(f"  • On-time rate: {baseline['on_time_percentage']:.1f}%")
    logger.info(f"  • Average lateness: {baseline['avg_lateness_minutes']:.1f} minutes\n")
    
    logger.info(f"GA Optimization:")
    logger.info(f"  • Number of routes: {ga['num_routes']}")
    logger.info(f"  • Total distance: {ga['total_distance_km']:,.0f} km")
    logger.info(f"  • Average distance per route: {ga['avg_distance_km']:.2f} km")
    logger.info(f"  • Average duration per route: {ga['avg_duration_hours']:.2f} hours")
    logger.info(f"  • Total cost: IDR {ga['total_cost_idr']/1e6:.2f}M\n")
    
    distance_improvement = ((baseline['total_distance_km'] - ga['total_distance_km']) / baseline['total_distance_km']) * 100
    cost_improvement = ((baseline['total_cost_idr'] - ga['total_cost_idr']) / baseline['total_cost_idr']) * 100
    time_improvement = ((baseline['total_duration_hours'] - ga['total_duration_hours']) / baseline['total_duration_hours']) * 100
    
    logger.info("GA vs Baseline Improvements:")
    logger.info(f"  • Distance reduction: {distance_improvement:.1f}%")
    logger.info(f"  • Cost reduction: {cost_improvement:.1f}%")
    logger.info(f"  • Duration reduction: {time_improvement:.1f}%\n")
    
    if dl_metrics:
        logger.info("Deep Learning Time Predictor:")
        logger.info(f"  • Training samples: {dl_metrics['train_samples']}")
        logger.info(f"  • Test samples: {dl_metrics['test_samples']}")
        logger.info(f"  • Mean Absolute Error: {dl_metrics['mae_minutes']:.2f} minutes")
        logger.info(f"  • Root Mean Squared Error: {dl_metrics['rmse_minutes']:.2f} minutes")
        logger.info(f"  • R² Score: {dl_metrics['r2_score']:.4f}\n")
    
    logger.info(f"Charts saved to: {results_dir}")
    logger.info(f"Total charts generated: {len(charts)}\n")
    
    for i, chart in enumerate(charts, 1):
        logger.info(f"  {i}. {chart}")
    
    logger.info("\n" + "="*80)
    logger.info("CHARTS GENERATION COMPLETE")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
