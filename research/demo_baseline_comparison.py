"""
Demo: Baseline extraction and comparison with GA results.
"""

import logging
from pathlib import Path
from ga_optimizer.baseline_extractor import BaselineExtractor
from ga_optimizer.comparison_reporter import ComparisonReporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run baseline extraction and comparison demo."""
    logger.info("\n" + "="*80)
    logger.info("BASELINE EXTRACTION & COMPARISON DEMO")
    logger.info("="*80 + "\n")
    
    # Extract baseline from historical data
    logger.info("=== Step 1: Extract Historical Baseline ===\n")
    extractor = BaselineExtractor(file_path="All Droping.xlsx")
    baseline = extractor.get_overall_baseline()
    
    logger.info(f"Historical Trips: {baseline.get('num_trips', 0)}")
    logger.info(f"Average Distance: {baseline.get('avg_distance_km', 0):.2f} km")
    logger.info(f"Total Distance: {baseline.get('total_distance_km', 0):.2f} km")
    logger.info(f"On-Time Rate: {baseline.get('on_time_percentage', 0):.1f}%")
    logger.info(f"Total Historical Cost: IDR {baseline.get('total_cost_idr', 0):,.0f}")
    logger.info(f"Average Lateness: {baseline.get('avg_lateness_minutes', 0):.1f} minutes\n")
    
    # Get monthly breakdown
    logger.info("=== Step 2: Monthly Baseline Breakdown ===\n")
    monthly = extractor.get_monthly_baseline()
    for month, metrics in list(monthly.items())[:3]:  # Show first 3 months
        logger.info(f"{month}:")
        logger.info(f"  Trips: {metrics['num_trips']}, "
                   f"Avg Distance: {metrics['avg_distance_km']:.1f} km, "
                   f"On-Time: {metrics['on_time_percentage']:.1f}%")
    
    # Get destination breakdown
    logger.info("\n=== Step 3: Destination Baseline Breakdown ===\n")
    by_dest = extractor.get_destination_baseline()
    for dest, metrics in list(by_dest.items())[:5]:  # Show first 5 destinations
        logger.info(f"{dest}:")
        logger.info(f"  Trips: {metrics['num_trips']}, "
                   f"Avg Distance: {metrics['avg_distance_km']:.1f} km, "
                   f"On-Time: {metrics['on_time_percentage']:.1f}%")
    
    # Simulate GA results
    logger.info("\n=== Step 4: Simulate GA Results for Comparison ===\n")
    ga_results = {
        'num_routes': 2,
        'makespan_s': 15480,  # ~4.3 hours
        'makespan_hours': 4.30,
        'total_time_s': 24600,  # ~6.8 hours
        'total_time_hours': 6.83,
        'total_distance_km': 340.4,
        'total_cost_idr': 482208,
    }
    logger.info(f"GA Routes: {ga_results['num_routes']}")
    logger.info(f"GA Makespan: {ga_results['makespan_hours']:.2f} hours")
    logger.info(f"GA Total Distance: {ga_results['total_distance_km']:.2f} km")
    logger.info(f"GA Total Cost: IDR {ga_results['total_cost_idr']:,.0f}\n")
    
    # Compare GA to baseline
    logger.info("=== Step 5: Generate Comparison Report ===\n")
    comparison = extractor.compare_ga_to_baseline(ga_results)
    
    logger.info(f"Distance Reduction: {comparison['improvements']['distance_reduction_km']:.2f} km "
               f"({comparison['improvements']['distance_reduction_pct']:.1f}%)")
    logger.info(f"Cost Reduction: IDR {comparison['improvements']['cost_reduction_idr']:,.0f} "
               f"({comparison['improvements']['cost_reduction_pct']:.1f}%)")
    
    # Generate reports
    logger.info("\n=== Step 6: Save Comparison Reports ===\n")
    reporter = ComparisonReporter(output_dir="results")
    
    # Text report
    text_report = reporter.generate_text_report(baseline, ga_results, comparison)
    logger.info(f"Text report saved to results/comparison_report.txt")
    
    # JSON report
    json_file = reporter.generate_json_report(baseline, ga_results, comparison)
    logger.info(f"JSON report saved to {json_file}")
    
    # Print console summary
    reporter.print_report(baseline, ga_results, comparison)
    summary_table = reporter.generate_summary_table(baseline, ga_results, comparison)
    print(summary_table)
    
    logger.info("="*80)
    logger.info("BASELINE EXTRACTION & COMPARISON COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
