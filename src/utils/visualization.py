"""
Visualization utilities for blood supply chain routes
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List
import logging

from ..data.models import Route, OptimizationResult, Location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_routes(
    routes: List[Route],
    title: str = "Optimized Blood Supply Routes",
    save_path: str = None
):
    """
    Plot optimized routes on a map
    
    Args:
        routes: List of Route objects
        title: Plot title
        save_path: Path to save figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(range(len(routes)))
    
    for route_idx, route in enumerate(routes):
        color = colors[route_idx]
        
        # Get all locations in route
        locations = []
        for delivery in route.deliveries:
            if delivery.origin not in locations:
                locations.append(delivery.origin)
            if delivery.destination not in locations:
                locations.append(delivery.destination)
        
        # Filter locations with coordinates
        valid_locations = [loc for loc in locations if loc.has_coordinates()]
        
        if not valid_locations:
            logger.warning(f"Route {route.route_id} has no locations with coordinates")
            continue
        
        # Plot locations
        lats = [loc.latitude for loc in valid_locations]
        lons = [loc.longitude for loc in valid_locations]
        
        ax.scatter(lons, lats, c=[color], s=100, alpha=0.6, 
                  label=f"Route {route_idx+1} ({route.vehicle.vehicle_id})")
        
        # Connect locations in sequence
        if len(route.deliveries) > 0:
            for i, delivery in enumerate(route.deliveries):
                if delivery.origin.has_coordinates() and delivery.destination.has_coordinates():
                    ax.plot(
                        [delivery.origin.longitude, delivery.destination.longitude],
                        [delivery.origin.latitude, delivery.destination.latitude],
                        c=color, alpha=0.4, linestyle='--', linewidth=1
                    )
        
        # Annotate locations
        for loc in valid_locations[:5]:  # Limit annotations to avoid clutter
            ax.annotate(
                loc.name[:20], 
                (loc.longitude, loc.latitude),
                fontsize=7, alpha=0.7
            )
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved route visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_optimization_metrics(
    result: OptimizationResult,
    title: str = "Optimization Results",
    save_path: str = None
):
    """
    Plot optimization metrics
    
    Args:
        result: OptimizationResult object
        title: Plot title
        save_path: Path to save figure (if None, displays instead)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract route metrics
    route_ids = [r.route_id for r in result.routes]
    distances = [r.total_distance_km for r in result.routes]
    times = [r.total_time_hours for r in result.routes]
    costs = [r.total_cost_idr / 1000 for r in result.routes]  # in thousands IDR
    
    # 1. Distance per route
    axes[0, 0].bar(range(len(route_ids)), distances, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Route')
    axes[0, 0].set_ylabel('Distance (km)')
    axes[0, 0].set_title('Distance per Route', fontweight='bold')
    axes[0, 0].set_xticks(range(len(route_ids)))
    axes[0, 0].set_xticklabels([f"R{i+1}" for i in range(len(route_ids))])
    
    # 2. Time per route
    axes[0, 1].bar(range(len(route_ids)), times, color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('Route')
    axes[0, 1].set_ylabel('Time (hours)')
    axes[0, 1].set_title('Time per Route', fontweight='bold')
    axes[0, 1].set_xticks(range(len(route_ids)))
    axes[0, 1].set_xticklabels([f"R{i+1}" for i in range(len(route_ids))])
    
    # 3. Cost per route
    axes[1, 0].bar(range(len(route_ids)), costs, color='seagreen', alpha=0.7)
    axes[1, 0].set_xlabel('Route')
    axes[1, 0].set_ylabel('Cost (thousands IDR)')
    axes[1, 0].set_title('Cost per Route', fontweight='bold')
    axes[1, 0].set_xticks(range(len(route_ids)))
    axes[1, 0].set_xticklabels([f"R{i+1}" for i in range(len(route_ids))])
    
    # 4. Summary metrics
    summary_labels = ['Total Distance\n(km)', 'Total Time\n(hours)', 'Total Cost\n(M IDR)']
    summary_values = [
        result.total_distance_km,
        result.total_time_hours,
        result.total_cost_idr / 1000000  # in millions
    ]
    
    axes[1, 1].bar(range(3), summary_values, 
                  color=['steelblue', 'coral', 'seagreen'], alpha=0.7)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Overall Summary', fontweight='bold')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_xticklabels(summary_labels, fontsize=9)
    
    # Add value labels on bars
    for i, v in enumerate(summary_values):
        axes[1, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_report(result: OptimizationResult, output_path: str = None):
    """
    Generate a text report of optimization results
    
    Args:
        result: OptimizationResult object
        output_path: Path to save report (if None, prints to console)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BLOOD SUPPLY CHAIN ROUTE OPTIMIZATION - RESULTS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Routes: {len(result.routes)}")
    lines.append(f"Total Distance: {result.total_distance_km:.2f} km")
    lines.append(f"Total Time: {result.total_time_hours:.2f} hours")
    lines.append(f"Total Cost: IDR {result.total_cost_idr:,.2f}")
    lines.append(f"Fitness Score: {result.fitness_score:.4f}")
    lines.append(f"Generations: {result.generation}")
    lines.append("")
    
    lines.append("ROUTE DETAILS")
    lines.append("-" * 80)
    
    for i, route in enumerate(result.routes, 1):
        lines.append(f"\nRoute {i}: {route.route_id}")
        lines.append(f"  Vehicle: {route.vehicle.vehicle_id}")
        lines.append(f"  Deliveries: {len(route.deliveries)}")
        lines.append(f"  Distance: {route.total_distance_km:.2f} km")
        lines.append(f"  Time: {route.total_time_hours:.2f} hours")
        lines.append(f"  Cost: IDR {route.total_cost_idr:,.2f}")
        
        if route.deliveries:
            lines.append(f"  Delivery Sequence:")
            for j, delivery in enumerate(route.deliveries, 1):
                lines.append(f"    {j}. {delivery.delivery_id}: "
                           f"{delivery.origin.name} → {delivery.destination.name} "
                           f"(Priority: {delivery.priority.name})")
    
    lines.append("")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {output_path}")
    else:
        print(report)
    
    return report
