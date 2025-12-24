"""
Batching strategy for grouping deliveries
"""
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict

from ..data.models import Delivery, PriorityLevel
from config.settings import BATCHING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeliveryBatcher:
    """Strategy for batching deliveries for efficient routing"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize delivery batcher
        
        Args:
            config: Batching configuration (uses default if None)
        """
        self.config = config or BATCHING_CONFIG
        logger.info("Initialized delivery batcher")
    
    def batch_by_priority(self, deliveries: List[Delivery]) -> Dict[PriorityLevel, List[Delivery]]:
        """
        Batch deliveries by priority level
        
        Args:
            deliveries: List of deliveries
            
        Returns:
            Dictionary mapping priority levels to delivery lists
        """
        batches = defaultdict(list)
        
        for delivery in deliveries:
            batches[delivery.priority].append(delivery)
        
        logger.info(f"Batched {len(deliveries)} deliveries by priority:")
        for priority, batch in batches.items():
            logger.info(f"  {priority.name}: {len(batch)} deliveries")
        
        return dict(batches)
    
    def batch_by_time_window(
        self, 
        deliveries: List[Delivery],
        window_minutes: int = None
    ) -> List[List[Delivery]]:
        """
        Batch deliveries by time windows
        
        Args:
            deliveries: List of deliveries
            window_minutes: Time window in minutes (uses config default if None)
            
        Returns:
            List of delivery batches
        """
        if window_minutes is None:
            window_minutes = self.config['time_window_minutes']
        
        # Sort deliveries by requested time
        sorted_deliveries = sorted(
            deliveries, 
            key=lambda d: d.requested_time if d.requested_time else datetime.now()
        )
        
        batches = []
        current_batch = []
        batch_start_time = None
        
        for delivery in sorted_deliveries:
            delivery_time = delivery.requested_time or datetime.now()
            
            if not current_batch:
                # Start new batch
                current_batch = [delivery]
                batch_start_time = delivery_time
            else:
                # Check if delivery fits in current time window
                time_diff = (delivery_time - batch_start_time).total_seconds() / 60
                
                if time_diff <= window_minutes and len(current_batch) < self.config['max_batch_size']:
                    current_batch.append(delivery)
                else:
                    # Start new batch
                    batches.append(current_batch)
                    current_batch = [delivery]
                    batch_start_time = delivery_time
        
        # Add last batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} time-based batches")
        return batches
    
    def batch_by_geographic_cluster(
        self, 
        deliveries: List[Delivery],
        num_clusters: int = None
    ) -> List[List[Delivery]]:
        """
        Batch deliveries by geographic clustering (simple proximity-based)
        
        Args:
            deliveries: List of deliveries
            num_clusters: Number of clusters (auto-calculated if None)
            
        Returns:
            List of delivery batches
        """
        # Filter deliveries with coordinates
        valid_deliveries = [
            d for d in deliveries 
            if d.destination.has_coordinates()
        ]
        
        if not valid_deliveries:
            logger.warning("No deliveries with coordinates for clustering")
            return [deliveries]
        
        # Simple clustering by latitude/longitude proximity
        if num_clusters is None:
            num_clusters = max(1, len(valid_deliveries) // self.config['max_batch_size'])
        
        # Sort by latitude then longitude for simple spatial grouping
        sorted_deliveries = sorted(
            valid_deliveries,
            key=lambda d: (d.destination.latitude or 0, d.destination.longitude or 0)
        )
        
        # Create batches
        batch_size = len(sorted_deliveries) // num_clusters + 1
        batches = [
            sorted_deliveries[i:i + batch_size]
            for i in range(0, len(sorted_deliveries), batch_size)
        ]
        
        logger.info(f"Created {len(batches)} geographic batches")
        return batches
    
    def create_optimized_batches(
        self, 
        deliveries: List[Delivery]
    ) -> List[List[Delivery]]:
        """
        Create optimized batches using combined strategies
        
        Strategy:
        1. Separate by priority
        2. Within each priority, batch by time window
        3. Further split large batches geographically
        
        Args:
            deliveries: List of deliveries
            
        Returns:
            List of optimized delivery batches
        """
        logger.info(f"Creating optimized batches for {len(deliveries)} deliveries")
        
        all_batches = []
        
        # Step 1: Batch by priority
        if self.config['priority_grouping']:
            priority_batches = self.batch_by_priority(deliveries)
            
            # Process each priority level separately
            for priority in sorted(priority_batches.keys(), key=lambda p: p.value):
                priority_deliveries = priority_batches[priority]
                
                # Step 2: Batch by time window
                time_batches = self.batch_by_time_window(priority_deliveries)
                
                # Step 3: Further split by geography if needed
                if self.config['geographic_clustering']:
                    for time_batch in time_batches:
                        if len(time_batch) > self.config['max_batch_size']:
                            geo_batches = self.batch_by_geographic_cluster(
                                time_batch,
                                num_clusters=len(time_batch) // self.config['max_batch_size'] + 1
                            )
                            all_batches.extend(geo_batches)
                        else:
                            all_batches.append(time_batch)
                else:
                    all_batches.extend(time_batches)
        else:
            # Simple batching without priority separation
            time_batches = self.batch_by_time_window(deliveries)
            all_batches = time_batches
        
        logger.info(f"Created {len(all_batches)} optimized batches:")
        for i, batch in enumerate(all_batches):
            priorities = [d.priority.name for d in batch]
            logger.info(f"  Batch {i+1}: {len(batch)} deliveries - Priorities: {set(priorities)}")
        
        return all_batches
    
    def get_batch_statistics(self, batches: List[List[Delivery]]) -> Dict:
        """
        Get statistics about batches
        
        Args:
            batches: List of delivery batches
            
        Returns:
            Dictionary with batch statistics
        """
        total_deliveries = sum(len(batch) for batch in batches)
        batch_sizes = [len(batch) for batch in batches]
        
        stats = {
            'num_batches': len(batches),
            'total_deliveries': total_deliveries,
            'avg_batch_size': total_deliveries / len(batches) if batches else 0,
            'min_batch_size': min(batch_sizes) if batch_sizes else 0,
            'max_batch_size': max(batch_sizes) if batch_sizes else 0,
            'priorities': defaultdict(int)
        }
        
        # Count deliveries by priority
        for batch in batches:
            for delivery in batch:
                stats['priorities'][delivery.priority.name] += 1
        
        return stats
