"""
Enhanced Multi-Sheet Analysis for Blood Supply Chain Route Optimization
=====================================================================

This script performs comprehensive EDA on ALL sheets in 'All Droping.xlsx' to understand
the complete blood supply chain data structure for route optimization research.

Enhanced analysis includes:
- Multi-sheet data exploration
- Time-based analysis for delivery optimization
- Blood component analysis for logistics planning
- Quality control metrics analysis

Author: Blood Supply Chain Research Team
Date: July 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
figures_dir = Path('./figures')
figures_dir.mkdir(exist_ok=True)

class EnhancedBloodSupplyChainEDA:
    """
    Enhanced EDA class for comprehensive blood supply chain multi-sheet analysis
    """
    
    def __init__(self, file_path):
        """
        Initialize the enhanced EDA class
        """
        self.file_path = file_path
        self.all_data = {}
        self.sheet_names = None
        self.figures_saved = []
        
    def load_all_sheets(self):
        """
        Load all sheets from the Excel file
        """
        print("=" * 60)
        print("ENHANCED BLOOD SUPPLY CHAIN ROUTE OPTIMIZATION - EDA")
        print("=" * 60)
        print(f"Loading all sheets from: {self.file_path}")
        
        try:
            excel_file = pd.ExcelFile(self.file_path)
            self.sheet_names = excel_file.sheet_names
            
            print(f"\nFound {len(self.sheet_names)} sheet(s):")
            for i, sheet in enumerate(self.sheet_names, 1):
                print(f"  {i}. {sheet}")
                self.all_data[sheet] = pd.read_excel(self.file_path, sheet_name=sheet)
                print(f"     Shape: {self.all_data[sheet].shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def save_figure(self, fig, filename, title=""):
        """Save figure with enhanced formatting"""
        filepath = figures_dir / f"{filename}.png"
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        self.figures_saved.append(str(filepath))
        print(f"Saved: {filepath}")
    
    def analyze_all_sheets_overview(self):
        """
        Create overview analysis of all sheets
        """
        print("\n" + "=" * 50)
        print("ALL SHEETS OVERVIEW ANALYSIS")
        print("=" * 50)
        
        # Create summary comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sheet sizes comparison
        sheet_sizes = {sheet: data.shape[0] for sheet, data in self.all_data.items()}
        ax1.bar(sheet_sizes.keys(), sheet_sizes.values(), color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Number of Records per Sheet')
        ax1.set_ylabel('Number of Records')
        ax1.tick_params(axis='x', rotation=45)
        
        # Column counts comparison
        sheet_cols = {sheet: data.shape[1] for sheet, data in self.all_data.items()}
        ax2.bar(sheet_cols.keys(), sheet_cols.values(), color=['orange', 'purple', 'gold'])
        ax2.set_title('Number of Columns per Sheet')
        ax2.set_ylabel('Number of Columns')
        ax2.tick_params(axis='x', rotation=45)
        
        # Missing data percentage
        missing_percentages = {}
        for sheet, data in self.all_data.items():
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            missing_percentages[sheet] = (missing_cells / total_cells) * 100
        
        ax3.bar(missing_percentages.keys(), missing_percentages.values(), color=['red', 'orange', 'yellow'])
        ax3.set_title('Missing Data Percentage per Sheet')
        ax3.set_ylabel('Missing Data (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Data types distribution
        dtype_summary = {}
        for sheet, data in self.all_data.items():
            dtype_counts = data.dtypes.value_counts()
            dtype_summary[sheet] = dtype_counts
        
        # Create stacked bar for data types
        all_dtypes = set()
        for counts in dtype_summary.values():
            all_dtypes.update(counts.index)
        
        dtype_matrix = pd.DataFrame(0, index=list(dtype_summary.keys()), columns=list(all_dtypes))
        for sheet, counts in dtype_summary.items():
            for dtype, count in counts.items():
                dtype_matrix.loc[sheet, dtype] = count
        
        dtype_matrix.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Data Types Distribution per Sheet')
        ax4.set_ylabel('Number of Columns')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self.save_figure(fig, "10_all_sheets_overview", "Complete Data Overview - All Sheets")
        #plt.show()
        
        # Print detailed summary
        for sheet_name, data in self.all_data.items():
            print(f"\n--- {sheet_name} ---")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            if data.shape[0] > 0:
                print(f"Sample data:\n{data.head(2)}")
    
    def analyze_blood_types_and_components(self):
        """
        Analyze blood types and components across sheets
        """
        print("\n" + "=" * 50)
        print("BLOOD TYPES AND COMPONENTS ANALYSIS")
        print("=" * 50)
        
        # Look for blood type data in all sheets
        blood_data = {}
        for sheet_name, data in self.all_data.items():
            blood_cols = [col for col in data.columns if any(keyword in col.lower() 
                         for keyword in ['darah', 'blood', 'golongan', 'type', 'komponen', 'component'])]
            if blood_cols:
                blood_data[sheet_name] = data[blood_cols]
        
        if blood_data:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            plot_idx = 0
            
            for sheet_name, data in blood_data.items():
                for col in data.columns:
                    if plot_idx < 4 and data[col].dtype == 'object':
                        value_counts = data[col].value_counts().head(10)
                        if len(value_counts) > 0:
                            value_counts.plot(kind='bar', ax=axes[plot_idx])
                            axes[plot_idx].set_title(f'{sheet_name} - {col}')
                            axes[plot_idx].tick_params(axis='x', rotation=45)
                            plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.save_figure(fig, "11_blood_types_components", "Blood Types and Components Analysis")
            #plt.show()
    
    def analyze_temporal_patterns(self):
        """
        Analyze temporal patterns in the data for delivery optimization
        """
        print("\n" + "=" * 50)
        print("TEMPORAL PATTERNS ANALYSIS")
        print("=" * 50)
        
        # Look for date/time columns across all sheets
        temporal_data = {}
        for sheet_name, data in self.all_data.items():
            date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
            if date_cols:
                temporal_data[sheet_name] = data[date_cols]
        
        if temporal_data:
            for sheet_name, data in temporal_data.items():
                if data.shape[1] > 0:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    axes = axes.flatten()
                    
                    plot_idx = 0
                    for col in data.columns[:4]:  # Analyze up to 4 date columns
                        if plot_idx < 4:
                            # Time series plot
                            valid_dates = data[col].dropna()
                            if len(valid_dates) > 0:
                                # Daily frequency
                                daily_counts = valid_dates.dt.date.value_counts().sort_index()
                                daily_counts.plot(ax=axes[plot_idx])
                                axes[plot_idx].set_title(f'{sheet_name} - {col} (Daily Pattern)')
                                axes[plot_idx].tick_params(axis='x', rotation=45)
                                plot_idx += 1
                    
                    # Hide unused subplots
                    for i in range(plot_idx, 4):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    filename = f"12_temporal_patterns_{sheet_name.lower().replace(' ', '_')}"
                    self.save_figure(fig, filename, f"Temporal Patterns - {sheet_name}")
                    #plt.show()
                    
                    # Analyze time gaps and patterns
                    self.analyze_time_gaps(sheet_name, data)
    
    def analyze_time_gaps(self, sheet_name, temporal_data):
        """
        Analyze time gaps between processes for delivery optimization
        """
        print(f"\nAnalyzing time gaps for {sheet_name}...")
        
        if temporal_data.shape[1] >= 2:
            # Calculate time differences between consecutive date columns
            cols = temporal_data.columns.tolist()
            
            fig, axes = plt.subplots(1, min(2, len(cols)-1), figsize=(15, 6))
            if len(cols) == 2:
                axes = [axes]
            
            for i in range(min(2, len(cols)-1)):
                col1, col2 = cols[i], cols[i+1]
                time_diff = temporal_data[col2] - temporal_data[col1]
                time_diff_hours = time_diff.dt.total_seconds() / 3600  # Convert to hours
                
                # Remove outliers for better visualization
                q1 = time_diff_hours.quantile(0.25)
                q3 = time_diff_hours.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_diff = time_diff_hours[(time_diff_hours >= lower_bound) & (time_diff_hours <= upper_bound)]
                
                if len(filtered_diff) > 0:
                    filtered_diff.hist(bins=20, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Time Gap: {col1} to {col2}')
                    axes[i].set_xlabel('Hours')
                    axes[i].set_ylabel('Frequency')
                    
                    # Add statistics
                    mean_hours = filtered_diff.mean()
                    median_hours = filtered_diff.median()
                    axes[i].axvline(mean_hours, color='red', linestyle='--', label=f'Mean: {mean_hours:.1f}h')
                    axes[i].axvline(median_hours, color='green', linestyle='--', label=f'Median: {median_hours:.1f}h')
                    axes[i].legend()
            
            plt.tight_layout()
            filename = f"13_time_gaps_{sheet_name.lower().replace(' ', '_')}"
            self.save_figure(fig, filename, f"Time Gaps Analysis - {sheet_name}")
            #plt.show()
    
    def analyze_delivery_delays(self):
        """
        Specific analysis for delivery delays if data is available
        """
        print("\n" + "=" * 50)
        print("DELIVERY DELAYS ANALYSIS")
        print("=" * 50)
        
        # Look for delay-related sheets
        delay_sheets = [sheet for sheet in self.sheet_names if any(keyword in sheet.lower() 
                       for keyword in ['delay', 'keterlambatan', 'waktu', 'time'])]
        
        if delay_sheets:
            for sheet_name in delay_sheets:
                data = self.all_data[sheet_name]
                print(f"\nAnalyzing delays in sheet: {sheet_name}")
                print(f"Shape: {data.shape}")
                print(f"Columns: {list(data.columns)}")
                
                if data.shape[0] > 0:
                    # Create comprehensive delay analysis
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    axes = axes.flatten()
                    
                    # Analyze numerical columns for delay metrics
                    numerical_cols = data.select_dtypes(include=[np.number]).columns
                    categorical_cols = data.select_dtypes(include=['object']).columns
                    
                    plot_idx = 0
                    
                    # Plot numerical delay metrics
                    for col in numerical_cols[:2]:
                        if plot_idx < 4:
                            data[col].hist(bins=20, ax=axes[plot_idx], alpha=0.7)
                            axes[plot_idx].set_title(f'Distribution: {col}')
                            axes[plot_idx].set_xlabel(col)
                            axes[plot_idx].set_ylabel('Frequency')
                            plot_idx += 1
                    
                    # Plot categorical delay factors
                    for col in categorical_cols[:2]:
                        if plot_idx < 4:
                            value_counts = data[col].value_counts().head(10)
                            if len(value_counts) > 0:
                                value_counts.plot(kind='bar', ax=axes[plot_idx])
                                axes[plot_idx].set_title(f'Delay Factors: {col}')
                                axes[plot_idx].tick_params(axis='x', rotation=45)
                                plot_idx += 1
                    
                    # Hide unused subplots
                    for i in range(plot_idx, 4):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    filename = f"14_delivery_delays_{sheet_name.lower().replace(' ', '_')}"
                    self.save_figure(fig, filename, f"Delivery Delays Analysis - {sheet_name}")
                    #plt.show()
        else:
            print("No delay-specific sheets found.")
    
    def route_optimization_insights(self):
        """
        Generate specific insights for route optimization
        """
        print("\n" + "=" * 50)
        print("ROUTE OPTIMIZATION INSIGHTS")
        print("=" * 50)
        
        insights = []
        
        # Analyze blood type distribution for demand planning
        for sheet_name, data in self.all_data.items():
            blood_type_cols = [col for col in data.columns if 'darah' in col.lower() or 'blood' in col.lower()]
            if blood_type_cols:
                for col in blood_type_cols:
                    if data[col].dtype == 'object':
                        distribution = data[col].value_counts()
                        insights.append(f"Sheet '{sheet_name}': {col} distribution shows {distribution.index[0]} is most common ({distribution.iloc[0]} occurrences)")
        
        # Analyze component types for logistics planning
        for sheet_name, data in self.all_data.items():
            component_cols = [col for col in data.columns if 'komponen' in col.lower() or 'component' in col.lower()]
            if component_cols:
                for col in component_cols:
                    if data[col].dtype == 'object':
                        distribution = data[col].value_counts()
                        insights.append(f"Component analysis in '{sheet_name}': {distribution.index[0]} is primary component ({distribution.iloc[0]} units)")
        
        # Time-based insights
        for sheet_name, data in self.all_data.items():
            date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
            if len(date_cols) >= 2:
                # Calculate average processing time
                time_diff = data[date_cols[1]] - data[date_cols[0]]
                avg_hours = time_diff.dt.total_seconds().mean() / 3600
                insights.append(f"Average processing time in '{sheet_name}': {avg_hours:.1f} hours between {date_cols[0]} and {date_cols[1]}")
        
        print("\nKey Insights for Route Optimization:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
    
    def generate_enhanced_summary(self):
        """
        Generate comprehensive summary with optimization recommendations
        """
        print("\n" + "=" * 60)
        print("ENHANCED COMPREHENSIVE SUMMARY REPORT")
        print("=" * 60)
        
        total_records = sum(data.shape[0] for data in self.all_data.values())
        total_columns = sum(data.shape[1] for data in self.all_data.values())
        
        print(f"Total Sheets Analyzed: {len(self.all_data)}")
        print(f"Total Records: {total_records}")
        print(f"Total Columns: {total_columns}")
        print(f"Figures Generated: {len(self.figures_saved)}")
        
        print(f"\nSheet Details:")
        for sheet_name, data in self.all_data.items():
            print(f"  {sheet_name}: {data.shape[0]} rows × {data.shape[1]} columns")
        
        print(f"\nData Quality Assessment:")
        for sheet_name, data in self.all_data.items():
            missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            quality = "High" if missing_pct < 5 else "Medium" if missing_pct < 15 else "Low"
            print(f"  {sheet_name}: {quality} quality ({missing_pct:.1f}% missing)")
        
        print(f"\n" + "=" * 60)
        print("ROUTE OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = [
            "1. LOCATION DATA: Add geographic coordinates for all collection and delivery points",
            "2. VEHICLE CAPACITY: Include vehicle specifications and capacity constraints",
            "3. TIME WINDOWS: Define delivery time windows for each destination",
            "4. DEMAND FORECASTING: Develop demand prediction models based on historical patterns",
            "5. PRIORITY LEVELS: Implement urgency levels for different blood types/components",
            "6. ROUTE CONSTRAINTS: Consider traffic patterns, road restrictions, and delivery schedules",
            "7. COLD CHAIN: Include temperature monitoring and storage requirements",
            "8. REAL-TIME TRACKING: Implement GPS tracking for dynamic route optimization",
            "9. INVENTORY OPTIMIZATION: Balance stock levels across multiple locations",
            "10. EMERGENCY PROTOCOLS: Develop fast-track routing for emergency situations"
        ]
        
        for rec in recommendations:
            print(rec)
        
        print(f"\nAll {len(self.figures_saved)} figures saved to './figures/' directory.")
    
    def run_enhanced_analysis(self):
        """
        Run the complete enhanced analysis
        """
        if not self.load_all_sheets():
            return False
        
        self.analyze_all_sheets_overview()
        self.analyze_blood_types_and_components()
        self.analyze_temporal_patterns()
        self.analyze_delivery_delays()
        self.route_optimization_insights()
        self.generate_enhanced_summary()
        
        return True

def main():
    """
    Main function to run the enhanced EDA
    """
    print("Starting Enhanced Blood Supply Chain EDA...")
    
    # Initialize enhanced EDA
    eda = EnhancedBloodSupplyChainEDA('All Droping.xlsx')
    
    # Run complete enhanced analysis
    success = eda.run_enhanced_analysis()
    
    if success:
        print(f"\n🎉 Enhanced EDA completed successfully!")
        print(f"📊 Generated {len(eda.figures_saved)} comprehensive visualizations")
        print(f"📁 All figures saved in: {figures_dir}")
        print(f"🔍 Multi-sheet analysis complete with route optimization insights")
    else:
        print("\n❌ Enhanced EDA failed. Please check the data file and try again.")

if __name__ == "__main__":
    main()
