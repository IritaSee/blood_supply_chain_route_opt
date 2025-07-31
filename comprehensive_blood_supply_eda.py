"""
COMPREHENSIVE BLOOD SUPPLY CHAIN EDA - FINAL ANALYSIS
====================================================

This is the comprehensive Python script that performs complete EDA on the blood supply chain
data for route optimization research. It combines both basic and enhanced analysis to provide
deep insights for developing optimization algorithms.

Key Features:
- Multi-sheet analysis of all Excel data
- Blood type and component distribution analysis
- Temporal pattern analysis for delivery optimization
- Data quality assessment for algorithm readiness
- Route optimization recommendations
- Comprehensive visualization suite

Based on RESEARCH_CONTEXT.md guidelines for blood supply chain route optimization.

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
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Global configuration
FIGURES_DIR = Path('./figures')
FIGURES_DIR.mkdir(exist_ok=True)

class ComprehensiveBloodSupplyChainEDA:
    """
    Comprehensive EDA class combining all analysis capabilities for blood supply chain optimization
    """
    
    def __init__(self, file_path):
        """Initialize comprehensive EDA"""
        self.file_path = file_path
        self.all_data = {}
        self.sheet_names = None
        self.figures_saved = []
        self.insights = defaultdict(list)
        
    def save_figure(self, fig, filename, title=""):
        """Enhanced figure saving with metadata"""
        filepath = FIGURES_DIR / f"{filename}.png"
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        self.figures_saved.append(str(filepath))
        print(f"✓ Saved: {filepath}")
    
    def load_and_explore_data(self):
        """Load all sheets and perform initial exploration"""
        print("🔍 LOADING AND EXPLORING DATA")
        print("=" * 60)
        
        try:
            excel_file = pd.ExcelFile(self.file_path)
            self.sheet_names = excel_file.sheet_names
            
            print(f"📊 Found {len(self.sheet_names)} sheets:")
            for i, sheet in enumerate(self.sheet_names, 1):
                self.all_data[sheet] = pd.read_excel(self.file_path, sheet_name=sheet)
                print(f"  {i}. {sheet}: {self.all_data[sheet].shape[0]} rows × {self.all_data[sheet].shape[1]} cols")
                
                # Store insights about each sheet
                self.insights['data_structure'].append(f"{sheet}: {self.all_data[sheet].shape}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def comprehensive_data_overview(self):
        """Create comprehensive overview of all data"""
        print("\n🔍 COMPREHENSIVE DATA OVERVIEW")
        print("=" * 60)
        
        # Create 2x3 subplot for comprehensive overview
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # 1. Sheet sizes
        sheet_sizes = {sheet: data.shape[0] for sheet, data in self.all_data.items()}
        axes[0].bar(sheet_sizes.keys(), sheet_sizes.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0].set_title('Records per Sheet', fontweight='bold')
        axes[0].set_ylabel('Number of Records')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Column distribution
        sheet_cols = {sheet: data.shape[1] for sheet, data in self.all_data.items()}
        axes[1].bar(sheet_cols.keys(), sheet_cols.values(), 
                   color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        axes[1].set_title('Columns per Sheet', fontweight='bold')
        axes[1].set_ylabel('Number of Columns')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. Data quality (missing percentage)
        missing_percentages = {}
        for sheet, data in self.all_data.items():
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            missing_percentages[sheet] = (missing_cells / total_cells) * 100
        
        colors = ['red' if pct > 50 else 'orange' if pct > 20 else 'green' 
                 for pct in missing_percentages.values()]
        axes[2].bar(missing_percentages.keys(), missing_percentages.values(), color=colors)
        axes[2].set_title('Data Quality (Missing %)', fontweight='bold')
        axes[2].set_ylabel('Missing Data (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 4. Data types heatmap
        dtype_data = []
        sheet_names = []
        for sheet, data in self.all_data.items():
            dtype_counts = data.dtypes.value_counts()
            sheet_names.append(sheet)
            row = [dtype_counts.get('object', 0), dtype_counts.get('float64', 0), 
                   dtype_counts.get('int64', 0), dtype_counts.get('datetime64[ns]', 0)]
            dtype_data.append(row)
        
        dtype_df = pd.DataFrame(dtype_data, 
                               columns=['Object', 'Float', 'Integer', 'DateTime'],
                               index=sheet_names)
        sns.heatmap(dtype_df, annot=True, fmt='d', cmap='Blues', ax=axes[3])
        axes[3].set_title('Data Types Distribution', fontweight='bold')
        
        # 5. Timeline analysis (if date columns exist)
        date_ranges = {}
        for sheet, data in self.all_data.items():
            date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
            if date_cols:
                for col in date_cols:
                    valid_dates = data[col].dropna()
                    if len(valid_dates) > 0:
                        date_ranges[f"{sheet}_{col}"] = (valid_dates.min(), valid_dates.max())
        
        if date_ranges:
            y_pos = np.arange(len(date_ranges))
            start_dates = [dr[0] for dr in date_ranges.values()]
            end_dates = [dr[1] for dr in date_ranges.values()]
            durations = [(end - start).days for start, end in date_ranges.values()]
            
            axes[4].barh(y_pos, durations, color='skyblue')
            axes[4].set_yticks(y_pos)
            axes[4].set_yticklabels(list(date_ranges.keys()), fontsize=8)
            axes[4].set_title('Date Range Coverage (Days)', fontweight='bold')
            axes[4].set_xlabel('Duration (Days)')
        else:
            axes[4].text(0.5, 0.5, 'No Date Columns Found', 
                        ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('Date Range Analysis', fontweight='bold')
        
        # 6. Completeness score
        completeness_scores = {}
        for sheet, data in self.all_data.items():
            non_null_percentage = (1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            completeness_scores[sheet] = non_null_percentage
        
        axes[5].pie(completeness_scores.values(), labels=completeness_scores.keys(), 
                   autopct='%1.1f%%', startangle=90)
        axes[5].set_title('Data Completeness Distribution', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, "00_comprehensive_overview", 
                        "Comprehensive Blood Supply Chain Data Overview")
        plt.show()
    
    def blood_supply_chain_analysis(self):
        """Analyze blood supply chain specific metrics"""
        print("\n🩸 BLOOD SUPPLY CHAIN ANALYSIS")
        print("=" * 60)
        
        # Analyze blood types across all sheets
        blood_type_data = {}
        component_data = {}
        
        for sheet_name, data in self.all_data.items():
            # Look for blood type columns
            blood_cols = [col for col in data.columns if any(keyword in str(col).lower() 
                         for keyword in ['darah', 'blood', 'golongan', 'type'])]
            
            # Look for component columns
            component_cols = [col for col in data.columns if any(keyword in str(col).lower() 
                             for keyword in ['komponen', 'component', 'product'])]
            
            if blood_cols:
                for col in blood_cols:
                    if data[col].dtype == 'object' and data[col].notna().sum() > 0:
                        blood_type_data[f"{sheet_name}_{col}"] = data[col].value_counts()
            
            if component_cols:
                for col in component_cols:
                    if data[col].dtype == 'object' and data[col].notna().sum() > 0:
                        component_data[f"{sheet_name}_{col}"] = data[col].value_counts()
        
        # Create blood supply chain visualization
        n_plots = len(blood_type_data) + len(component_data)
        if n_plots > 0:
            n_rows = (n_plots + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            plot_idx = 0
            
            # Plot blood type distributions
            for name, distribution in blood_type_data.items():
                if plot_idx < len(axes):
                    distribution.head(10).plot(kind='bar', ax=axes[plot_idx], color='crimson', alpha=0.7)
                    axes[plot_idx].set_title(f'Blood Types: {name}', fontweight='bold')
                    axes[plot_idx].tick_params(axis='x', rotation=45)
                    plot_idx += 1
                    
                    # Store insights
                    self.insights['blood_types'].append(f"{name}: Most common is {distribution.index[0]} ({distribution.iloc[0]} units)")
            
            # Plot component distributions
            for name, distribution in component_data.items():
                if plot_idx < len(axes):
                    distribution.head(10).plot(kind='bar', ax=axes[plot_idx], color='steelblue', alpha=0.7)
                    axes[plot_idx].set_title(f'Components: {name}', fontweight='bold')
                    axes[plot_idx].tick_params(axis='x', rotation=45)
                    plot_idx += 1
                    
                    # Store insights
                    self.insights['components'].append(f"{name}: Primary component is {distribution.index[0]} ({distribution.iloc[0]} units)")
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.save_figure(fig, "15_blood_supply_chain_analysis", 
                           "Blood Types and Components Distribution Analysis")
            plt.show()
    
    def temporal_optimization_analysis(self):
        """Analyze temporal patterns for delivery optimization"""
        print("\n⏰ TEMPORAL OPTIMIZATION ANALYSIS")
        print("=" * 60)
        
        temporal_insights = []
        
        for sheet_name, data in self.all_data.items():
            date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
            
            if len(date_cols) > 0:
                print(f"\nAnalyzing temporal patterns in: {sheet_name}")
                
                # Create temporal analysis for this sheet
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                plot_idx = 0
                
                # Daily patterns
                for col in date_cols[:2]:
                    if plot_idx < 4:
                        valid_dates = data[col].dropna()
                        if len(valid_dates) > 0:
                            # Daily frequency
                            daily_counts = valid_dates.dt.date.value_counts().sort_index()
                            daily_counts.plot(ax=axes[plot_idx], kind='line', marker='o')
                            axes[plot_idx].set_title(f'Daily Pattern: {col}', fontweight='bold')
                            axes[plot_idx].tick_params(axis='x', rotation=45)
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
                            
                            # Store insights
                            peak_day = daily_counts.idxmax()
                            temporal_insights.append(f"{sheet_name} - {col}: Peak activity on {peak_day}")
                
                # Time gap analysis
                if len(date_cols) >= 2:
                    for i in range(min(2, len(date_cols)-1)):
                        if plot_idx < 4:
                            col1, col2 = date_cols[i], date_cols[i+1]
                            time_diff = data[col2] - data[col1]
                            time_diff_hours = time_diff.dt.total_seconds() / 3600
                            
                            # Remove extreme outliers
                            q1, q3 = time_diff_hours.quantile([0.25, 0.75])
                            iqr = q3 - q1
                            filtered_diff = time_diff_hours[
                                (time_diff_hours >= q1 - 1.5*iqr) & 
                                (time_diff_hours <= q3 + 1.5*iqr)
                            ]
                            
                            if len(filtered_diff) > 0:
                                filtered_diff.hist(bins=20, ax=axes[plot_idx], alpha=0.7, color='orange')
                                axes[plot_idx].set_title(f'Time Gap: {col1} → {col2}', fontweight='bold')
                                axes[plot_idx].set_xlabel('Hours')
                                axes[plot_idx].set_ylabel('Frequency')
                                
                                # Add statistics
                                mean_hours = filtered_diff.mean()
                                axes[plot_idx].axvline(mean_hours, color='red', linestyle='--', 
                                                      label=f'Mean: {mean_hours:.1f}h')
                                axes[plot_idx].legend()
                                plot_idx += 1
                                
                                # Store insights
                                temporal_insights.append(f"{sheet_name}: Average {col1}→{col2} time: {mean_hours:.1f} hours")
                
                # Hide unused subplots
                for i in range(plot_idx, 4):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                filename = f"16_temporal_analysis_{sheet_name.lower().replace(' ', '_').replace('&', 'and')}"
                self.save_figure(fig, filename, f"Temporal Analysis - {sheet_name}")
                plt.show()
        
        # Store temporal insights
        self.insights['temporal_patterns'] = temporal_insights
    
    def route_optimization_readiness(self):
        """Assess readiness for route optimization algorithms"""
        print("\n🚚 ROUTE OPTIMIZATION READINESS ASSESSMENT")
        print("=" * 60)
        
        assessment_criteria = {
            'Location Data': 0,
            'Demand Data': 0,
            'Supply Data': 0,
            'Time Constraints': 0,
            'Capacity Data': 0,
            'Quality Control': 0,
            'Vehicle Info': 0,
            'Priority Levels': 0
        }
        
        # Assess each criteria across all sheets
        for sheet_name, data in self.all_data.items():
            columns = [str(col).lower() for col in data.columns]
            
            # Location indicators
            if any(keyword in ' '.join(columns) for keyword in ['location', 'address', 'coordinate', 'lat', 'lng', 'gps']):
                assessment_criteria['Location Data'] += 1
            
            # Demand indicators
            if any(keyword in ' '.join(columns) for keyword in ['demand', 'need', 'request', 'order', 'requirement']):
                assessment_criteria['Demand Data'] += 1
            
            # Supply indicators
            if any(keyword in ' '.join(columns) for keyword in ['supply', 'stock', 'inventory', 'available']):
                assessment_criteria['Supply Data'] += 1
            
            # Time constraint indicators
            if any(keyword in ' '.join(columns) for keyword in ['time', 'date', 'schedule', 'window', 'deadline']):
                assessment_criteria['Time Constraints'] += 1
            
            # Capacity indicators
            if any(keyword in ' '.join(columns) for keyword in ['capacity', 'volume', 'weight', 'size', 'limit']):
                assessment_criteria['Capacity Data'] += 1
            
            # Quality control indicators
            if any(keyword in ' '.join(columns) for keyword in ['test', 'quality', 'check', 'validation', 'pemeriksaan']):
                assessment_criteria['Quality Control'] += 1
            
            # Vehicle information indicators
            if any(keyword in ' '.join(columns) for keyword in ['vehicle', 'truck', 'car', 'transport', 'driver']):
                assessment_criteria['Vehicle Info'] += 1
            
            # Priority level indicators
            if any(keyword in ' '.join(columns) for keyword in ['priority', 'urgent', 'emergency', 'critical']):
                assessment_criteria['Priority Levels'] += 1
        
        # Normalize scores (0-1 scale)
        max_score = len(self.all_data)
        normalized_scores = {k: v/max_score for k, v in assessment_criteria.items()}
        
        # Create assessment visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Radar chart for readiness assessment
        categories = list(normalized_scores.keys())
        values = list(normalized_scores.values())
        
        # Close the radar chart
        categories += [categories[0]]
        values += [values[0]]
        
        angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
        
        ax1.plot(angles, values, 'o-', linewidth=2, label='Current State', color='blue')
        ax1.fill(angles, values, alpha=0.25, color='blue')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories[:-1], fontsize=10)
        ax1.set_ylim(0, 1)
        ax1.set_title('Route Optimization Readiness\n(Radar Chart)', fontweight='bold', pad=20)
        ax1.grid(True)
        
        # Bar chart for detailed scores
        colors = ['green' if score > 0.6 else 'orange' if score > 0.3 else 'red' 
                 for score in normalized_scores.values()]
        bars = ax2.bar(range(len(normalized_scores)), list(normalized_scores.values()), color=colors)
        ax2.set_xticks(range(len(normalized_scores)))
        ax2.set_xticklabels(list(normalized_scores.keys()), rotation=45, ha='right')
        ax2.set_ylabel('Readiness Score (0-1)')
        ax2.set_title('Detailed Readiness Scores', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Add score labels on bars
        for bar, score in zip(bars, normalized_scores.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self.save_figure(fig, "17_optimization_readiness", 
                        "Route Optimization Readiness Assessment")
        plt.show()
        
        # Store readiness insights
        overall_readiness = np.mean(list(normalized_scores.values()))
        self.insights['optimization_readiness'] = [
            f"Overall readiness score: {overall_readiness:.2f}",
            f"Strongest area: {max(normalized_scores, key=normalized_scores.get)}",
            f"Weakest area: {min(normalized_scores, key=normalized_scores.get)}"
        ]
        
        return normalized_scores
    
    def generate_final_report(self):
        """Generate comprehensive final report with actionable insights"""
        print("\n📋 FINAL COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Calculate summary statistics
        total_records = sum(data.shape[0] for data in self.all_data.values())
        total_columns = sum(data.shape[1] for data in self.all_data.values())
        total_figures = len(self.figures_saved)
        
        print(f"📊 ANALYSIS SUMMARY")
        print(f"   • Sheets analyzed: {len(self.all_data)}")
        print(f"   • Total records: {total_records:,}")
        print(f"   • Total columns: {total_columns}")
        print(f"   • Figures generated: {total_figures}")
        
        print(f"\n🔍 KEY INSIGHTS")
        
        # Data structure insights
        if 'data_structure' in self.insights:
            print(f"   Data Structure:")
            for insight in self.insights['data_structure']:
                print(f"     - {insight}")
        
        # Blood type insights
        if 'blood_types' in self.insights:
            print(f"   Blood Types:")
            for insight in self.insights['blood_types'][:3]:  # Show top 3
                print(f"     - {insight}")
        
        # Component insights
        if 'components' in self.insights:
            print(f"   Components:")
            for insight in self.insights['components'][:3]:  # Show top 3
                print(f"     - {insight}")
        
        # Temporal insights
        if 'temporal_patterns' in self.insights:
            print(f"   Temporal Patterns:")
            for insight in self.insights['temporal_patterns'][:3]:  # Show top 3
                print(f"     - {insight}")
        
        # Optimization readiness
        if 'optimization_readiness' in self.insights:
            print(f"   Optimization Readiness:")
            for insight in self.insights['optimization_readiness']:
                print(f"     - {insight}")
        
        print(f"\n🚀 ROUTE OPTIMIZATION RECOMMENDATIONS")
        recommendations = [
            "1. 📍 LOCATION INFRASTRUCTURE: Implement GPS coordinates for all collection/delivery points",
            "2. 🚛 VEHICLE MANAGEMENT: Add vehicle capacity, type, and availability data",
            "3. ⏰ TIME OPTIMIZATION: Define delivery time windows and service times",
            "4. 📈 DEMAND FORECASTING: Develop predictive models using historical blood type patterns",
            "5. 🩸 INVENTORY BALANCING: Implement real-time stock level monitoring",
            "6. 🚨 EMERGENCY PROTOCOLS: Create priority-based routing for urgent deliveries",
            "7. 🌡️ COLD CHAIN MONITORING: Add temperature tracking for blood product safety",
            "8. 📱 REAL-TIME TRACKING: Implement GPS tracking and dynamic route adjustment",
            "9. 🔄 SUPPLY CHAIN VISIBILITY: Create dashboards for end-to-end monitoring",
            "10. 🤖 AI INTEGRATION: Develop machine learning models for optimization"
        ]
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n💡 NEXT STEPS FOR IMPLEMENTATION")
        next_steps = [
            "• Clean and standardize the dataset (address missing values)",
            "• Collect geographic coordinates for all locations",
            "• Integrate with real-time traffic and GPS data",
            "• Develop Vehicle Routing Problem (VRP) model",
            "• Implement optimization algorithms (GA, PSO, or MILP)",
            "• Create performance monitoring dashboard",
            "• Pilot test with a subset of routes",
            "• Scale to full blood supply chain network"
        ]
        
        for step in next_steps:
            print(f"   {step}")
        
        print(f"\n📁 Generated {total_figures} figures saved in: {FIGURES_DIR}")
        print("   All visualizations are ready for presentation and further analysis.")
        
        return {
            'total_records': total_records,
            'total_figures': total_figures,
            'insights': dict(self.insights),
            'recommendations': recommendations
        }
    
    def run_complete_analysis(self):
        """Execute the complete comprehensive analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE BLOOD SUPPLY CHAIN EDA")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_and_explore_data():
            return False
        
        # Step 2: Comprehensive overview
        self.comprehensive_data_overview()
        
        # Step 3: Blood supply chain specific analysis
        self.blood_supply_chain_analysis()
        
        # Step 4: Temporal optimization analysis
        self.temporal_optimization_analysis()
        
        # Step 5: Route optimization readiness
        self.route_optimization_readiness()
        
        # Step 6: Final report
        final_report = self.generate_final_report()
        
        return final_report

def main():
    """Main execution function"""
    print("🩸 BLOOD SUPPLY CHAIN ROUTE OPTIMIZATION - COMPREHENSIVE EDA")
    print("=" * 70)
    print("This analysis will help optimize blood delivery routes and logistics")
    print("=" * 70)
    
    # Initialize and run comprehensive analysis
    eda = ComprehensiveBloodSupplyChainEDA('All Droping.xlsx')
    result = eda.run_complete_analysis()
    
    if result:
        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 {result['total_figures']} visualizations created")
        print(f"📈 {result['total_records']:,} records analyzed")
        print(f"💡 Ready for route optimization algorithm development")
    else:
        print(f"\n❌ Analysis failed. Please check data file and try again.")

if __name__ == "__main__":
    main()
