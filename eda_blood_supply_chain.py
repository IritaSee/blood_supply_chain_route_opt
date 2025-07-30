"""
Exploratory Data Analysis for Blood Supply Chain Route Optimization
================================================================

This script performs comprehensive EDA on the 'All Droping.xlsx' file to understand
blood collection centers/drop-off points data for route optimization research.

Based on RESEARCH_CONTEXT.md, this analysis focuses on:
- Location analysis for route optimization
- Capacity and operational details
- Geographic distribution patterns
- Data quality assessment for optimization algorithms

Author: Blood Supply Chain Research Team
Date: July 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
figures_dir = Path('./figures')
figures_dir.mkdir(exist_ok=True)

class BloodSupplyChainEDA:
    """
    A comprehensive EDA class for blood supply chain data analysis
    """
    
    def __init__(self, file_path):
        """
        Initialize the EDA class with the Excel file path
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        self.data = None
        self.sheet_names = None
        self.figures_saved = []
        
    def load_data(self):
        """
        Load data from Excel file and explore its structure
        """
        print("=" * 60)
        print("BLOOD SUPPLY CHAIN ROUTE OPTIMIZATION - EDA")
        print("=" * 60)
        print(f"Loading data from: {self.file_path}")
        
        try:
            # Read Excel file and get sheet names
            excel_file = pd.ExcelFile(self.file_path)
            self.sheet_names = excel_file.sheet_names
            
            print(f"\nFound {len(self.sheet_names)} sheet(s):")
            for i, sheet in enumerate(self.sheet_names, 1):
                print(f"  {i}. {sheet}")
            
            # Load the first sheet or main data sheet
            self.data = pd.read_excel(self.file_path, sheet_name=self.sheet_names[0])
            print(f"\nLoaded data from sheet: '{self.sheet_names[0]}'")
            print(f"Dataset shape: {self.data.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def basic_info(self):
        """
        Display basic information about the dataset
        """
        print("\n" + "=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        print("\nColumn Data Types:")
        print(self.data.dtypes)
        
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nLast 5 rows:")
        print(self.data.tail())
        
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Missing values analysis
        print("\nMissing Values:")
        missing_vals = self.data.isnull().sum()
        missing_percent = (missing_vals / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_vals,
            'Missing Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
    
    def save_figure(self, fig, filename, title=""):
        """
        Save figure to the figures directory
        
        Args:
            fig: matplotlib figure object
            filename (str): filename for the saved figure
            title (str): optional title for the figure
        """
        filepath = figures_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        self.figures_saved.append(str(filepath))
        print(f"Saved: {filepath}")
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
    
    def missing_values_analysis(self):
        """
        Analyze and visualize missing values pattern
        """
        print("\n" + "=" * 50)
        print("MISSING VALUES ANALYSIS")
        print("=" * 50)
        
        # Missing values heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap of missing values
        sns.heatmap(self.data.isnull(), cbar=True, ax=ax1, cmap='viridis')
        ax1.set_title('Missing Values Heatmap')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Rows')
        
        # Bar plot of missing values percentage
        missing_percent = (self.data.isnull().sum() / len(self.data)) * 100
        missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
        
        if len(missing_percent) > 0:
            missing_percent.plot(kind='bar', ax=ax2)
            ax2.set_title('Missing Values Percentage by Column')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('Missing Percentage (%)')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Missing Values Found', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Missing Values Analysis')
        
        plt.tight_layout()
        self.save_figure(fig, "01_missing_values_analysis", 
                        "Missing Values Analysis for Blood Supply Chain Data")
        plt.show()
    
    def geographical_analysis(self):
        """
        Analyze geographical distribution if location data is available
        """
        print("\n" + "=" * 50)
        print("GEOGRAPHICAL DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Look for potential location columns
        location_keywords = ['lat', 'lng', 'longitude', 'latitude', 'coord', 'location', 
                           'address', 'city', 'region', 'province', 'state', 'area']
        
        location_cols = []
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in location_keywords):
                location_cols.append(col)
        
        if location_cols:
            print(f"Found potential location columns: {location_cols}")
            
            # Create subplots for location analysis
            n_cols = len(location_cols)
            n_rows = (n_cols + 2) // 3  # Arrange in rows of 3
            
            fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(15, 5*n_rows))
            if n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(location_cols):
                if i < len(axes):
                    if self.data[col].dtype in ['float64', 'int64']:
                        # Numerical location data - histogram
                        self.data[col].hist(bins=20, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                    else:
                        # Categorical location data - value counts
                        value_counts = self.data[col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=axes[i])
                        axes[i].set_title(f'Top 10 {col} Values')
                        axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(location_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.save_figure(fig, "02_geographical_distribution", 
                           "Geographical Distribution Analysis")
            plt.show()
            
            # If we have lat/lng, create a scatter plot
            lat_cols = [col for col in location_cols if 'lat' in col.lower()]
            lng_cols = [col for col in location_cols if 'lng' in col.lower() or 'lon' in col.lower()]
            
            if lat_cols and lng_cols:
                fig, ax = plt.subplots(figsize=(12, 8))
                self.data.plot.scatter(x=lng_cols[0], y=lat_cols[0], ax=ax, alpha=0.6)
                ax.set_title('Geographic Distribution of Blood Collection Points')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                
                self.save_figure(fig, "03_geographic_scatter_plot", 
                               "Geographic Scatter Plot of Collection Points")
                plt.show()
        else:
            print("No obvious location columns found in the dataset.")
    
    def capacity_analysis(self):
        """
        Analyze capacity-related metrics if available
        """
        print("\n" + "=" * 50)
        print("CAPACITY AND OPERATIONAL ANALYSIS")
        print("=" * 50)
        
        # Look for capacity/operational columns
        capacity_keywords = ['capacity', 'volume', 'amount', 'quantity', 'count', 
                           'supply', 'demand', 'inventory', 'stock']
        
        capacity_cols = []
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in capacity_keywords):
                if self.data[col].dtype in ['float64', 'int64']:
                    capacity_cols.append(col)
        
        if capacity_cols:
            print(f"Found potential capacity columns: {capacity_cols}")
            
            # Create comprehensive capacity analysis
            n_cols = len(capacity_cols)
            fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(15, 10))
            if n_cols == 1:
                axes = axes.reshape(-1)
            axes = axes.flatten()
            
            for i, col in enumerate(capacity_cols):
                if i < len(axes):
                    # Distribution plot
                    self.data[col].hist(bins=20, ax=axes[i], alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    
                    # Add statistics text
                    stats_text = f'Mean: {self.data[col].mean():.2f}\nStd: {self.data[col].std():.2f}'
                    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(len(capacity_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.save_figure(fig, "04_capacity_analysis", 
                           "Capacity and Operational Metrics Analysis")
            plt.show()
            
            # Correlation analysis if multiple capacity columns
            if len(capacity_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = self.data[capacity_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Matrix - Capacity Metrics')
                
                self.save_figure(fig, "05_capacity_correlation", 
                               "Capacity Metrics Correlation Analysis")
                plt.show()
        else:
            print("No obvious capacity/operational columns found.")
    
    def categorical_analysis(self):
        """
        Analyze categorical variables in the dataset
        """
        print("\n" + "=" * 50)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("=" * 50)
        
        # Identify categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            print(f"Found categorical columns: {categorical_cols}")
            
            # Analyze each categorical column
            for col in categorical_cols:
                print(f"\nAnalyzing column: {col}")
                value_counts = self.data[col].value_counts()
                print(f"Unique values: {self.data[col].nunique()}")
                print(f"Top 5 values:\n{value_counts.head()}")
                
                # Create visualization for top categories
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar plot of top 10 values
                top_values = value_counts.head(10)
                top_values.plot(kind='bar', ax=ax1)
                ax1.set_title(f'Top 10 Values - {col}')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                
                # Pie chart for top 5 values (if reasonable number of categories)
                if self.data[col].nunique() <= 20:
                    top_5 = value_counts.head(5)
                    if len(top_5) < len(value_counts):
                        others_count = value_counts.iloc[5:].sum()
                        top_5['Others'] = others_count
                    
                    top_5.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
                    ax2.set_title(f'Distribution - {col}')
                    ax2.set_ylabel('')
                else:
                    ax2.text(0.5, 0.5, f'Too many categories\n({self.data[col].nunique()} unique values)', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax2.transAxes, fontsize=12)
                    ax2.set_title(f'Category Count - {col}')
                
                plt.tight_layout()
                filename = f"06_categorical_{col.lower().replace(' ', '_')}"
                self.save_figure(fig, filename, f"Categorical Analysis - {col}")
                plt.show()
        else:
            print("No categorical columns found in the dataset.")
    
    def numerical_analysis(self):
        """
        Comprehensive analysis of numerical variables
        """
        print("\n" + "=" * 50)
        print("NUMERICAL VARIABLES ANALYSIS")
        print("=" * 50)
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            print(f"Found numerical columns: {numerical_cols}")
            
            # Overall distribution plots
            n_cols = len(numerical_cols)
            n_rows = (n_cols + 2) // 3
            
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    # Check if column has enough non-null values for analysis
                    non_null_data = self.data[col].dropna()
                    if len(non_null_data) > 1:
                        # Histogram with KDE
                        non_null_data.hist(bins=20, ax=axes[i], alpha=0.7, density=True)
                        try:
                            non_null_data.plot(kind='kde', ax=axes[i], color='red', linewidth=2)
                        except:
                            # Skip KDE if it fails
                            pass
                        axes[i].set_title(f'Distribution - {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Density')
                    else:
                        axes[i].text(0.5, 0.5, f'Insufficient data\n({len(non_null_data)} values)', 
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=axes[i].transAxes, fontsize=12)
                        axes[i].set_title(f'Distribution - {col}')
            
            # Hide unused subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.save_figure(fig, "07_numerical_distributions", 
                           "Numerical Variables Distribution Analysis")
            plt.show()
            
            # Box plots for outlier detection
            if len(numerical_cols) > 0:
                # Filter columns with sufficient data
                valid_cols = []
                for col in numerical_cols:
                    if self.data[col].dropna().nunique() > 1:
                        valid_cols.append(col)
                
                if valid_cols:
                    fig, ax = plt.subplots(figsize=(15, 8))
                    self.data[valid_cols].boxplot(ax=ax)
                    ax.set_title('Box Plots - Outlier Detection')
                    ax.set_xlabel('Variables')
                    ax.set_ylabel('Values')
                    plt.xticks(rotation=45)
                    
                    self.save_figure(fig, "08_numerical_boxplots", 
                                   "Box Plots for Outlier Detection")
                    plt.show()
                else:
                    print("No suitable numerical columns for box plot analysis.")
        else:
            print("No numerical columns found in the dataset.")
    
    def optimization_readiness_assessment(self):
        """
        Assess data readiness for route optimization algorithms
        """
        print("\n" + "=" * 50)
        print("ROUTE OPTIMIZATION READINESS ASSESSMENT")
        print("=" * 50)
        
        assessment = {
            'Location Data': 'Not Found',
            'Capacity Data': 'Not Found',
            'Demand Data': 'Not Found',
            'Time Data': 'Not Found',
            'Data Quality': 'Unknown'
        }
        
        # Check for location data
        location_keywords = ['lat', 'lng', 'longitude', 'latitude', 'coord', 'address']
        if any(keyword in str(col).lower() for col in self.data.columns for keyword in location_keywords):
            assessment['Location Data'] = 'Available'
        
        # Check for capacity data
        capacity_keywords = ['capacity', 'volume', 'supply', 'inventory']
        if any(keyword in str(col).lower() for col in self.data.columns for keyword in capacity_keywords):
            assessment['Capacity Data'] = 'Available'
        
        # Check for demand data
        demand_keywords = ['demand', 'requirement', 'need', 'request']
        if any(keyword in str(col).lower() for col in self.data.columns for keyword in demand_keywords):
            assessment['Demand Data'] = 'Available'
        
        # Check for time data
        time_keywords = ['time', 'date', 'hour', 'schedule', 'window']
        if any(keyword in str(col).lower() for col in self.data.columns for keyword in time_keywords):
            assessment['Time Data'] = 'Available'
        
        # Data quality assessment
        missing_percentage = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        if missing_percentage < 5:
            assessment['Data Quality'] = 'High (< 5% missing)'
        elif missing_percentage < 15:
            assessment['Data Quality'] = 'Medium (5-15% missing)'
        else:
            assessment['Data Quality'] = 'Low (> 15% missing)'
        
        print("Assessment Results:")
        for key, value in assessment.items():
            status = "✓" if value in ['Available', 'High (< 5% missing)'] else "✗" if value == 'Not Found' else "⚠"
            print(f"  {status} {key}: {value}")
        
        # Create assessment visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = list(assessment.keys())
        statuses = [1 if 'Available' in val or 'High' in val else 0.5 if 'Medium' in val else 0 for val in assessment.values()]
        colors = ['green' if s == 1 else 'orange' if s == 0.5 else 'red' for s in statuses]
        
        bars = ax.bar(categories, statuses, color=colors, alpha=0.7)
        ax.set_title('Route Optimization Data Readiness Assessment')
        ax.set_ylabel('Readiness Score')
        ax.set_ylim(0, 1.2)
        
        # Add text annotations
        for bar, status in zip(bars, assessment.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   status, ha='center', va='bottom', rotation=45, fontsize=8)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.save_figure(fig, "09_optimization_readiness", 
                        "Route Optimization Data Readiness Assessment")
        plt.show()
        
        return assessment
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EDA SUMMARY REPORT")
        print("=" * 60)
        
        print(f"Dataset: {self.file_path}")
        print(f"Analysis Date: July 30, 2025")
        print(f"Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        
        print(f"\nColumns: {list(self.data.columns)}")
        
        print(f"\nData Types:")
        print(self.data.dtypes.value_counts())
        
        print(f"\nMissing Data:")
        missing_count = self.data.isnull().sum().sum()
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_percentage = (missing_count / total_cells) * 100
        print(f"Total missing values: {missing_count} ({missing_percentage:.2f}%)")
        
        print(f"\nFigures Generated: {len(self.figures_saved)}")
        for fig_path in self.figures_saved:
            print(f"  - {fig_path}")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS FOR ROUTE OPTIMIZATION")
        print("=" * 60)
        
        recommendations = [
            "1. Verify and validate location data (coordinates) for accurate distance calculations",
            "2. Ensure capacity and demand data are in compatible units",
            "3. Add time window constraints for realistic route planning",
            "4. Consider adding vehicle capacity and type information",
            "5. Include blood type compatibility data for matching algorithms",
            "6. Add operational hours and service time requirements",
            "7. Consider adding priority levels for emergency vs. routine deliveries"
        ]
        
        for rec in recommendations:
            print(rec)
        
        print(f"\nTotal analysis completed. {len(self.figures_saved)} figures saved to './figures/' directory.")
    
    def run_complete_eda(self):
        """
        Run the complete EDA pipeline
        """
        if not self.load_data():
            return False
        
        self.basic_info()
        self.missing_values_analysis()
        self.geographical_analysis()
        self.capacity_analysis()
        self.categorical_analysis()
        self.numerical_analysis()
        self.optimization_readiness_assessment()
        self.generate_summary_report()
        
        return True

def main():
    """
    Main function to run the EDA
    """
    # Initialize EDA
    eda = BloodSupplyChainEDA('All Droping.xlsx')
    
    # Run complete analysis
    success = eda.run_complete_eda()
    
    if success:
        print(f"\n🎉 EDA completed successfully!")
        print(f"📊 Generated {len(eda.figures_saved)} visualizations")
        print(f"📁 All figures saved in: {figures_dir}")
    else:
        print("\n❌ EDA failed. Please check the data file and try again.")

if __name__ == "__main__":
    main()
