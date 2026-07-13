# Graph Report - .  (2026-07-14)

## Corpus Check
- 80 files · ~160,958 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 752 nodes · 1124 edges · 58 communities (46 shown, 12 thin omitted)
- Extraction: 94% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.72)
- Token cost: 1,229,105 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Research Architecture and Terminology|Research Architecture and Terminology]]
- [[_COMMUNITY_Thesis Background and GA Approach|Thesis Background and GA Approach]]
- [[_COMMUNITY_Config and Domain Models|Config and Domain Models]]
- [[_COMMUNITY_OSRM Routing Engine|OSRM Routing Engine]]
- [[_COMMUNITY_GA Chromosome Model|GA Chromosome Model]]
- [[_COMMUNITY_Data Loader Module|Data Loader Module]]
- [[_COMMUNITY_Basic EDA Module|Basic EDA Module]]
- [[_COMMUNITY_Enhanced EDA Module|Enhanced EDA Module]]
- [[_COMMUNITY_CNN Delivery Time Model|CNN Delivery Time Model]]
- [[_COMMUNITY_DL Preprocessing Pipeline|DL Preprocessing Pipeline]]
- [[_COMMUNITY_Data Extraction Pipeline|Data Extraction Pipeline]]
- [[_COMMUNITY_Baseline Metrics Extractor|Baseline Metrics Extractor]]
- [[_COMMUNITY_Geocoder and Caching|Geocoder and Caching]]
- [[_COMMUNITY_Comprehensive EDA Module|Comprehensive EDA Module]]
- [[_COMMUNITY_Comparison Chart Generation|Comparison Chart Generation]]
- [[_COMMUNITY_Sample Data Loader|Sample Data Loader]]
- [[_COMMUNITY_Routing and Geocoding Utils|Routing and Geocoding Utils]]
- [[_COMMUNITY_CNN Model IO|CNN Model I/O]]
- [[_COMMUNITY_Optimization Utilities and Route Model|Optimization Utilities and Route Model]]
- [[_COMMUNITY_DEAP GA Route Optimizer|DEAP GA Route Optimizer]]
- [[_COMMUNITY_GA Data Extraction and Optimization|GA Data Extraction and Optimization]]
- [[_COMMUNITY_Sheet Overview Dashboard|Sheet Overview Dashboard]]
- [[_COMMUNITY_Comparison Reporting|Comparison Reporting]]
- [[_COMMUNITY_Optimization Pipeline Orchestrator|Optimization Pipeline Orchestrator]]
- [[_COMMUNITY_GA Candidate Generation|GA Candidate Generation]]
- [[_COMMUNITY_Route Optimization Readiness v1|Route Optimization Readiness v1]]
- [[_COMMUNITY_Geocoding Utilities|Geocoding Utilities]]
- [[_COMMUNITY_Demand and Facility Extraction|Demand and Facility Extraction]]
- [[_COMMUNITY_CNN Training Callbacks|CNN Training Callbacks]]
- [[_COMMUNITY_CNN Feature Engineering|CNN Feature Engineering]]
- [[_COMMUNITY_Dataclass Serialization|Dataclass Serialization]]
- [[_COMMUNITY_Blood Type Distribution|Blood Type Distribution]]
- [[_COMMUNITY_RSSA Temporal Shelf-Life Analysis|RSSA Temporal Shelf-Life Analysis]]
- [[_COMMUNITY_Route Readiness Assessment v2|Route Readiness Assessment v2]]
- [[_COMMUNITY_RSSA Delivery Temporal Patterns|RSSA Delivery Temporal Patterns]]
- [[_COMMUNITY_RSSA Time Gap Analysis|RSSA Time Gap Analysis]]
- [[_COMMUNITY_Delivery Delay Analysis|Delivery Delay Analysis]]
- [[_COMMUNITY_DL Route Selector Init|DL Route Selector Init]]
- [[_COMMUNITY_Blood Component Distribution|Blood Component Distribution]]
- [[_COMMUNITY_All-Sheets Data Overview|All-Sheets Data Overview]]
- [[_COMMUNITY_Blood Type and Component Analysis|Blood Type and Component Analysis]]
- [[_COMMUNITY_RSSA Dataset Distribution|RSSA Dataset Distribution]]
- [[_COMMUNITY_Vehicle Route Model|Vehicle Route Model]]
- [[_COMMUNITY_Missing Values Analysis|Missing Values Analysis]]
- [[_COMMUNITY_Numerical Distribution Analysis|Numerical Distribution Analysis]]
- [[_COMMUNITY_HBsAg Screening Result|HBsAg Screening Result]]
- [[_COMMUNITY_Outlier Detection Boxplots|Outlier Detection Boxplots]]
- [[_COMMUNITY_Rh Factor Distribution|Rh Factor Distribution]]
- [[_COMMUNITY_DL Predictor Package Init|DL Predictor Package Init]]
- [[_COMMUNITY_Blood Bag ID Distribution|Blood Bag ID Distribution]]
- [[_COMMUNITY_GA Package Init|GA Package Init]]
- [[_COMMUNITY_Route Optimization Package Init|Route Optimization Package Init]]
- [[_COMMUNITY_Examination Result Distribution|Examination Result Distribution]]
- [[_COMMUNITY_HCV Screening Result|HCV Screening Result]]
- [[_COMMUNITY_Syphilis Screening Result|Syphilis Screening Result]]
- [[_COMMUNITY_Unnamed Column 13 Distribution|Unnamed Column 13 Distribution]]
- [[_COMMUNITY_Unnamed Column 14 Distribution|Unnamed Column 14 Distribution]]

## God Nodes (most connected - your core abstractions)
1. `DeliveryTimeCNN` - 29 edges
2. `Location` - 25 edges
3. `OptimizationPipeline` - 21 edges
4. `Delivery` - 21 edges
5. `TripDataPreprocessor` - 20 edges
6. `DataLoader` - 17 edges
7. `BaselineExtractor` - 16 edges
8. `GeneticAlgorithm` - 15 edges
9. `BloodSupplyChainEDA` - 15 edges
10. `OSRMRouter` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Algoritma Genetika (thesis GA method)` --semantically_similar_to--> `Approach A: Genetic Algorithm (GA)`  [INFERRED] [semantically similar]
  Draf Proposal.pdf → .github/copilot-instructions.md
- `Artificial Intelligence / Machine Learning / Deep Learning (thesis)` --semantically_similar_to--> `Approach B: Deep Learning (DL) Route Selector`  [INFERRED] [semantically similar]
  Draf Proposal.pdf → .github/copilot-instructions.md
- `Domain Terminology (BDRS, UTD, Sequence, Lateness)` --semantically_similar_to--> `Bank Darah Rumah Sakit (BDRS)`  [INFERRED] [semantically similar]
  .github/copilot-instructions.md → Draf Proposal.pdf
- `Domain Terminology (BDRS, UTD, Sequence, Lateness)` --semantically_similar_to--> `Unit Transfusi Darah (UTD)`  [INFERRED] [semantically similar]
  .github/copilot-instructions.md → Draf Proposal.pdf
- `Multi-objective Fitness Function (70% time / 30% cost)` --semantically_similar_to--> `Algoritma Genetika (thesis GA method)`  [INFERRED] [semantically similar]
  IMPLEMENTATION.md → Draf Proposal.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Documents describing the Parallel Competitive Model (GA vs DL vs Baseline)** — _github_copilot_instructions_parallelcompetitivemodel, parallel_competitive_model_architecture, implementation_summary_root, quickstart_root, comparison_report_root [INFERRED 0.85]
- **Deep Learning time-prediction pipeline components** — dl_predictor_data_preprocessor_module, dl_predictor_cnn_model_module, dl_predictor_train_module, config_dl_config_module, dl_predictor_route_selector_module [EXTRACTED 0.95]
- **Two conflicting GA implementation lineages described across docs (ga_optimizer/ custom-no-DEAP vs src/ DEAP-based)** — readme_gaoptimizerimpl, readme_srcimpl, implementation_gaengine, ga_optimizer_genetic_algorithm_module, src_optimization_genetic_algorithm_module [INFERRED 0.75]

## Communities (58 total, 12 thin omitted)

### Community 0 - "Research Architecture and Terminology"
Cohesion: 0.08
Nodes (45): DeepLearningRouteSelector class spec, Approach B: Deep Learning (DL) Route Selector, Parallel Competitive Model (research architecture), GitHub Copilot Instructions for Blood Supply Chain Optimization, Domain Terminology (BDRS, UTD, Sequence, Lateness), Historical Baseline Metrics (287 trips, 25.4% on-time), Genetic Algorithm Results (2 routes, 340.4km, IDR 482,208), Baseline vs GA Comparison Report (2025-12-25) (+37 more)

### Community 1 - "Thesis Background and GA Approach"
Cohesion: 0.06
Nodes (40): Approach A: Genetic Algorithm (GA), config/settings.py, demo_ga_optimization.py, Import Conflict Resolution (ga_optimizer.geocoder vs src.routing.geocoding), Yudha Prambudia, S.T. M.Sc. Ph.D (advisor), Risma Nurizza Saputri (thesis author), Blood Supply Chain (BSC), Capacitated Vehicle Routing Problem (CVRP) (+32 more)

### Community 2 - "Config and Domain Models"
Cohesion: 0.09
Nodes (22): Blood Supply Chain Route Optimization - Configuration Module, Configuration file for Blood Supply Chain Route Optimization, Enum, Simple example demonstrating the blood supply chain route optimization, Delivery, PriorityLevel, Data models for Blood Supply Chain Route Optimization, Priority levels for blood deliveries (+14 more)

### Community 3 - "OSRM Routing Engine"
Cohesion: 0.07
Nodes (19): OSRMRouter, ndarray, OSRM routing module with batching, caching, and rate limiting. Computes distance, Cache distance/duration for a location pair., OSRM routing with batching, caching, and fallback., Initialize OSRM router.                  Args:             cache_file: SQLite ca, Enforce rate limiting (max 1 req/sec)., Calculate haversine distance in meters. (+11 more)

### Community 4 - "GA Chromosome Model"
Cohesion: 0.09
Nodes (17): Chromosome, ndarray, Get list of routes (customer sequences per vehicle)., Set routes and update assignments., Create a copy of this chromosome., Initialize GA.                  Args:             num_customers: Number of deliv, Evaluate fitness of a chromosome.                  Lexicographic objective:, Order crossover (OX) between two parents. (+9 more)

### Community 5 - "Data Loader Module"
Cohesion: 0.11
Nodes (25): Blood Supply Chain Route Optimization - Data Module, Data loader for Blood Supply Chain Route Optimization Reads Excel files and crea, BloodProduct, Location, Represents a location (hospital, blood bank, etc.), Check if location has valid coordinates, Convert to dictionary, Represents a blood product (+17 more)

### Community 6 - "Basic EDA Module"
Cohesion: 0.11
Nodes (17): BloodSupplyChainEDA, main(), Exploratory Data Analysis for Blood Supply Chain Route Optimization ============, Save figure to the figures directory                  Args:             fig: mat, Analyze and visualize missing values pattern, Analyze geographical distribution if location data is available, Analyze capacity-related metrics if available, Analyze categorical variables in the dataset (+9 more)

### Community 7 - "Enhanced EDA Module"
Cohesion: 0.11
Nodes (16): EnhancedBloodSupplyChainEDA, main(), Enhanced Multi-Sheet Analysis for Blood Supply Chain Route Optimization ========, Analyze blood types and components across sheets, Analyze temporal patterns in the data for delivery optimization, Analyze time gaps between processes for delivery optimization, Specific analysis for delivery delays if data is available, Generate specific insights for route optimization (+8 more)

### Community 8 - "CNN Delivery Time Model"
Cohesion: 0.08
Nodes (15): DeliveryTimeCNN, Callback, History, Model, ndarray, 1D CNN model for delivery time prediction., Create training callbacks., Train the model.                  Args:             X_train: Training features ( (+7 more)

### Community 9 - "DL Preprocessing Pipeline"
Cohesion: 0.12
Nodes (18): Deep Learning predictor configuration. 1D CNN for delivery time estimation., 1D CNN model for delivery time prediction., Data preprocessing for deep learning time predictor., Preprocess historical trip data for 1D CNN training., Initialize preprocessor., TripDataPreprocessor, Training script for 1D CNN delivery time predictor., Train 1D CNN model for delivery time prediction.          Args:         data_fil (+10 more)

### Community 10 - "Data Extraction Pipeline"
Cohesion: 0.10
Nodes (15): DataExtractor, Extract Malang blood supply chain data from Excel files., Initialize data extractor., OptimizationPipeline, DataFrame, Extract baseline metrics from historical trip data., Train or load DL time predictor., Save optimization results to JSON/CSV. (+7 more)

### Community 11 - "Baseline Metrics Extractor"
Cohesion: 0.10
Nodes (15): BaselineExtractor, DataFrame, Extract baseline metrics from historical trip data for comparison., Return trip dataframe for detailed analysis., Get baseline metrics by month., Extract historical trip metrics from All Droping.xlsx., Get baseline metrics by destination., Initialize baseline extractor. (+7 more)

### Community 12 - "Geocoder and Caching"
Cohesion: 0.11
Nodes (14): Geocoder, GeocoderCache, Geocoding module with Nominatim (OSM) support, caching, and rate limiting. Conve, Nominatim geocoder with caching and rate limiting., Enforce rate limiting (max 1 req/sec)., Geocode an address using Nominatim with retries.                  Args:, Geocode multiple locations with progress tracking.                  Args:, Get cache statistics. (+6 more)

### Community 13 - "Comprehensive EDA Module"
Cohesion: 0.13
Nodes (14): ComprehensiveBloodSupplyChainEDA, main(), COMPREHENSIVE BLOOD SUPPLY CHAIN EDA - FINAL ANALYSIS ==========================, Analyze blood supply chain specific metrics, Analyze temporal patterns for delivery optimization, Assess readiness for route optimization algorithms, Comprehensive EDA class combining all analysis capabilities for blood supply cha, Generate comprehensive final report with actionable insights (+6 more)

### Community 14 - "Comparison Chart Generation"
Cohesion: 0.13
Nodes (21): create_cost_comparison(), create_distance_comparison(), create_dl_performance(), create_integrated_summary(), create_quality_comparison(), create_time_comparison(), get_baseline_metrics(), get_dl_metrics() (+13 more)

### Community 15 - "Sample Data Loader"
Cohesion: 0.11
Nodes (12): DataLoader, DataFrame, Extract location information from DataFrame                  Args:             d, Create sample delivery requests for testing                  Args:             n, Load and process blood supply chain data from Excel files, Initialize data loader                  Args:             data_dir: Directory co, Create sample vehicles for testing                  Args:             num_vehicl, Get summary of loaded data (+4 more)

### Community 16 - "Routing and Geocoding Utils"
Cohesion: 0.14
Nodes (10): Geocoding utilities using OpenStreetMap Nominatim, Blood Supply Chain Route Optimization - Routing Module, OSRMRouter, Calculate haversine distance between two locations                  Args:, Interface to OSRM routing service, Estimate distance matrix using haversine distance for locations without coordina, Initialize OSRM router                  Args:             server_url: URL of OSR, Clear the route cache (+2 more)

### Community 17 - "CNN Model I/O"
Cohesion: 0.12
Nodes (8): DeliveryTimeCNN, Model, 1D CNN for predicting delivery time/lateness., Initialize 1D CNN model.                  Args:             config: Model config, Export training history to CSV for offline analysis., Export evaluation metrics to CSV (single row)., Load model from file., Build 1D CNN architecture.                  Args:             input_shape: (sequ

### Community 18 - "Optimization Utilities and Route Model"
Cohesion: 0.19
Nodes (14): create_simple_example(), Create a simple example with 5 deliveries, OptimizationResult, Represents a delivery route, Results from route optimization, Route, Blood Supply Chain Route Optimization - Utilities Module, generate_report() (+6 more)

### Community 19 - "DEAP GA Route Optimizer"
Cohesion: 0.17
Nodes (9): Calculate distance, time, and cost for a route                  Args:, Run genetic algorithm optimization                  Args:             verbose: P, Genetic Algorithm-based route optimizer, Convert an individual (sequence of delivery indices) to Route objects, Initialize optimizer after dataclass initialization, Setup DEAP genetic algorithm framework, Compute distance and time matrices for all delivery locations, Evaluate fitness of a route sequence                  Args:             individu (+1 more)

### Community 20 - "GA Data Extraction and Optimization"
Cohesion: 0.18
Nodes (10): Extract and parse data from Excel files (Data PMI.xlsx, All Droping.xlsx) for GA, GeneticAlgorithm, Genetic Algorithm for 2-vehicle blood supply routing optimization. Minimizes del, GA for 2-vehicle routing optimization.     Objectives: minimize delivery time (p, haversine_distance(), main(), Complete working demonstration of GA blood supply optimization for Malang Regenc, Calculate haversine distance in meters. (+2 more)

### Community 21 - "Sheet Overview Dashboard"
Cohesion: 0.24
Nodes (13): All Base Sheet, Columns per Sheet Bar Chart, Data Completeness Distribution Pie Chart, Data Quality Missing Percentage Chart, Data Types Distribution Heatmap, Date Range Coverage (Days) Chart, Droping RSSA Sheet, Keterlambatan & Waktu Trip Sheet (+5 more)

### Community 22 - "Comparison Reporting"
Cohesion: 0.15
Nodes (7): ComparisonReporter, Generate comparison reports between baseline and GA results., Generate JSON comparison report., Generate a summary comparison table., Generate formatted comparison reports., Print formatted report to console., Generate detailed text comparison report.

### Community 23 - "Optimization Pipeline Orchestrator"
Cohesion: 0.18
Nodes (8): DeepLearningRouteSelector, Deep Learning Route Selector for Parallel Competitive Model.  Generates candidat, DL-based route selector using candidate generation and CNN prediction., Main optimization pipeline orchestrator. Coordinates data extraction, geocoding,, main(), Quick test of integrated pipeline (GA + DL). Uses haversine distance to avoid OS, Run integrated pipeline with GA and DL., Quick test of the GA optimization pipeline using haversine (no OSRM calls). Uses

### Community 24 - "GA Candidate Generation"
Cohesion: 0.17
Nodes (6): Random permutation with balanced split.                  Args:             custo, Apply random swaps within routes for diversity.                  Args:, Predict total duration, lateness risk, and cost for a route solution., Generate candidates and select the best route using DL predictions., Generate diverse route permutations using heuristics.                  Uses:, Nearest neighbor heuristic with simple split.                  Args:

### Community 25 - "Route Optimization Readiness v1"
Cohesion: 0.18
Nodes (12): Capacity Data Readiness Dimension, Demand Data Readiness Dimension, Detailed Readiness Scores Bar Chart, Location Data Readiness Dimension, Priority Levels Readiness Dimension, Quality Control Readiness Dimension, Route Optimization Readiness Radar Chart, Route Optimization (Blood Supply Chain) (+4 more)

### Community 26 - "Geocoding Utilities"
Cohesion: 0.20
Nodes (7): Geocoder, Geocode multiple locations with rate limiting                  Args:, Reverse geocode coordinates to address                  Args:             latitu, Geocoding service using OpenStreetMap Nominatim, Initialize geocoder                  Args:             user_agent: User agent st, Geocode an address to coordinates                  Args:             address: Ad, Geocode a Location object and update its coordinates                  Args:

### Community 27 - "Demand and Facility Extraction"
Cohesion: 0.25
Nodes (6): DataFrame, Extract demand by hospital/facility from Permintaan Perwilayah 2024 sheet., Get all unique facility locations combining facilities and hospitals.         Re, Get data summary for validation., Extract unique facility locations from Jarak Pengiriman sheet.         Returns D, Extract historical trip data from Keterlambatan sheet.         Returns DataFrame

### Community 28 - "CNN Training Callbacks"
Cohesion: 0.22
Nodes (6): Callback, History, ndarray, Create training callbacks., Train the model.                  Args:             X_train: Training features (, Evaluate model on test data.

### Community 29 - "CNN Feature Engineering"
Cohesion: 0.24
Nodes (6): DataFrame, ndarray, Create sequences for 1D CNN.                  Args:             df: DataFrame wi, Full preprocessing pipeline.                  Returns:             Dict with tra, Load and clean trip data., Create features for model training.

### Community 30 - "Dataclass Serialization"
Cohesion: 0.25
Nodes (4): Convert to dictionary, Convert to dictionary, Convert to dictionary, Convert to dictionary

### Community 31 - "Blood Type Distribution"
Cohesion: 0.33
Nodes (7): Blood Supply Chain Route Optimization Project, Blood Type A (24.6%, 32 records), Blood Type AB (3.8%, 5 records), Blood Type B (29.2%, 38 records), Blood Type O (42.3%, 55 records), Blood Type (Golongan Darah) Distribution Chart, Golongan Darah (Blood Type) Categorical Feature

### Community 32 - "RSSA Temporal Shelf-Life Analysis"
Cohesion: 0.33
Nodes (7): Blood Product Shelf Life (Fixed 35-Day / 840-Hour Window), Daily Pattern: Tgl. Aftap (Blood Draw Date), Daily Pattern: Tgl. Kadaluarsa (Expiry Date), RSSA Hospital (Delivery/Drop-off Destination), Temporal Analysis - Droping RSSA, Time Gap: Tgl. Aftap to Tgl. Kadaluarsa Histogram, Time Gap: Tgl. Kadaluarsa to Tgl. Pengolahan Histogram

### Community 33 - "Route Readiness Assessment v2"
Cohesion: 0.33
Nodes (6): Capacity Data Readiness, Data Quality (Low, >15% missing), Demand Data Readiness, Location Data Readiness, Route Optimization Data Readiness Assessment, Time Data Readiness

### Community 34 - "RSSA Delivery Temporal Patterns"
Cohesion: 0.40
Nodes (6): Blood Supply Chain Route Optimization Project, RSSA Hospital (Droping Destination), Temporal Patterns - Droping RSSA (Delivery Chart), Tgl. Aftap (Blood Draw Date) Daily Pattern, Tgl. Kadaluarsa (Expiry Date) Daily Pattern, Tgl. Pengolahan (Processing Date) Daily Pattern

### Community 35 - "RSSA Time Gap Analysis"
Cohesion: 0.50
Nodes (5): Blood Product Expiry / Shelf-Life Window (840h ~35 days), RSSA Hospital Blood Drop-off, Time Gap: Tgl. Aftap to Tgl. Kadaluarsa Histogram, Time Gap: Tgl. Kadaluarsa to Tgl. Pengolahan Histogram, Time Gaps Analysis - Droping RSSA

### Community 36 - "Delivery Delay Analysis"
Cohesion: 0.50
Nodes (5): Delay Factors: Unnamed 1 (numeric bins 0-9), Delay Factors: Unnamed 2 (months Bulan-September), Delivery Delays Analysis - Keterlambatan & Waktu Trip, Distribution: Unnamed 0 (empty), Distribution: Unnamed 21 (empty)

### Community 37 - "DL Route Selector Init"
Cohesion: 0.50
Nodes (3): DeliveryTimeCNN, ndarray, Initialize DL Route Selector.                  Args:             cnn_model: Trai

### Community 38 - "Blood Component Distribution"
Cohesion: 0.67
Nodes (4): Komponen (Blood Component) Distribution Chart, PRC (Packed Red Cells) Component, TC (Thrombocyte Concentrate) Component, WB/DB (Whole Blood/Double Blood) Component

### Community 39 - "All-Sheets Data Overview"
Cohesion: 0.83
Nodes (4): All Base Sheet, Complete Data Overview - All Sheets, Droping RSSA Sheet, Keterlambatan & Waktu Trip Sheet

### Community 40 - "Blood Type and Component Analysis"
Cohesion: 0.67
Nodes (4): Blood Types and Components Analysis, Droping RSSA Blood Type Distribution (Golongan Darah: O, B, A, AB), Droping RSSA Blood Component Distribution (PRC, TC, WB/DB), RSSA Droping (Blood Delivery/Supply Drop) Data

### Community 41 - "RSSA Dataset Distribution"
Cohesion: 0.83
Nodes (4): Blood Type Distribution (RSSA_Golongan Darah), Blood Component Distribution (RSSA_Komponen), Blood Types and Components Distribution Analysis (Figure), RSSA Hospital Dataset (Droping Data)

### Community 42 - "Vehicle Route Model"
Cohesion: 0.50
Nodes (3): Represents a single vehicle's route., Return route summary., VehicleRoute

### Community 43 - "Missing Values Analysis"
Cohesion: 0.67
Nodes (3): Blood Bag Inventory Dataset (raw columns: No, No. Bag, Golongan Darah, RH, Komponen, Tgl. Aftap, Tgl. Kadaluarsa, Tgl. Pengolahan, Hasil Pemeriksaan, Unnamed 9-14), Data Cleaning / Missing Value Handling Step (drop Unnamed: 12-14 columns), Missing Values Analysis (Heatmap + Bar Chart)

### Community 44 - "Numerical Distribution Analysis"
Cohesion: 0.67
Nodes (3): 'No' Column Distribution (Right-Skewed Histogram + KDE), Numerical Distributions Figure, 'Unnamed: 12' Column - Insufficient Data (0 values)

## Ambiguous Edges - Review These
- `DL Predictor Module (dl_predictor/)` → `Missing tensorflow/keras/scikit-learn dependencies despite DL predictor requiring them`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to
- `Multi-objective Fitness Function (70% time / 30% cost)` → `GA fitness changed from lexicographic to weighted sum (70/30)`  [AMBIGUOUS]
  IMPLEMENTATION_SUMMARY.md · relation: conceptually_related_to
- `Custom GA implementation (no DEAP, ga_optimizer/ package, lexicographic fitness)` → `DEAP-based GA implementation (src/ package structure)`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `requirements.txt (Python dependencies)` → `Missing tensorflow/keras/scikit-learn dependencies despite DL predictor requiring them`  [AMBIGUOUS]
  requirements.txt · relation: references

## Knowledge Gaps
- **71 isolated node(s):** `Feature Engineering (7 features per sequence step)`, `OSRM Integration (routing)`, `Smart Batching Strategy (priority/time/geo)`, `Data Models (Location, BloodProduct, Delivery, Vehicle, Route, OptimizationResult)`, `Risma Nurizza Saputri (thesis author)` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `DL Predictor Module (dl_predictor/)` and `Missing tensorflow/keras/scikit-learn dependencies despite DL predictor requiring them`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Multi-objective Fitness Function (70% time / 30% cost)` and `GA fitness changed from lexicographic to weighted sum (70/30)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Custom GA implementation (no DEAP, ga_optimizer/ package, lexicographic fitness)` and `DEAP-based GA implementation (src/ package structure)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `requirements.txt (Python dependencies)` and `Missing tensorflow/keras/scikit-learn dependencies despite DL predictor requiring them`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `DeliveryTimeCNN` connect `CNN Model I/O` to `DL Preprocessing Pipeline`, `Data Extraction Pipeline`, `Comparison Chart Generation`, `Optimization Pipeline Orchestrator`, `CNN Training Callbacks`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `OptimizationPipeline` connect `Data Extraction Pipeline` to `CNN Model I/O`, `GA Data Extraction and Optimization`, `DL Preprocessing Pipeline`, `Optimization Pipeline Orchestrator`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `BaselineExtractor` connect `Baseline Metrics Extractor` to `DL Preprocessing Pipeline`, `Comparison Chart Generation`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._