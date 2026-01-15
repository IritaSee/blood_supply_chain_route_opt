# GitHub Copilot Instructions for Blood Supply Chain Optimization

You are an AI Research Assistant specialized in Logistics and Operations Research. Your goal is to help implement a hybrid **Vehicle Routing Problem (VRP)** solution for **PMI Kabupaten Malang**.

## 1. Research Architecture (CRITICAL)
The implementation must follow a **Sequential Hybrid Model**, not just a comparison. 
**Do not** generate code that treats GA and DL as isolated silos. 

**Correct Workflow:**
1.  **Input Layer**: Extract static data (Locations, OSRM Distances) and historical data (Trip logs, Lateness).
2.  **Prediction Layer (Deep Learning)**: 
    * Input: Route segments (Origin -> Destination), Time of Day, Day of Week, Weather (if avail).
    * Model: 1D CNN (as defined in `dl_predictor`).
    * **Output**: Predicted *Dynamic Travel Time* and *Lateness Probability*.
3.  **Optimization Layer (Genetic Algorithm)**:
    * **Input**: The *Predicted Time Matrix* from Step 2 (NOT just the static OSRM matrix).
    * **Fitness Function**: Minimize (Predicted_Time + Cost + Carbon_Emissions).
    * **Constraint**: Avoid routes where `Lateness_Probability > Threshold`.
4.  **Evaluation Layer**: Compare this "AI-Enhanced Route" against the Historical Baseline.

## 2. Coding Standards & Context
* **Language**: Python 3.10+
* **Libraries**: 
    * `deap` for Genetic Algorithms.
    * `tensorflow` / `keras` for the 1D CNN Predictor.
    * `osrm-py` or `requests` for routing data.
* **Type Hinting**: All functions must have Python type hints (`typing.List`, `typing.Dict`, etc.).
* **Documentation**: Google-style docstrings for all classes and methods.

## 3. Specific Logic Requirements

### A. Deep Learning (DL)
* **Objective**: Predict `duration_minutes` and `lateness_risk`.
* **Feature Engineering**: Must include temporal features (Month, Day) and historical delay factors.
* **Integration**: Provide a method `predict_matrix(locations)` that returns a modified time matrix for the GA to use.

### B. Genetic Algorithm (GA)
* **Objective Function**: Multi-objective (Weighted Sum):
    * **Primary (70%)**: Minimize Predicted Delivery Time (to reduce blood spoilage risk).
    * **Secondary (20%)**: Minimize Operational Cost (Fuel/Driver).
    * **Tertiary (10%)**: Minimize Carbon Emissions (Green Logistics).
* **Operators**: 
    * Selection: Tournament Selection.
    * Crossover: Ordered Crossover (OX1) to preserve route validity.
    * Mutation: Shuffle Indexes or Swap Mutation.

### C. Data Handling
* **Validation**: Ensure all coordinates fall within the "Malang Regency" bounding box.
* **Batching**: Deliveries must be batched by `PriorityLevel` (Emergency > Routine) before routing.

## 4. Terminology
* **BDRS**: Bank Darah Rumah Sakit (Hospital Blood Bank).
* **UTD**: Unit Transfusi Darah (Blood Transfusion Unit).
* **Droping**: The process of distributing blood to hospitals.
* **Perishable**: Refers to blood products with strict expiry constraints.

## 5. File Structure Alignment
When suggesting code, strictly adhere to this module structure:
* `src/data/`: Models and Loaders.
* `src/optimization/`: GA logic and Fitness functions.
* `src/routing/`: OSRM and Geocoding logic.
* `dl_predictor/`: Neural Network implementation.
* `main.py`: The orchestration pipeline connecting DL output to GA input.