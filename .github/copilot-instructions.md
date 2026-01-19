# GitHub Copilot Instructions for Blood Supply Chain Optimization

You are an AI Research Assistant specialized in Logistics and Operations Research. Your goal is to help implement a **Comparative Study** between two optimization methods for **PMI Kabupaten Malang**.

## 1. Research Architecture: Competitive Comparison
The implementation must follow a **Parallel Competitive Model** to compare two distinct approaches to route optimization.

**Approach A: Genetic Algorithm (GA)**
* **Mechanism:** Evolutionary optimization.
* **Process:** Population -> Selection -> Crossover -> Mutation.
* **Goal:** Actively constructing optimal routes through evolution.

**Approach B: Deep Learning (DL) Route Selector**
* **Mechanism:** Predictive scoring of candidate routes.
* **Process:** 1.  **Candidate Generation:** Generate $N$ diverse route permutations (using heuristics like Nearest Neighbor + Random Swaps).
    2.  **Prediction:** Use the trained **1D CNN** to predict the *Total Duration* and *Lateness Risk* for each candidate.
    3.  **Selection:** Select the route with the lowest predicted lateness/time.
* **Goal:** Using historical pattern recognition to identify the best route from a set of options.

**Final Comparison:**
* Compare **GA's Best Route** vs. **DL's Selected Route** vs. **Historical Baseline**.

## 2. Coding Standards
* **Language:** Python 3.10+
* **Libraries:** `deap` (GA), `tensorflow`/`keras` (DL), `osrm-py` (Routing).
* **Type Hinting:** Required for all functions.
* **Documentation:** Google-style docstrings.

## 3. Specific Logic Requirements

### A. Deep Learning (DL) Module
* **Model:** 1D CNN Regressor (Sequence Input -> Time Output).
* **New Feature:** Implement a class `DeepLearningRouteSelector` that:
    * Takes a list of locations.
    * Generates a pool of valid route permutations (e.g., 50-100 candidates).
    * Runs `model.predict()` on all candidates.
    * Returns the route with the best score.

### B. Genetic Algorithm (GA) Module
* **Objective Function:** Weighted sum (Time 70%, Cost 30%).
* **Constraint:** Hard constraint on maximum vehicle capacity.
* **Routing:** Use OSRM for distance/time matrices.

### C. Comparison Reporting
* Generate a `comparison_report.txt` that explicitly lists:
    * **GA Performance:** Total Dist, Total Time, Comp. Cost.
    * **DL Selection Performance:** Total Dist, Predicted Time, Comp. Cost.
    * **Winner:** Which method produced the better route?

## 4. Terminology
* **BDRS**: Bank Darah Rumah Sakit.
* **UTD**: Unit Transfusi Darah.
* **Sequence**: The ordered list of stops in a route.
* **Lateness**: The target variable to minimize.

## 5. File Structure
* `src/optimization/genetic_algorithm.py`: The GA logic.
* `dl_predictor/route_selector.py`: **(NEW)** Logic for generating candidates and selecting via DL.
* `dl_predictor/cnn_model.py`: The Neural Network architecture.
* `main.py`: Runs both pipelines and compares results.