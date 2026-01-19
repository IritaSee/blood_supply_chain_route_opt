# Parallel Competitive Model Implementation

## Overview

This implementation follows a **Parallel Competitive Model** to compare two distinct route optimization approaches for PMI Kabupaten Malang blood supply chain.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         PARALLEL COMPETITIVE MODEL ARCHITECTURE             │
└─────────────────────────────────────────────────────────────┘

Input Data (Locations, Trip History)
        ↓
        ├──────────────────┬──────────────────┐
        ↓                  ↓                  ↓
   APPROACH A:        APPROACH B:        HISTORICAL
   Genetic Algorithm  DL Route Selector  BASELINE
        ↓                  ↓                  ↓
    Evolutionary      Candidate Gen +    Historical
    Optimization      CNN Prediction      Metrics
        ↓                  ↓                  ↓
    Best Route        Best Route         Avg Metrics
        └──────────────────┴──────────────────┘
                           ↓
                  COMPARISON REPORT
              (Winner + Improvements)
```

## Approach A: Genetic Algorithm (GA)

**Mechanism:** Evolutionary optimization  
**Process:**
1. Initialize random population of route solutions
2. Selection: Tournament selection
3. Crossover: Ordered crossover (OX1)
4. Mutation: Random swaps

**Objective Function:** Weighted sum (70% Time + 30% Cost)

**File:** `ga_optimizer/genetic_algorithm.py`

## Approach B: Deep Learning Route Selector

**Mechanism:** Predictive scoring of candidate routes  
**Process:**
1. **Candidate Generation:** Generate 50-100 diverse route permutations using:
   - Nearest Neighbor heuristic
   - Random swaps for diversity
   - Balanced vehicle assignment
2. **Prediction:** Use trained 1D CNN to predict total duration and lateness risk for each candidate
3. **Selection:** Select the route with the lowest predicted time/lateness score

**File:** `dl_predictor/route_selector.py`

## Usage

```python
from main import OptimizationPipeline

# Run Parallel Competitive Model
pipeline = OptimizationPipeline(use_osrm=True)
results = pipeline.run_full_pipeline(
    population_size=150,      # GA population size
    generations=800,          # GA generations
    train_dl=True,            # Train DL model
    dl_epochs=50,             # DL training epochs
    num_dl_candidates=100,    # Number of candidates for DL selection
    run_comparison=True       # Compare both approaches
)

# Check comparison report
print(f"Comparison report: {results['comparison_report']}")
```

## Output

The pipeline generates:

1. **GA Results** (`results/ga_results.json`)
   - Total distance, time, cost
   - Route details per vehicle

2. **DL Selection Results** (`results/dl_selection_results.json`)
   - Candidates evaluated
   - Best route metrics
   - Lateness risk prediction

3. **Comparison Report** (`results/comparison_report.txt`)
   - Side-by-side comparison
   - **Winner determination**
   - Improvement percentages

4. **DL Model** (`results/dl_models/trained_model.keras`)
   - Trained 1D CNN model
   - Evaluation metrics

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestrator |
| `ga_optimizer/genetic_algorithm.py` | GA implementation |
| `dl_predictor/route_selector.py` | DL route selector (NEW) |
| `dl_predictor/cnn_model.py` | 1D CNN architecture |
| `results/comparison_report.txt` | Final comparison (NEW) |

## Comparison Criteria

Both approaches are evaluated on:
- **Total Distance** (km)
- **Total Time** (hours)
- **Total Cost** (IDR)
- **Weighted Score:** 70% Time + 30% Cost

The approach with the **lower weighted score wins**.

## Running the Pipeline

```bash
# Full pipeline with comparison
python main.py

# Or customize parameters
python -c "
from main import OptimizationPipeline
p = OptimizationPipeline()
results = p.run_full_pipeline(
    population_size=200,
    generations=1000,
    num_dl_candidates=150
)
"
```

## Dependencies

- `deap` - Genetic Algorithm framework
- `tensorflow`/`keras` - Deep Learning
- `osrm-py` or `requests` - Routing
- `numpy`, `pandas` - Data processing
- `openpyxl` - Excel I/O

## Validation

The comparison report (`comparison_report.txt`) explicitly lists:
- **GA Performance**: Total Dist, Total Time, Total Cost
- **DL Selection Performance**: Total Dist, Predicted Time, Total Cost
- **Winner**: Which method produced the better route
- **Improvement**: Percentage improvement of winner over the other

---

**Architecture:** Parallel Competitive Model  
**Updated:** January 19, 2026  
**Status:** Implementation Complete
