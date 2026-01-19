# Quick Start Guide - Parallel Competitive Model

## 🚀 Run the Full Pipeline

```bash
python main.py
```

That's it! The pipeline will:
1. Load data
2. Train DL model
3. Run GA optimization
4. Run DL route selection
5. Generate comparison report

---

## 📊 Check Results

```bash
# View comparison report
cat results/comparison_report.txt

# View GA results
cat results/ga_optimization_results.json

# View DL selection results  
cat results/dl_selection_results.json
```

---

## ⚙️ Customize Parameters

```python
from main import OptimizationPipeline

pipeline = OptimizationPipeline(use_osrm=True)

results = pipeline.run_full_pipeline(
    # GA Parameters
    population_size=150,      # Population size for GA
    generations=800,          # Number of generations
    
    # DL Parameters
    train_dl=True,            # Whether to train DL model
    dl_epochs=50,             # Training epochs
    num_dl_candidates=100,    # Number of routes to evaluate
    
    # Comparison
    run_comparison=True       # Generate comparison report
)
```

---

## 🎯 Expected Output

```
================================================================
PARALLEL COMPETITIVE MODEL - COMPARISON REPORT
================================================================

APPROACH A: GENETIC ALGORITHM (GA)
----------------------------------
Total Distance:    245.3 km
Total Time:        4.2 hours
Total Cost:        IDR 1,234,567
Weighted Score:    3.14

APPROACH B: DEEP LEARNING ROUTE SELECTOR
-----------------------------------------
Total Distance:    238.7 km
Total Time:        3.9 hours
Total Cost:        IDR 1,198,432
Weighted Score:    2.89
Candidates Evaluated: 100

HISTORICAL BASELINE
-------------------
Average Distance:  267.4 km
Average Time:      5.1 hours
Average Cost:      IDR 1,456,789

================================================================
WINNER: Deep Learning Route Selector
================================================================
Improvement over GA: 7.96%
Improvement over Baseline: 23.14%
```

---

## 🔧 Troubleshooting

**Error: "No module named 'deap'"**
```bash
pip install -r requirements.txt
```

**Error: "OSRM server not reachable"**
- Set `use_osrm=False` in pipeline initialization
- Or ensure OSRM server is running

**DL model fails to train**
- Check if data file exists: `config/cleaned_trip_data.xlsx`
- Verify columns: `Waktu Keberangkatan`, `Waktu Tiba`, `Tujuan`

---

## 📁 File Locations

| File | Description |
|------|-------------|
| `main.py` | Pipeline orchestrator |
| `ga_optimizer/genetic_algorithm.py` | GA implementation |
| `dl_predictor/route_selector.py` | DL route selector |
| `dl_predictor/cnn_model.py` | 1D CNN model |
| `results/comparison_report.txt` | Comparison output |

---

## 🧪 Test with Sample Data

```bash
# Quick test with small parameters
python -c "
from main import OptimizationPipeline
p = OptimizationPipeline()
results = p.run_full_pipeline(
    population_size=50,       # Small for testing
    generations=100,          # Quick run
    dl_epochs=10,             # Fast training
    num_dl_candidates=20      # Few candidates
)
"
```

---

**Need Help?** Check:
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Detailed changes
- [PARALLEL_COMPETITIVE_MODEL.md](PARALLEL_COMPETITIVE_MODEL.md) - Architecture overview
- `.github/copilot-instructions.md` - Research requirements
