# NetSmart — Smart Internet Usage Optimization
## University Project | Probability & Statistics

> **Run the app:** `streamlit run app.py`

---

## Project Architecture

This project follows a **clean modular architecture** where every concern is isolated into its own file. `app.py` acts as a thin router — it loads data, trains models, and dispatches to page-specific render functions.

```
Smart Internet Usage Optimization/
│
├── app.py                  ← ENTRY POINT (run: streamlit run app.py)
├── config.py               ← Global settings, feature maps, and UI tokens
├── data_loader.py          ← Unified data loading (Excel + Live CSV + Online)
├── model_trainer.py        ← ML logic, feature encoding, and comparison tools
├── prob_stats_tools.py     ← Gaussian, Poisson, and Binomial math tools
├── styles/
│   └── main.css            ← "Warm Academic" theme (CSS)
├── views/                  ← Page-specific rendering logic
│   ├── page_home.py
│   ├── page_speed_check.py
│   ├── page_insights.py
│   ├── page_model_report.py
│   ├── page_comparison.py
│   └── page_about.py
├── Data/                   ← Storage for Excel and CSV datasets
├── test_project.py         ← Unit testing suite
└── requirements.txt        ← Project dependencies
```

---

## Module Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Entry point, global UI configuration, and page routing |
| `config.py` | Centralized constants, feature maps, and color palette |
| `data_loader.py` | Merging and cleaning of static Excel and dynamic CSV data |
| `model_trainer.py` | Training 4 models (ML + Statistical) and predicting speeds |
| `prob_stats_tools.py` | Probabilistic distribution calculations (Normal, Poisson, Binomial) |
| `styles/main.css` | Custom theme overrides for Streamlit and Plotly |
| `views/` | Individual modules for rendering each application page |

---

## The Four AI Models

| Model | Type | Purpose |
|---|---|---|
| **Linear Regression** | Supervised ML | Fast, interpretable speed baseline |
| **Bayesian Ridge** | Probabilistic ML | Handles uncertainty in predictions |
| **Gradient Boosting** | Ensemble ML | Most accurate DL/UL predictor |
| **Parametric Statistical** | Group-wise stats | Theoretical Gaussian baseline |

---

## Probability Distributions Used

| Distribution | Applied To | Page |
|---|---|---|
| **Normal (Gaussian)** | Speed uncertainty, slow probability | Speed Check, Model Report |
| **Poisson** | User-count / congestion modelling | Speed Check, Model Report |
| **Binomial** | Weekly slow-session reliability | Speed Check, Model Report |

---

## Running the Tests

```bash
# Run the full unit-test suite (508 lines, 40+ tests)
python test_project.py
```

## Dependencies

```
streamlit
pandas
numpy
scikit-learn
scipy
plotly
speedtest-cli
statsmodels
```
