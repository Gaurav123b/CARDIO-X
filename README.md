# Cardio-X: An Intelligent ML-Driven System for Cholesterol Risk Detection, Prevention & Personalized Lifestyle Advisory

## 1. Project Objective
Cardio-X is an academic healthcare informatics project aimed at developing an intelligent, machine-learning-driven system to detect elevated cardiovascular and cholesterol risks, identify contributing clinical/lifestyle factors, and deliver evidence-based, personalized advisory recommendations.

The current phase focuses on setting up a reproducible, modular development environment and project framework.

---

## 2. Planned Technology Stack
- **Language:** Python 3.11+ (running in isolated `.venv`)
- **Data Manipulation & Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning & Preprocessing:** scikit-learn
- **Prototyping & Exploration:** Jupyter Notebooks (`jupyter`, `ipykernel`)
- **Testing & Verification:** pytest
- **Future Application Layer:** Streamlit (to be introduced in Phase 5)
- **Model Persistence:** Joblib (to be introduced in Phase 4)

---

## 3. Directory Structure
```text
CARDIO-X/
│
├── .gitignore              # Git ignore rules for virtualenvs, cache, and sensitive data
├── README.md               # Main project documentation and setup guide
├── requirements.txt        # Initial dependencies for data analysis and testing
│
├── data/                   # Clinical datasets (local only, excluded from git)
│   └── README.md           # Dataset documentation and confidentiality guidelines
│
├── notebooks/              # Jupyter notebooks for exploratory data analysis (EDA)
│   └── .gitkeep
│
├── src/                    # Modular Python source package
│   └── __init__.py         # Package initialization
│
├── models/                 # Serialized model artifacts and encoders
│   └── .gitkeep
│
├── reports/                # Visualizations, metric reports, and summary figures
│   └── .gitkeep
│
└── tests/                  # Automated verification and unit tests
    ├── __init__.py
    └── test_environment.py # Verification tests for project structure and environment
```

---

## 4. Environment Setup Instructions (macOS / Linux)

### 4.1 Activate Virtual Environment
A local virtual environment `.venv` has been initialized. To activate it in your terminal:
```bash
source .venv/bin/activate
```

*(On Windows PowerShell, the command would be: `.\.venv\Scripts\Activate.ps1`)*

### 4.2 Install Dependencies
After activating the virtual environment, install the pinned minimal dependencies:
```bash
pip install -r requirements.txt
```

### 4.3 Run Verification Tests
Verify environment integrity and directory structure using `pytest` or direct Python execution:
```bash
python -m pytest tests/
```
Or directly:
```bash
python -c "import sys; print(f'Python environment verified: {sys.version}')"
```

---

## 5. Current Development Status & Disclaimers
> [!NOTE]
> **Phase 1 (Environment Setup and Verification): Completed.**
> Machine-learning model development has **not** started yet. No claims regarding model training, predictive performance, or accuracy are made at this stage.

> [!IMPORTANT]
> **Target Variable Validation Notice:** Before any machine learning pipeline is constructed, the prospective prediction target (e.g., hypercholesterolemia, Ten-Year CHD risk, or elevated lipid markers) must be strictly validated against the actual columns and distributions of the underlying dataset during Phase 2.
