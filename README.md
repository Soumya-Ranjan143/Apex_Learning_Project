# 🏦 Loan Eligibility Prediction & Classification Pipeline

An end-to-end Machine Learning project designed to automate loan approval predictions using tabular financial and demographic data. 

## Key Highlights
- **End-to-End Pipeline**: Handles missing value imputation (mode/median), categorical encoding, feature scaling, and polynomial expansion.
- **Multi-Model Evaluation**: Benchmarks **Logistic Regression**, **Polynomial Logistic Regression**, **Decision Tree**, and **Random Forest** models across standardized metrics.
- **Winning Model**: **Random Forest Classifier** achieved **86.99% Accuracy** and an **ROC-AUC of 0.8879**, providing high sensitivity to creditworthy candidates.
- **Serialized Artifacts**: Production-ready `.pkl` model weights generated and saved for rapid deployment.



##  Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| **Baseline Logistic Regression** | 86.18% | 84.00% | 98.82% | 0.9081 | 0.8511 |
| **Polynomial Logistic (deg=2)** | 73.98% | 79.78% | 83.53% | 0.8161 | 0.7279 |
| **Decision Tree** | 85.37% | 84.54% | 96.47% | 0.9011 | 0.7814 |
| **Random Forest (Winner)** 🏆 | **86.99%** | **84.85%** | **98.82%** | **0.9130** | **0.8879** |

---

## Tech Stack & Dependencies
- **Language**: Python 3.x
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`
- **Environment**: JupyterLab / VS Code

---

## Repository Structure
```text
.
├── models/                     # Serialized .pkl model weights
│   ├── model_1_logistic.pkl
│   ├── model_2_polynomial.pkl
│   ├── model_3_decision_tree.pkl
│   └── model_4_random_forest.pkl
├── 1.Model_Train.ipynb         # End-to-end training notebook
├── train_ctrUa4K.csv           # Dataset
├── README.md                   # Project documentation
└── .gitignore                  # Git tracking rules
