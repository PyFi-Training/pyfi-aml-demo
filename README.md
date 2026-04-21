# PyFi-Applied Machine Learning - demo

Case study notebooks and datasets from PyFi's **Applied Machine Learning** class (the "AML" in the repo name). This repo has been trimmed down to just the data and Jupyter notebooks so students (and the curious) can see the shape of the problem and run the model end-to-end without any app or infra scaffolding in the way.

If you want the full walkthrough that shows you how to build this model from scratch, line by line, that lives in the paid course: [pyfi.com/products/applied-machine-learning](https://pyfi.com/products/applied-machine-learning).

## The case study: investor classifier

You are given a dataset of historical private-deal invitations and investor responses. Each row captures a deal that was offered to an investor along with attributes of the deal (size, interest rate, covenants, fees, fee share, rating, prior tier, invite tier) and the outcome (did the investor commit).

The question the case study asks is simple to state and non-trivial to answer well: **given a new deal and an investor, can you predict whether that investor will commit?**

The notebooks walk through it in two passes:

- `investor_classifier_part1.ipynb` covers EDA, feature engineering, and a baseline classifier.
- `investor_classifier_part2.ipynb` picks up from there with model refinement, evaluation, and discussion of the trade-offs you run into on imbalanced, real-world tabular data.

## Repo contents

```
pyfi-aml-demo/
├── investor_classifier_part1.ipynb   # EDA, feature engineering, baseline classifier
├── investor_classifier_part2.ipynb   # Model refinement and evaluation
├── investor_data.csv                 # Primary training dataset
├── investor_data_2.csv               # Secondary dataset used in part 2
├── requirements.txt
└── README.md
```

Column schema (both CSVs): `investor, commit, deal_size, invite, rating, int_rate, covenants, total_fees, fee_share, prior_tier, invite_tier, ...`

`commit` is the target for the classifier.

> Note on the data: these are teaching datasets. They are representative of the structure of real private-market deal flow but are not live production data and should not be used for investment decisions.

## Setup

**Requirements:** Python 3.10 or newer, pip, and Jupyter.

```bash
# clone
git clone https://github.com/PyFi-Training/pyfi-aml-demo.git
cd pyfi-aml-demo

# create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
pip install jupyterlab

# launch
jupyter lab
```

Open `investor_classifier_part1.ipynb` first, then `investor_classifier_part2.ipynb`. The notebooks assume the CSV files sit in the same directory as the notebook.

Dependencies are intentionally light: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## What the full Applied Machine Learning course covers

This repo is one case study. The full course is a deep dive into building production-grade ML systems in Python. Topics include:

- Problem framing and data diligence for financial use cases
- Feature engineering on messy, real-world tabular data
- Classification with logistic regression, tree-based models, and gradient boosting
- Cross-validation, leakage traps, and honest model evaluation
- Class imbalance, cost-sensitive learning, and threshold calibration
- Model interpretability (permutation importance, SHAP, partial dependence)
- Packaging models for handoff to engineering
- A written, line-by-line walkthrough of the investor classifier in this repo

Full curriculum, sample lessons, and enrollment: [pyfi.com/products/applied-machine-learning](https://pyfi.com/products/applied-machine-learning).

## Who this is for

Analysts, associates, data professionals, and engineers who want to move from "I've read about ML" to "I can build, evaluate, and defend a model on real finance data." You should be comfortable with Python and basic pandas before starting. Everything else is taught.

## License and contributions

This is a teaching repository. Code and data are provided for educational use in connection with the PyFi Applied Machine Learning course. Redistribution of the datasets is not permitted.

Pull requests are not accepted on this repo. If you spot a bug or a typo, open an issue and we will take a look. For broader questions about the course, reach the PyFi team through the product page linked above.

---

Built by [PyFi](https://pyfi.com). If you work through the notebooks and get stuck, the full walkthrough in the paid course will unstick you faster than any Stack Overflow thread.
