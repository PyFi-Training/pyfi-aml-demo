# pyfi-aml-demo

Case study notebooks and datasets from PyFi's **Applied Machine Learning** class. This repo has been trimmed down to just the data and Jupyter notebooks so students (and the curious) can see the shape of the problem and run the models end-to-end without any app or infra scaffolding in the way.

If you want the full walkthrough that shows you how to build these models from scratch, line by line, that lives in the paid course: [pyfi.com/products/applied-machine-learning](https://pyfi.com/products/applied-machine-learning).

## The case study

You are given a dataset of historical private-deal invitations and investor responses. Each row captures a deal that was offered to an investor along with attributes of the deal (size, interest rate, covenants, fees, fee share, rating, prior tier, invite tier) and the investor's decision (commit).

The case study asks two supervised learning questions:

1. **Investor classifier.** Given a new deal and an investor, can you predict whether that investor will commit? This is the classification problem worked through in `investor_classifier_part1.ipynb` and refined in `investor_classifier_part2.ipynb`.
2. **Liquidity predictor.** Across the committed capital in the book, can you forecast expected liquidity under varying deal conditions? This regression task is layered onto the same feature set.

The two problems share a feature space but force different framings, loss functions, and evaluation lenses. That contrast is the point of the case study.

## Repo contents

```
pyfi-aml-demo/
├── investor_classifier_part1.ipynb   # EDA, feature engineering, baseline classifier
├── investor_classifier_part2.ipynb   # Model refinement, evaluation, liquidity predictor
├── investor_data.csv                 # Primary training dataset
├── investor_data_2.csv               # Secondary dataset used in part 2
├── requirements.txt
└── README.md
```

Column schema (both CSVs): `investor, commit, deal_size, invite, rating, int_rate, covenants, total_fees, fee_share, prior_tier, invite_tier, ...`

`commit` is the target for the classifier. Numeric features drive the liquidity predictor.

> Note on the data: the datasets here are teaching datasets. They are representative of the structure of real private-market deal flow but are not live production data and should not be used for investment decisions.

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

This repo is one case study. The full course is a 10-week deep dive into building production-grade ML systems in Python. Topics include:

- Problem framing and data diligence for financial use cases
- Feature engineering on messy, real-world tabular data
- Classification (logistic regression, tree-based models, gradient boosting)
- Regression, including robust methods for heavy-tailed financial targets
- Cross-validation, leakage traps, and honest model evaluation
- Class imbalance, cost-sensitive learning, and threshold calibration
- Model interpretability (permutation importance, SHAP, partial dependence)
- Packaging models for handoff to engineering
- A written, line-by-line walkthrough of the investor classifier and liquidity predictor in this repo

Full curriculum, sample lessons, and enrollment: [pyfi.com/products/applied-machine-learning](https://pyfi.com/products/applied-machine-learning).

## Who this is for

Analysts, associates, data professionals, and engineers who want to move from "I've read about ML" to "I can build, evaluate, and defend a model on real finance data." You should be comfortable with Python and basic pandas before starting. Everything else is taught.

## License and contributions

This is a teaching repository. Code and data are provided for educational use in connection with the PyFi Applied Machine Learning course. Redistribution of the datasets is not permitted.

Pull requests are not accepted on this repo. If you spot a bug or a typo, open an issue and we will take a look. For broader questions about the course, reach the PyFi team through the product page linked above.

---

Built by [PyFi](https://pyfi.com). If you work through the notebooks and get stuck, the full walkthrough in the paid course will unstick you faster than any Stack Overflow thread.
