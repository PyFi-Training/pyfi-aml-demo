import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="PyFi - Liquidity Predictor", layout="wide")

# ============================================================
# SECTION 1: THE STORY
# ============================================================

st.title("Corporate Liquidity Predictor")
st.caption("Applied Machine Learning for Investment Banking Advisory")

st.markdown("""
You are an investment banking analyst on a team that advises large corporations on raising
capital. Your clients are the treasurers and CFOs of companies like Macy's, McDonald's,
Nordstrom, Tiffany & Co., and some of the largest companies on earth. They regularly ask
your team how much liquidity they should maintain and how they compare to their peers.

Liquidity is corporate spending power. Your personal equivalent would be the cash in your
bank account plus your credit card limit. For a large corporation, maintaining excess
liquidity is expensive, but maintaining too little is risky. If liquidity hits zero, the
company goes bankrupt. This is exactly the kind of optimization problem where machine
learning thrives.

Historically, your team has advised clients using a simple linear regression built in Excel.
That model achieves an R-squared of roughly 0.48, meaning it explains less than half of the
variation in the variable you are trying to predict. Your objective today is to construct a
machine learning regression algorithm that outperforms this existing analysis, using data
from 800+ public companies.
""")

st.divider()

# ============================================================
# SECTION 2: EXPLORE THE DATA
# ============================================================

st.header("Step 1: Explore the Data")

@st.cache_data
def load_data():
    df = pd.read_csv('liquidity_data.csv')
    return df

df = load_data()

col_info, col_preview = st.columns([1, 2])

with col_info:
    st.markdown(f"""
    **Dataset:** {len(df)} public companies

    **Target variable:** Available Liquidity (what we are predicting)

    **Input features:** S&P credit score, market cap, total debt,
    LTM capex, LTM EBITDA, LTM free cash flow, and LTM revenue.
    All figures are in millions of dollars.
    """)

with col_preview:
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ============================================================
# SECTION 3: SPLIT THE DATA
# ============================================================

st.header("Step 2: Split into Training and Testing Sets")

st.markdown("""
Before training any model, we separate the data into two groups. The training set (80%)
is what the models learn from. The testing set (20%) is held back and used to evaluate
how well each model performs on data it has never seen. This prevents overfitting.
""")

target = df['available_liquidity']
inputs = df.drop(columns=['available_liquidity'])

input_train, input_test, target_train, target_test = train_test_split(
    inputs, target, test_size=0.2, random_state=1
)

split_col1, split_col2 = st.columns(2)
with split_col1:
    st.metric("Training Observations", f"{len(input_train)}")
with split_col2:
    st.metric("Testing Observations", f"{len(input_test)}")

st.divider()

# ============================================================
# SECTION 4: TRAIN AND COMPETE
# ============================================================

st.header("Step 3: Train Models and Select a Winner")

st.markdown("""
We are going to train six different regression algorithms on the same training data,
then evaluate each one against the testing data to find the best performer. The models
range from the simple linear regression your team currently uses in Excel, all the way
to advanced ensemble methods like Gradient Boosting.

The machine learning models also go through hyperparameter tuning and cross-validation,
which means the algorithm automatically tests hundreds of different configurations to find
the best settings for each model. This is something you cannot do in a spreadsheet.
""")

if st.button("Train All Models", type="primary"):

    results = {}

    progress_bar = st.progress(0, text="Preparing data...")
    time.sleep(0.5)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(input_train)
    X_test_scaled = scaler.transform(input_test)

    progress_bar.progress(10, text="Scaling data...")
    time.sleep(0.3)

    # Store scaler and test data in session state for later use
    st.session_state['scaler'] = scaler
    st.session_state['X_test_scaled'] = X_test_scaled
    st.session_state['target_test'] = target_test
    st.session_state['feature_names'] = inputs.columns

    # ---- Model 1: Linear Regression (the Excel baseline) ----
    progress_bar.progress(15, text="Training Linear Regression (the Excel baseline)...")
    time.sleep(0.5)
    lr = LinearRegression()
    lr.fit(X_train_scaled, target_train)
    lr_pred = lr.predict(X_test_scaled)
    results['Linear Regression'] = {
        'r2': r2_score(target_test, lr_pred),
        'mae': mean_absolute_error(target_test, lr_pred),
        'pred': lr_pred,
        'color': '#999999'
    }

    # ---- Model 2: Lasso ----
    progress_bar.progress(25, text="Training Lasso Regression...")
    time.sleep(0.5)
    lasso_pipe = make_pipeline(StandardScaler(), Lasso(random_state=1))
    lasso_grid = GridSearchCV(
        lasso_pipe,
        {'lasso__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5]},
        cv=5
    )
    lasso_grid.fit(input_train, target_train)
    lasso_pred = lasso_grid.predict(input_test)
    results['Lasso'] = {
        'r2': r2_score(target_test, lasso_pred),
        'mae': mean_absolute_error(target_test, lasso_pred),
        'pred': lasso_pred,
        'color': '#5B9BD5'
    }

    # ---- Model 3: Ridge ----
    progress_bar.progress(38, text="Training Ridge Regression...")
    time.sleep(0.5)
    ridge_pipe = make_pipeline(StandardScaler(), Ridge(random_state=1))
    ridge_grid = GridSearchCV(
        ridge_pipe,
        {'ridge__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5]},
        cv=5
    )
    ridge_grid.fit(input_train, target_train)
    ridge_pred = ridge_grid.predict(input_test)
    results['Ridge'] = {
        'r2': r2_score(target_test, ridge_pred),
        'mae': mean_absolute_error(target_test, ridge_pred),
        'pred': ridge_pred,
        'color': '#70AD47'
    }

    # ---- Model 4: Elastic Net ----
    progress_bar.progress(50, text="Training Elastic Net...")
    time.sleep(0.5)
    enet_pipe = make_pipeline(StandardScaler(), ElasticNet(random_state=1))
    enet_grid = GridSearchCV(
        enet_pipe,
        {
            'elasticnet__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5],
            'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        cv=5
    )
    enet_grid.fit(input_train, target_train)
    enet_pred = enet_grid.predict(input_test)
    results['Elastic Net'] = {
        'r2': r2_score(target_test, enet_pred),
        'mae': mean_absolute_error(target_test, enet_pred),
        'pred': enet_pred,
        'color': '#FFC000'
    }

    # ---- Model 5: Random Forest ----
    progress_bar.progress(65, text="Training Random Forest...")
    time.sleep(0.5)
    rf_pipe = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1))
    rf_grid = GridSearchCV(
        rf_pipe,
        {
            'randomforestregressor__n_estimators': [100, 200],
            'randomforestregressor__max_features': [0.3, 0.6, 1.0]
        },
        cv=5
    )
    rf_grid.fit(input_train, target_train)
    rf_pred = rf_grid.predict(input_test)
    results['Random Forest'] = {
        'r2': r2_score(target_test, rf_pred),
        'mae': mean_absolute_error(target_test, rf_pred),
        'pred': rf_pred,
        'color': '#ED7D31'
    }

    # ---- Model 6: Gradient Boosting ----
    progress_bar.progress(82, text="Training Gradient Boosting...")
    time.sleep(0.5)
    gb_pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1))
    gb_grid = GridSearchCV(
        gb_pipe,
        {
            'gradientboostingregressor__n_estimators': [100, 200],
            'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
            'gradientboostingregressor__max_depth': [1, 3, 5]
        },
        cv=5
    )
    gb_grid.fit(input_train, target_train)
    gb_pred = gb_grid.predict(input_test)
    results['Gradient Boosting'] = {
        'r2': r2_score(target_test, gb_pred),
        'mae': mean_absolute_error(target_test, gb_pred),
        'pred': gb_pred,
        'color': '#C00000'
    }

    progress_bar.progress(100, text="All models trained. Evaluating results...")
    time.sleep(0.5)
    progress_bar.empty()

    # Store results and winning model in session state
    st.session_state['results'] = results
    st.session_state['gb_model'] = gb_grid

    # ---- RESULTS TABLE ----
    st.subheader("Model Performance on Test Data")

    # Sort by R-squared descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)

    results_df = pd.DataFrame([
        {
            'Model': name,
            'R-Squared': round(data['r2'], 3),
            'Mean Absolute Error ($M)': round(data['mae'], 0)
        }
        for name, data in sorted_results
    ])
    results_df.index = range(1, len(results_df) + 1)
    results_df.index.name = 'Rank'

    st.dataframe(results_df, use_container_width=True)

    winner_name = sorted_results[0][0]
    winner_r2 = sorted_results[0][1]['r2']
    baseline_r2 = results['Linear Regression']['r2']
    improvement = ((winner_r2 - baseline_r2) / baseline_r2) * 100

    st.success(
        f"Winner: {winner_name} with R-squared of {winner_r2:.3f}. "
        f"That is a {improvement:.0f}% improvement over the Linear Regression baseline ({baseline_r2:.3f})."
    )

    # ---- SCATTER PLOTS: BASELINE vs WINNER ----
    st.subheader("Visual Comparison: Baseline vs Winner")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline: Linear Regression
    ax1.scatter(
        target_test, results['Linear Regression']['pred'],
        alpha=0.5, s=35, color='#999999', edgecolors='white', linewidth=0.5
    )
    min_val = min(target_test.min(), results['Linear Regression']['pred'].min())
    max_val = max(target_test.max(), results['Linear Regression']['pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Liquidity ($M)', fontsize=11)
    ax1.set_ylabel('Predicted Liquidity ($M)', fontsize=11)
    ax1.set_title(
        f'Linear Regression (R² = {baseline_r2:.3f})',
        fontsize=12, fontweight='bold'
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Winner: Gradient Boosting
    winner_data = results[winner_name]
    ax2.scatter(
        target_test, winner_data['pred'],
        alpha=0.5, s=35, color=winner_data['color'], edgecolors='white', linewidth=0.5
    )
    min_val2 = min(target_test.min(), winner_data['pred'].min())
    max_val2 = max(target_test.max(), winner_data['pred'].max())
    ax2.plot([min_val2, max_val2], [min_val2, max_val2], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Liquidity ($M)', fontsize=11)
    ax2.set_ylabel('Predicted Liquidity ($M)', fontsize=11)
    ax2.set_title(
        f'{winner_name} (R² = {winner_r2:.3f})',
        fontsize=12, fontweight='bold'
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.divider()

# ============================================================
# SECTION 5: PREDICT FOR A CLIENT
# ============================================================

if 'gb_model' in st.session_state:

    st.header("Step 4: Advise a Client")

    st.markdown("""
    Now that we have a trained model, we can use it to advise a real client. Enter your
    client's financial data below. The model will estimate their ideal available liquidity
    based on the patterns it learned from 800+ public companies.
    """)

    st.sidebar.header("Client Financial Data")
    st.sidebar.caption("Enter your client's financials below. All figures in millions ($M).")

    sp_score = st.sidebar.slider(
        "S&P Credit Score",
        min_value=0,
        max_value=10,
        value=3,
        help="Standard & Poor's credit rating on a 0-10 scale. Higher = stronger credit."
    )

    market_cap = st.sidebar.number_input(
        "Market Cap ($M)",
        value=20000,
        help="Total market capitalization in millions."
    )

    total_debt = st.sidebar.number_input(
        "Total Debt ($M)",
        value=5000,
        help="Total outstanding debt in millions."
    )

    ltm_capex = st.sidebar.number_input(
        "LTM Capital Expenditure ($M)",
        value=-500,
        help="Last twelve months capital expenditure. Typically negative (cash outflow)."
    )

    ltm_ebitda = st.sidebar.number_input(
        "LTM EBITDA ($M)",
        value=2000,
        help="Last twelve months earnings before interest, taxes, depreciation, and amortization."
    )

    ltm_fcf = st.sidebar.number_input(
        "LTM Free Cash Flow ($M)",
        value=1000,
        help="Last twelve months free cash flow in millions."
    )

    ltm_revenue = st.sidebar.number_input(
        "LTM Revenue ($M)",
        value=10000,
        help="Last twelve months total revenue in millions."
    )

    input_data = pd.DataFrame([{
        'sp_score': sp_score,
        'market_cap': market_cap,
        'total_debt': total_debt,
        'ltm_capex': ltm_capex,
        'ltm_ebitda': ltm_ebitda,
        'ltm_fcf': ltm_fcf,
        'ltm_revenue': ltm_revenue
    }])

    # Align columns with training data
    for col in st.session_state['feature_names']:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[st.session_state['feature_names']]

    prediction = st.session_state['gb_model'].predict(input_data)[0]

    st.metric(
        "Recommended Available Liquidity",
        f"${prediction:,.0f}M"
    )

    st.markdown("""
    This recommendation is based on the collective financial behavior of 800+ public
    companies. The final amount your client chooses to maintain may be adjusted based on
    qualitative factors or non-public information, but this serves as the quantitative
    baseline for your advisory conversation.
    """)

else:
    st.info("Click 'Train All Models' above to begin the analysis. Once trained, you can advise a client using the sidebar.")

st.divider()
st.caption("Built with Python, Scikit-Learn & Streamlit")
