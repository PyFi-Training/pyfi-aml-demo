import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="PyFi AML Demo", layout="wide")

# Title and subtitle
st.title("PyFi Applied Machine Learning Demo")
st.caption("Powered by Python + Scikit-Learn")

# Load and prepare data for Investor Classifier
@st.cache_resource
def load_investor_model():
    # Load data
    df = pd.read_csv('investor_data_2.csv')

    # Drop specified columns
    df_processed = df.drop(columns=['invite_tier', 'fee_share', 'invite'])

    # Create dummy variables
    df_dummy = pd.get_dummies(df_processed)

    # Drop the commit_Commit column
    df_dummy = df_dummy.drop(columns=['commit_Commit'])

    # Define target and features
    y = df_dummy['commit_Decline']
    X = df_dummy.drop(columns=['commit_Decline'])

    # Create and train pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        random_state=1,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X_scaled, y)

    return model, scaler, X.columns, df

# Load and prepare data for Liquidity Predictor
@st.cache_resource
def load_liquidity_model():
    # Load data
    df = pd.read_csv('liquidity_data.csv')

    # Define target and features
    y = df['available_liquidity']
    X = df.drop(columns=['available_liquidity'])

    # Create and train pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        random_state=1,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X_scaled, y)

    # Store actual values for visualization
    y_pred = model.predict(X_scaled)

    return model, scaler, X.columns, X_scaled, y, y_pred

# Create tabs
tab1, tab2 = st.tabs(["Investor Classifier", "Liquidity Predictor"])

# ==================== TAB 1: INVESTOR CLASSIFIER ====================
with tab1:
    st.header("Investor Commitment Classifier")

    # Load model
    classifier, scaler_inv, feature_names, inv_df = load_investor_model()

    # Create sidebar inputs for Investor Classifier
    st.sidebar.subheader("Investor Classifier Inputs")

    investor = st.sidebar.selectbox(
        "Investor",
        ["Goldman Sachs", "Deutsche Bank", "Bank of America", "Wells Fargo", "MUFG Union"],
        key="investor"
    )

    deal_size = st.sidebar.number_input(
        "Deal Size",
        value=1000,
        key="deal_size"
    )

    rating = st.sidebar.slider(
        "Rating",
        min_value=1,
        max_value=10,
        value=3,
        key="rating"
    )

    int_rate = st.sidebar.selectbox(
        "Interest Rate",
        ["Market", "Above", "Below"],
        key="int_rate"
    )

    covenants = st.sidebar.slider(
        "Covenants",
        min_value=1,
        max_value=3,
        value=2,
        key="covenants"
    )

    total_fees = st.sidebar.number_input(
        "Total Fees",
        value=100,
        key="total_fees"
    )

    prior_tier = st.sidebar.selectbox(
        "Prior Tier",
        ["Bookrunner", "Participant"],
        key="prior_tier"
    )

    tier_change = st.sidebar.selectbox(
        "Tier Change",
        ["None", "Promoted", "Demoted"],
        key="tier_change"
    )

    fee_percent = st.sidebar.slider(
        "Fee Percent",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.01,
        key="fee_percent"
    )

    invite_percent = st.sidebar.slider(
        "Invite Percent",
        min_value=0.0,
        max_value=0.3,
        value=0.12,
        step=0.01,
        key="invite_percent"
    )

    # Prepare input for prediction
    input_data = {
        'deal_size': deal_size,
        'rating': rating,
        'covenants': covenants,
        'total_fees': total_fees,
        'fee_percent': fee_percent,
        'invite_percent': invite_percent
    }

    # Add investor dummy variable
    for inv_name in ["Goldman Sachs", "Deutsche Bank", "Bank of America", "Wells Fargo", "MUFG Union"]:
        input_data[f'investor_{inv_name}'] = 1 if investor == inv_name else 0

    # Add int_rate dummy variable
    for ir in ["Market", "Above", "Below"]:
        input_data[f'int_rate_{ir}'] = 1 if int_rate == ir else 0

    # Add prior_tier dummy variable
    for pt in ["Bookrunner", "Participant"]:
        input_data[f'prior_tier_{pt}'] = 1 if prior_tier == pt else 0

    # Add tier_change dummy variable
    for tc in ["None", "Promoted", "Demoted"]:
        input_data[f'tier_change_{tc}'] = 1 if tier_change == tc else 0

    # Create DataFrame with proper feature alignment
    input_df = pd.DataFrame([input_data])

    # Align features with training data
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    # Scale input
    input_scaled = scaler_inv.transform(input_df)

    # Make prediction
    prediction = classifier.predict(input_scaled)[0]
    probability = classifier.predict_proba(input_scaled)[0]

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Result")
        prediction_text = "Commit" if prediction == 0 else "Decline"
        st.metric("Predicted Outcome", prediction_text)

        # Probability gauge
        commit_prob = probability[0] * 100
        decline_prob = probability[1] * 100

        fig, ax = plt.subplots(figsize=(6, 3))
        categories = ["Commit", "Decline"]
        probabilities = [commit_prob, decline_prob]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.barh(categories, probabilities, color=colors)
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax.text(prob + 2, i, f"{prob:.1f}%", va="center")
        ax.set_title("Commitment Probability")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Model Performance (Test Set)")

        # Hardcoded confusion matrix
        cm = np.array([[1124, 22], [23, 278]])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Commit', 'Decline'],
                    yticklabels=['Commit', 'Decline'],
                    cbar=False, ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)

        # Metrics
        st.metric("AUROC Score", "0.9683")

        # Calculate accuracy from confusion matrix
        total = cm.sum()
        correct = cm[0, 0] + cm[1, 1]
        accuracy = correct / total
        st.metric("Accuracy", f"{accuracy:.2%}")

# ==================== TAB 2: LIQUIDITY PREDICTOR ====================
with tab2:
    st.header("Liquidity Predictor")

    # Load model
    regressor, scaler_liq, feature_names_liq, X_train_scaled, y_train, y_train_pred = load_liquidity_model()

    # Create sidebar inputs for Liquidity Predictor
    st.sidebar.subheader("Liquidity Predictor Inputs")

    sp_score = st.sidebar.slider(
        "S&P Score",
        min_value=0,
        max_value=10,
        value=3,
        key="sp_score"
    )

    market_cap = st.sidebar.number_input(
        "Market Cap",
        value=20000,
        key="market_cap"
    )

    total_debt = st.sidebar.number_input(
        "Total Debt",
        value=5000,
        key="total_debt"
    )

    ltm_capex = st.sidebar.number_input(
        "LTM Capex",
        value=-500,
        key="ltm_capex"
    )

    ltm_ebitda = st.sidebar.number_input(
        "LTM EBITDA",
        value=2000,
        key="ltm_ebitda"
    )

    ltm_fcf = st.sidebar.number_input(
        "LTM FCF",
        value=1000,
        key="ltm_fcf"
    )

    ltm_revenue = st.sidebar.number_input(
        "LTM Revenue",
        value=10000,
        key="ltm_revenue"
    )

    # Prepare input for prediction
    input_data_liq = {
        'sp_score': sp_score,
        'market_cap': market_cap,
        'total_debt': total_debt,
        'ltm_capex': ltm_capex,
        'ltm_ebitda': ltm_ebitda,
        'ltm_fcf': ltm_fcf,
        'ltm_revenue': ltm_revenue
    }

    # Create DataFrame with proper feature alignment
    input_df_liq = pd.DataFrame([input_data_liq])

    # Align features with training data
    for col in feature_names_liq:
        if col not in input_df_liq.columns:
            input_df_liq[col] = 0

    input_df_liq = input_df_liq[feature_names_liq]

    # Scale input
    input_scaled_liq = scaler_liq.transform(input_df_liq)

    # Make prediction
    predicted_liquidity = regressor.predict(input_scaled_liq)[0]

    # Display results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Predicted Liquidity")
        st.metric("Available Liquidity", f"${predicted_liquidity:,.0f}")

    with col2:
        st.subheader("Model Performance")
        # Calculate R-squared
        ss_res = np.sum((y_train - y_train_pred) ** 2)
        ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        st.metric("R² Score", f"{r_squared:.4f}")

    # Scatter plot of predicted vs actual
    st.subheader("Predicted vs Actual (Training Data)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_train, y_train_pred, alpha=0.5, s=30)

    # Add perfect prediction line
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Liquidity')
    ax.set_ylabel('Predicted Liquidity')
    ax.set_title('Model Predictions vs Actual Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)

# Footer
st.divider()
st.caption("ML-driven financial analytics for institutional investors • Built with Streamlit & Scikit-Learn")
