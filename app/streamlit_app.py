import sys
import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import DiscountPredictor

st.set_page_config(
    page_title="Discount Recommendation Model",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_predictor():
    return DiscountPredictor(
        model_path=os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"),
        feature_columns_path=os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
    )

@st.cache_data
def load_global_shap():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "models", "global_shap.csv"))
    return df.sort_values("importance", ascending=False).head(10)

predictor      = load_predictor()
global_shap_df = load_global_shap()

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.stApp {
    background-color: #f0f0f0 !important;
}

.app-header {
    background: linear-gradient(135deg, #FF6200 0%, #FF8C38 100%);
    text-align: center;
    padding: 16px 24px 12px;
    box-shadow: 0 4px 18px rgba(255, 98, 0, 0.28);
    margin-bottom: 0;
}
.app-header h1 {
    color: #ffffff;
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: 0.3px;
}
.app-header p {
    color: rgba(255,255,255,0.93);
    font-size: 0.85rem;
    font-style: italic;
    font-weight: 300;
    margin: 0;
}

[data-testid="stHorizontalBlock"] {
    padding: 14px 20px !important;
    gap: 20px !important;
    align-items: flex-start !important;
    background: transparent !important;
}

[data-testid="stHorizontalBlock"]:first-of-type > div {
    background: #ffffff !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.09) !important;
    padding: 20px 24px !important;
    border: 1px solid #ebebeb !important;
    min-height: auto !important;
}

[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] > div {
    background: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    border: none !important;
    min-height: unset !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: #FF6200 !important;
    height: 16px !important;
    border-radius: 99px !important;
}
[data-testid="stSlider"] > div > div > div {
    height: 16px !important;
    border-radius: 99px !important;
    background: #e0e0e0 !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background: #FF6200 !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 0 0 2px #FF6200 !important;
    width: 26px !important;
    height: 26px !important;
}
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p,
[data-testid="stSlider"] label span,
[data-testid="stSlider"] .st-emotion-cache-ue6h4q,
div[data-testid="stSlider"] > label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #1e1e1e !important;
}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #FF6200 !important;
    font-weight: 700 !important;
}

.stFormSubmitButton {
    display: flex !important;
    justify-content: center !important;
    margin-top: 6px !important;
}
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #FF6200, #FF8C38) !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 40px !important;
    width: 95% !important;
    white-space: nowrap !important;
    letter-spacing: 0.5px !important;
    transition: box-shadow 0.2s, transform 0.2s !important;
}
.stFormSubmitButton > button:hover {
    box-shadow: 0 10px 28px rgba(255,98,0,0.45) !important;
    transform: translateY(-2px) !important;
}

.stCaptionContainer p {
    font-size: 0.78rem !important;
    color: #999 !important;
    margin-top: 2px !important;
}

hr { border: none; border-top: 1px solid #f0f0f0; margin: 6px 0 14px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🏷️ Discount Recommendation Model</h1>
    <p>Smarter discounts, happier customers — powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<p style="font-size:20px;font-weight:700;color:#1e1e1e;margin:0 0 2px;">📋 Customer Details</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:0.85rem;color:#888;margin:0 0 12px;">Adjust the sliders to match the customer\'s profile.</p>',
        unsafe_allow_html=True
    )

    with st.form("eval_form"):

        st.markdown('<p style="font-size:15px;font-weight:600;color:#1e1e1e;margin-bottom:0;">Number of Orders Placed</p>', unsafe_allow_html=True)
        orders = st.slider("Number of Orders Placed", min_value=1, max_value=100, value=25, step=1, label_visibility="collapsed")
        st.caption("Lifetime order count for this customer.")
        st.markdown('<hr style="margin:2px 0 6px 0;">', unsafe_allow_html=True)

        st.markdown('<p style="font-size:15px;font-weight:600;color:#1e1e1e;margin-bottom:0;">Discount Preference &nbsp;<span style="font-weight:400;color:#888;font-size:13px;">(1 = Not Interested · 5 = Highly Interested)</span></p>', unsafe_allow_html=True)
        discount = st.slider("Discount Preference", min_value=1, max_value=5, value=3, step=1, label_visibility="collapsed")
        st.caption("Propensity score based on past coupon redemptions.")
        st.markdown('<hr style="margin:2px 0 6px 0;">', unsafe_allow_html=True)

        st.markdown('<p style="font-size:15px;font-weight:600;color:#1e1e1e;margin-bottom:0;">Average Order Value (₹)</p>', unsafe_allow_html=True)
        order_value = st.slider("Average Order Value", min_value=100, max_value=2000, value=500, step=50, label_visibility="collapsed")
        st.caption("Mean gross transaction value across all successful orders.")
        st.markdown('<hr style="margin:2px 0 6px 0;">', unsafe_allow_html=True)

        st.markdown('<p style="font-size:15px;font-weight:600;color:#1e1e1e;margin-bottom:0;">Delivery Experience &nbsp;<span style="font-weight:400;color:#888;font-size:13px;">(1 = Very Poor · 5 = Excellent)</span></p>', unsafe_allow_html=True)
        delivery_exp = st.slider("Delivery Experience", min_value=1, max_value=5, value=3, step=1, label_visibility="collapsed")
        st.caption("Subjective sentiment score based on logistics feedback.")

        submitted = st.form_submit_button("⚡  Evaluate Customer for Discount")

    if submitted:
        with st.spinner("Analysing customer data…"):
            try:
                res = predictor.predict({
                    "orders":       orders,
                    "discount":     discount,
                    "order_value":  float(order_value),
                    "delivery_exp": delivery_exp,
                })
                st.session_state.result = res
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.session_state.result = None

with right:
    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            min-height:60vh; color:#ccc; text-align:center; gap:14px;
        ">
            <div style="font-size:3.8rem;">📊</div>
            <p style="font-size:1.5rem; max-width:500px; line-height:1.65; color:#888;">
                Adjust the sliders on the left and click
                <strong style="color:#aaa;">⚡ Evaluate Customer</strong>
                to see the recommendation and SHAP analysis here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        recommend  = result.get("recommend_discount", 0)
        confidence = result.get("confidence", 0.5)
        reason     = result.get("reason", "ML-based decision")

        yes   = recommend == 1
        pct   = round(confidence * 100)
        v_txt = "✅  Discount Recommended" if yes else "❌  Discount Not Recommended"
        v_col = "#28a745" if yes else "#dc3545"

        st.markdown(f"""
        <div style="
            background:#fff8f4;
            border-radius:14px;
            padding:22px 26px;
            border-left:5px solid #FF6200;
            box-shadow:0 2px 12px rgba(255,98,0,0.10);
            margin-bottom:22px;
        ">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1.1px;color:#aaa;margin-bottom:10px;">
                Discount Recommendation
            </div>
            <div style="font-size:1.8rem;font-weight:800;color:{v_col};">{v_txt}</div>
            <span style="
                display:inline-block; margin-top:12px;
                background:#FFF0E6; color:#FF6200;
                font-size:0.8rem; font-weight:600;
                padding:5px 14px; border-radius:99px;
            ">💡 {reason}</span>
            <div style="margin-top:18px;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#888;margin-bottom:6px;">
                    <span>Model Confidence</span><span>{pct}%</span>
                </div>
                <div style="height:10px;background:#ead9ce;border-radius:99px;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#FF6200,#FF8C38);border-radius:99px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom:6px;">
            <span style="font-size:1.2rem;font-weight:700;color:#1e1e1e;">🌐 Global SHAP Analysis</span><br>
            <span style="font-size:0.78rem;color:#999;">Overall feature importance across the entire trained model.</span>
        </div>
        <hr style="border-top:1px solid #ebebeb;margin:8px 0 14px;">
        """, unsafe_allow_html=True)

        gdf = global_shap_df.sort_values("importance", ascending=False).copy()

        global_palette = [
            "#FF6200", "#1A73E8", "#28A745", "#DC3545", "#FFC107",
            "#6F42C1", "#17A2B8", "#E83E8C", "#20C997", "#FD7E14",
        ]
        global_colors = [global_palette[i % len(global_palette)] for i in range(len(gdf))]

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        wedges, texts, autotexts = ax.pie(
            gdf["importance"],
            labels=gdf["feature"],
            colors=global_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.78,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        for t in texts:
            t.set_fontsize(8.5)
            t.set_color("#333")
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("white")
            at.set_fontweight("bold")
        ax.set_title("Global Feature Importance Distribution", fontsize=11, color="#555", pad=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
