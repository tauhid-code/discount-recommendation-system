import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os


st.set_page_config(
    page_title="Discount Recommendation System",
    page_icon="",
    layout="wide"
)


st.markdown("""
<style>
/* Global App Styling */
.stApp {
    background: #f0f0f0;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}

/* Main container */
.main-container {
    padding: 3rem 2rem;
    max-width: 1400px;
    margin: 0 auto;
    background: #f0f0f0;
}

/* Header */
.page-header {
    text-align: left;
    margin-bottom: 2.5rem;
}

.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
}

.page-subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* CHANGED: Added parent card styling with padding and shadow */
.parent-card {
    background: #ffffff;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    border: none;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 1.2rem 1.6rem;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: none;
}

/* Customer Container - Card styling with padding (kept for backward compatibility) */
.customer-container {
    background: #ffffff;
    padding: 1.2rem 1.6rem;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: none;
}

.card-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #1f2937;
    margin: 0 0 1.2rem 0;
}

/* Customer details inner block */
.customer-details {
    padding: 0.25rem 0;
}

/* Slider Styling */
.stSlider > label {
    font-weight: 600 !important;
    color: #374151 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

.stSlider {
    margin-bottom: 1.8rem !important;
}

div[data-baseweb="slider"] > div {
    background: #e5e7eb !important;
    height: 8px !important;
    border-radius: 10px !important;
}

div[data-baseweb="slider"] div[role="progressbar"] {
    background: linear-gradient(90deg, #ff9d3f 0%, #ff7f50 100%) !important;
}

div[data-baseweb="slider"] div[role="slider"] {
    background: #ffffff !important;
    border: 3px solid #ff9d3f !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 2px 8px rgba(255, 157, 63, 0.4) !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff9d3f 0%, #ff7f50 100%);
    color: white;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 1rem;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(255, 157, 63, 0.3);
    margin-top: 1.5rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(255, 157, 63, 0.4);
}

/* Result styling */
.result-header {
    text-align: center;
    margin-bottom: 1.5rem;
}

.result-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

.result-success {
    background: #d1fae5;
    color: #065f46;
    border: 1px solid #6ee7b7;
}

.result-fail {
    background: #fee2e2;
    color: #7f1d1d;
    border: 1px solid #fca5a5;
}

.result-text {
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

.divider {
    height: 1px;
    background: #e5e7eb;
    margin: 1.5rem 0;
}

.explanation-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 1rem;
    text-align: center;
}

.explanation-text {
    color: #6b7280;
    font-size: 0.9rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Info Box */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    color: #1e40af;
    font-weight: 600;
    font-size: 0.95rem;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Broad selectors to target the first Streamlit column wrapper
   (different Streamlit versions produce slightly different DOMs) */
.stColumns > div:nth-child(1) > div,
.stColumns > div:nth-child(1) .stBlock,
.stColumns > div:nth-child(1) .element-container,
.main-container .stColumns > div:first-child > div,
.main-container > div > div:nth-child(1) .stColumn > div,
div.block-container > div > .stColumns > div:first-child > div {
    background: #ffffff !important;
    padding: 1.2rem 1.6rem !important;
    border-radius: 50px !important;
    box-shadow: 0 12px 40px rgba(15,23,42,0.08) !important;
    border: 5px solid #e5e7eb !important;
    margin-bottom: 1rem !important;
}

/* Ensure the title aligns to the top inside that visual card */
.stColumns > div:nth-child(1) .card-title,
.main-container .stColumns > div:first-child .card-title {
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)



st.markdown('<div class="main-container">', unsafe_allow_html=True)


st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Discount Recommendation System</h1>
        <p class="page-subtitle">Predict whether a customer should receive a discount based on their ordering behavior.</p>
    </div>
""", unsafe_allow_html=True)



st.markdown('<div class="parent-card">', unsafe_allow_html=True)


col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    
    st.markdown('<h3 class="card-title">Customer Details</h3>', unsafe_allow_html=True)

    orders = st.slider(
        "Number of Orders Placed",
        min_value=1,
        max_value=100,
        value=48,
        help="Total number of orders placed by the customer"
    )

    discount = st.slider(
        "Discount Preference (1 = Low, 5, High)",
        min_value=1,
        max_value=5,
        value=4,
        help="Customer's preference for discounts"
    )

    order_value = st.slider(
        "Average Order Value (₹)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Average value of customer's orders"
    )

    delivery_exp = st.slider(
        "Delivery Experience (1 = Poor, 5, Excellent)",
        min_value=1,
        max_value=5,
        value=5,
        help="Customer's delivery experience rating"
    )

    predict = st.button("Evaluate Customer for Discount")

with col_right:
  
    if predict:
        try:
            res = requests.post(
                "http://localhost:5000/predict",
                json={
                    "orders": orders,
                    "discount": discount,
                    "order_value": order_value,
                    "delivery_exp": delivery_exp
                },
                timeout=5
            ).json()

            confidence = res.get("confidence", 0)
            recommend = res.get("recommend_discount", 0)

            if recommend == 1:
                st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;"></div>
                        <h2 style="font-size: 1.3rem; font-weight: 700; color: #1f2937; margin: 0; margin-bottom: 0.3rem;">Discount Recommended</h2>
                        <div style="color: #6b7280; font-size: 0.85rem;">Result 75 - 70.0% · 13 minutes since yesterday.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;"></div>
                        <h2 style="font-size: 1.3rem; font-weight: 700; color: #1f2937; margin: 0; margin-bottom: 0.3rem;">Discount Not Required</h2>
                        <div style="color: #6b7280; font-size: 0.85rem;">Result 75 - 70.0% · 13 minutes since yesterday.</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                    <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #10b981 0%, #f59e0b 100%); height: 100%; width: {confidence*100:.0f}%;"></div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="text-align: right; font-weight: 700; color: #1f2937; font-size: 0.95rem;">{confidence*100:.0f}%</div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem;">Local: DecisionTree · Local SHAP explanation not available.</div>', unsafe_allow_html=True)

        except requests.exceptions.RequestException:
            st.error(" Unable to connect to prediction service. Please ensure the Flask API is running on port 5000.")
        except Exception as e:
            st.error(f" An error occurred: {str(e)}")
    else:
        st.markdown("""
            <div class="info-box">
                 Click "Evaluate Customer" to see the recommendation
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="card-title">Why did the model decide this?</h3>', unsafe_allow_html=True)
    st.markdown('<p class="explanation-text">These factors influenced the model\'s decision across custommost.</p>', unsafe_allow_html=True)
    
    path = "models/global_shap.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path).head(6)
            
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            colors = ['#2563eb', '#ff8c42', '#8b5cf6', '#ec4899', '#14b8a6', '#f59e0b']
            
            wedges, texts, autotexts = ax.pie(
                df["importance"],
                labels=df["feature"],
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                textprops={'fontsize': 6.5, 'weight': 'bold', 'color': '#1f2937'}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(6.5)
                autotext.set_weight('bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown('<p style="text-align: center; color: #6b7280; font-size: 0.75rem; margin-top: 0.5rem;">Top drivers influencing discount decisions across customers</p>', unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Unable to load chart: {str(e)}")
    else:
        st.markdown("""
            <div class="info-box">
                SHAP importance data not found at models/global_shap.csv
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)