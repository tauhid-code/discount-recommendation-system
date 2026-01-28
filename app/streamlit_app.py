import streamlit as st
import requests


st.set_page_config(
    page_title="Discount Recommendation System",
    page_icon="🍔",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #fff5f5;
}

h1, h2, h3 {
    color: #e23744;
    font-family: 'Segoe UI', sans-serif;
}

.stButton > button {
    background-color: #e23744;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #cb202d;
    color: white;
}

.result-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🍕 Discount Recommendation System")
st.write(
    "Predict whether a customer should receive a discount based on their ordering behavior."
)

st.divider()

st.subheader("📋 Customer Details")

orders = st.slider(
    "📦 Number of Orders Placed",
    min_value=1,
    max_value=100,
    value=15
)

discount = st.slider(
    "🏷️ Discount Preference (1 = Low, 5 = High)",
    min_value=1,
    max_value=5,
    value=3
)

order_value = st.slider(
    "💰 Average Order Value (₹)",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

delivery_exp = st.slider(
    "🚴 Delivery Experience (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4
)


if st.button("🔍 Predict Discount Eligibility"):
    payload = {
        "orders": orders,
        "discount": discount,
        "order_value": order_value,
        "delivery_exp": delivery_exp
    }

    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json=payload,
            timeout=5
        )
        result = response.json()

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        if result["recommend_discount"] == 1:
            st.success(" Discount Recommended")
            st.write(
                f"**Confidence:** {result['confidence'] * 100:.1f}%"
            )
            st.write(
                "This customer is discount-sensitive and not yet highly loyal. "
                "Offering a small discount can increase engagement and order frequency."
            )
        else:
            st.error(" Discount Not Required")
            st.write(
                f"**Confidence:** {result['confidence'] * 100:.1f}%"
            )
            st.write(
                "This customer already shows strong engagement. "
                "Providing discounts may not lead to additional growth."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Unable to connect to prediction service. Make sure Flask is running.")

st.divider()
st.caption(
    "Built with  using Machine Learning, Flask & Streamlit"
)