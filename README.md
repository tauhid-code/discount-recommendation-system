#  Discount Recommendation System (End-to-End ML Project)

An end-to-end Machine Learning project designed to identify customers who are likely to increase their order frequency when offered a small discount (5–10%).  
The system helps businesses optimize discount allocation by targeting the *right* customers instead of rewarding already loyal users.

---

##  Project Objective

The objective of this project is to build a predictive system that identifies **discount-responsive but non-loyal customers** in an online food delivery platform.

Instead of offering discounts to customers who already place frequent orders, the model focuses on customers who:
- Show high sensitivity towards discounts
- Have lower order frequency
- Are likely to increase engagement if incentivized

This enables **cost-effective growth** and **better marketing ROI**.

---

##  Business Logic Behind the Target Variable

The target variable is **not directly available** in the dataset and is created using business-driven rules.

### Step 1: Discount Sensitivity
A customer is considered *discount-sensitive* if:
More Offers and Discount ≥ 4

### Step 2: Order Frequency
A customer is considered a *high-order (loyal) customer* if:
No. of orders placed ≥ 70th percentile

### Step 3: Target Definition
The final target variable `target_discount_growth` is defined as:
target_discount_growth = 1
if customer is discount-sensitive AND NOT a high-order customer
else 0

This ensures discounts are recommended **only to customers who can potentially grow**, avoiding unnecessary spending on already loyal users.

## Explainability with SHAP

This project includes both Global and Local SHAP explainability, making the model fully interpretable.

### Global SHAP (Model-Level Explainability)

Purpose:
Understand which features influence discount decisions across all customers.

How it works:

SHAP values are computed on the training dataset

Mean absolute SHAP values are aggregated

Top contributing features are saved to: models/global_shap.csv

Usage:

Displayed as a pie chart in the Streamlit UI

Helps stakeholders answer:

“What generally drives discount recommendations?”

### Local SHAP (Prediction-Level Explainability)

Purpose:
Explain why a specific customer received (or didn’t receive) a discount.

How it works:

SHAP values are computed per prediction

Shows how each feature pushes the probability up or down

Used alongside the predicted probability

Usage:

Returned by the Flask API

Can be visualized as:

Feature contribution breakdown

Explanation text in UI