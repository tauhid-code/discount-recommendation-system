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