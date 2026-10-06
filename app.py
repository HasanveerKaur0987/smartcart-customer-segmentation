import streamlit as st
import pandas as pd
import joblib

# ------ PageSetup -------
st.set_page_config(page_title="SmartCart Segmentation", layout = "wide")
st.title("SmartCart Customer Segmentation")
st.write("Grouping customers by shopping behavior to plan better marketing.")

# ------load model and data -------
@st.cache_resource
def load_model():
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    behavior_cols = joblib.load("models/behavior_cols.pkl")
    segment_names = joblib.load("models/segment_names.pkl")
    return scaler, kmeans, behavior_cols, segment_names

@st.cache_data
def load_data():
    return pd.read_csv("data/customers_with_segments.csv")

scaler, kmeans, behavior_cols, segment_names = load_model()
df = load_data()

# ---------Tabs ------------
tab1, tab2 = st.tabs(["Segment Overview", "Find a Customer's Segment"])

# ---------- Tab 1: Overview ----------
with tab1:

    # ---------- Part 1: How many customers in each segment ----------
    st.subheader("How many customers are in each segment?")

    # Count customers in each segment
    counts = df["Segment"].value_counts()

    # Make 4 boxes side by side
    col1, col2, col3, col4 = st.columns(4)

    # Show one segment in each box
    col1.metric("Premium Big Spenders", counts["Premium Big Spenders"])
    col2.metric("Deal-Seeking Regulars", counts["Deal-Seeking Regulars"])
    col3.metric("Active Budget Shoppers", counts["Active Budget Shoppers"])
    col4.metric("Inactive / At-Risk", counts["Inactive / At-Risk"])


    # ---------- Part 2: Average behavior table ----------
    st.subheader("Average behavior per segment")

    # Columns we want to show in the table
    table_cols = behavior_cols + ["Response"]

    # Average of each column for each segment
    summary = df.groupby("Segment")[table_cols].mean()

    # Round to 1 decimal place so it is easy to read
    summary = summary.round(1)

    # Show the table
    st.dataframe(summary)


    # ---------- Part 3: Response rate chart ----------
    st.subheader("Campaign response rate (%)")

    # Average response for each segment (for example, 0.24)
    response = df.groupby("Segment")["Response"].mean()

    # Turn it into a percent (0.24 becomes 24)
    response = response * 100

    # Show a bar chart
    st.bar_chart(response, horizontal=True)


with tab2:

    # ---------- Part 1: Intro ----------
    st.subheader("Find a customer's segment")
    st.write("Enter a customer's shopping details and click the button.")


    # ---------- Part 2: Input boxes ----------
    # Two columns so the form is not too long
    left, right = st.columns(2)

    income = left.number_input("Yearly income", min_value=0, value=52000, step=1000)
    total_spending = left.number_input("Total spending", min_value=0, value=400, step=50)
    recency = left.number_input("Days since last purchase", min_value=0, max_value=100, value=49)
    deals = left.number_input("Purchases with a discount", min_value=0, value=2)

    web = right.number_input("Web purchases", min_value=0, value=4)
    catalog = right.number_input("Catalog purchases", min_value=0, value=2)
    store = right.number_input("Store purchases", min_value=0, value=5)
    web_visits = right.number_input("Website visits per month", min_value=0, value=5)


    # ---------- Part 3: What the store should do for each segment ----------
    actions = {
        "Premium Big Spenders": "Offer VIP rewards, premium products and early access to new items.",
        "Deal-Seeking Regulars": "Send discount codes and online sale alerts.",
        "Active Budget Shoppers": "Offer low-cost bundles and loyalty points to slowly raise spending.",
        "Inactive / At-Risk": "Send a special 'we miss you' offer to win them back."
    }


    # ---------- Part 4: Predict when the button is clicked ----------
    if st.button("Find segment"):

        # Put the inputs in a table with the SAME column names and order as training
        new_customer = pd.DataFrame([{
            "Income": income,
            "Total_Spending": total_spending,
            "Recency": recency,
            "NumDealsPurchases": deals,
            "NumWebPurchases": web,
            "NumCatalogPurchases": catalog,
            "NumStorePurchases": store,
            "NumWebVisitsMonth": web_visits
        }])
        new_customer = new_customer[behavior_cols]

        # Scale it the same way as the training data
        new_customer_scaled = scaler.transform(new_customer)

        # Ask the model which cluster it belongs to (gives 0, 1, 2 or 3)
        cluster_number = kmeans.predict(new_customer_scaled)[0]

        # Turn the number into a name
        segment = segment_names[cluster_number]

        # Show the result
        st.success(f"This customer is in: **{segment}**")
        st.info(f"Suggested action: {actions[segment]}")