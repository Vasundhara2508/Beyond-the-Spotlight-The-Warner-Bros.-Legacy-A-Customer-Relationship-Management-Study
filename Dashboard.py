# =========================================================
# WARNER BROS CRM ANALYTICS DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Warner Bros CRM Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

body {
    background-color:#0f172a;
}

h1,h2,h3 {
    color:#1f2937;
}

.metric-container {
    background:#f1f5f9;
    padding:20px;
    border-radius:10px;
}

.sidebar .sidebar-content {
    background:#020617;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("🎬 CRM Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Warner Bros Franchise Analysis",
        "Netflix Churn Analysis",
        "Survey Insights",
        "Churn Prediction Simulator",
        "CRM Gap Analysis"
    ]
)


# =========================================================
# PAGE 1 — PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title("🎬 Warner Bros CRM Analytics")

    st.markdown("""
### Project Goal
Analyze **customer engagement and churn patterns** in the streaming industry
by comparing **Warner Bros and Netflix CRM strategies**.

### Key Components
• Warner Bros franchise performance  
• Netflix churn prediction modeling  
• Consumer perception survey  
• CRM gap analysis
""")



# =========================================================
# PAGE 2 — FRANCHISE ANALYSIS
# =========================================================

elif page == "Warner Bros Franchise Analysis":

    st.title("🎥 Warner Bros Franchise Performance")

    movies = pd.read_csv("tmdb_5000_movies.csv")

    movies = movies[['title','release_date','revenue','budget','popularity','vote_average','production_companies']]

    wb_movies = movies[movies['production_companies'].str.contains("Warner", na=False)]

    wb_movies = wb_movies[wb_movies['revenue'] > 0]

    wb_movies['release_year'] = pd.to_datetime(
        wb_movies['release_date'], errors='coerce'
    ).dt.year


    # FILTER

    year_filter = st.slider(
        "Filter Movies by Year",
        int(wb_movies['release_year'].min()),
        int(wb_movies['release_year'].max()),
        (2000,2020)
    )

    filtered = wb_movies[
        (wb_movies['release_year'] >= year_filter[0]) &
        (wb_movies['release_year'] <= year_filter[1])
    ]


    # TOP MOVIES

    st.subheader("Top Warner Bros Movies")

    top_movies = filtered.sort_values(
        by="revenue", ascending=False
    ).head(10)

    fig, ax = plt.subplots()

    sns.barplot(
        x="revenue",
        y="title",
        data=top_movies,
        ax=ax
    )

    st.pyplot(fig)


    # POPULARITY VS REVENUE

    st.subheader("Audience Demand")

    fig2, ax2 = plt.subplots()

    sns.scatterplot(
        x="popularity",
        y="revenue",
        data=filtered,
        ax=ax2
    )

    st.pyplot(fig2)



# =========================================================
# PAGE 3 — NETFLIX CHURN ANALYSIS
# =========================================================

elif page == "Netflix Churn Analysis":

    st.title("📊 Netflix Churn Analysis")

    netflix = pd.read_csv("Netflix_customer_churn.csv")

    st.write("Dataset Shape:", netflix.shape)


    # CHURN RATE

    churn_rate = netflix['churned'].value_counts(normalize=True)*100

    st.metric("Churn Rate %", round(churn_rate[1],2))


    # CHURN DISTRIBUTION

    fig, ax = plt.subplots()

    sns.countplot(
        x=netflix["churned"],
        ax=ax
    )

    st.pyplot(fig)



# =========================================================
# PAGE 4 — SURVEY ANALYSIS
# =========================================================

elif page == "Survey Insights":

    st.title("📋 Viewer Survey Insights")

    survey = pd.read_csv("Survey.csv")

    survey.columns = survey.columns.str.strip()

    st.write("Responses:", len(survey))


    st.subheader("Brand Familiarity")

    fig, ax = plt.subplots()

    sns.countplot(
        y=survey["How familiar are you with Warner Bros as a brand?"],
        order=survey["How familiar are you with Warner Bros as a brand?"].value_counts().index
    )

    st.pyplot(fig)



# =========================================================
# PAGE 5 — CHURN SIMULATOR
# =========================================================

elif page == "Churn Prediction Simulator":

    st.title("🤖 Churn Prediction Simulator")

    survey = pd.read_csv("Survey.csv")

    survey.columns = survey.columns.str.strip()


    familiarity_map = {
        "Very familiar":5,
        "Somewhat familiar":4,
        "Neutral":3,
        "Heard of it":2,
        "Not familiar":1
    }

    frequency_map = {
        "Daily":4,
        "Weekly":3,
        "Occasionally":2,
        "Rarely":1
    }


    survey["familiarity_score"] = survey[
        "How familiar are you with Warner Bros as a brand?"
    ].map(familiarity_map)


    survey["engagement_score"] = survey[
        "How often do you watch movies or web series?"
    ].map(frequency_map)


    survey["churn_risk"] = survey["familiarity_score"].apply(
        lambda x: 1 if x <=3 else 0
    )


    X = survey[["familiarity_score","engagement_score"]].dropna()

    y = survey["churn_risk"].dropna()


    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )


    model = LogisticRegression()

    model.fit(X_train,y_train)


    st.subheader("Simulate User Behavior")


    familiarity = st.slider("Brand Familiarity",1,5,3)

    engagement = st.slider("Engagement Level",1,4,2)


    prediction = model.predict([[familiarity,engagement]])

    if prediction[0] == 1:

        st.error("⚠ High Churn Risk")

    else:

        st.success("✅ Loyal Viewer")



# =========================================================
# PAGE 6 — CRM GAP ANALYSIS
# =========================================================

elif page == "CRM Gap Analysis":

    st.title("📉 Netflix vs Warner Bros CRM Gap")

    data = pd.DataFrame({

        "CRM Capability":[
            "Personalization",
            "Churn Prediction",
            "Customer Data Integration",
            "Recommendation Systems"
        ],

        "Netflix":[
            9,
            9,
            9,
            9
        ],

        "Warner Bros":[
            6,
            5,
            6,
            5
        ]

    })


    fig, ax = plt.subplots()

    data.set_index("CRM Capability").plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)