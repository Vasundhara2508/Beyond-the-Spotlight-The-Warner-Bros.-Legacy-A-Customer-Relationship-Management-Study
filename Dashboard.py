import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Image, Spacer
from reportlab.lib.pagesizes import letter
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WBD · CRM Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()
# ─────────────────────────────────────────────
# COLOR CONSTANTS (REQUIRED 🔥)
# ─────────────────────────────────────────────
DARK   = "#0A0A10"
PANEL  = "#12121C"
GOLD   = "#F5C842"
RED    = "#E8233A"
BLUE   = "#3B82F6"
GREEN  = "#10B981"
PURPLE = "#A855F7"
MUTED  = "#8888AA"
TEXT   = "#E8E8F0"
BORDER = "#2A2A3E"

PALETTE = [
    GOLD, RED, BLUE, GREEN, PURPLE,
    "#F97316", "#06B6D4", "#EC4899"
]

# ─────────────────────────────────────────────
#  THEME HELPER  — call after every plot
# ─────────────────────────────────────────────
def apply_theme(fig, ax_list=None):
    fig.patch.set_facecolor(PANEL)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(GOLD)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(axis="both", labelsize=9)
    return fig


def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf


def create_pdf(fig_list):
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    elements = []

    for fig in fig_list:
        img = fig_to_image(fig)
        elements.append(Image(img, width=500, height=300))
        elements.append(Spacer(1, 20))

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer


# ─────────────────────────────────────────────
#  LOAD REAL DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    # ── WBD stock ──────────────────────────────────
    wbd = pd.read_csv("WBD.csv")
    wbd["Date"] = pd.to_datetime(
        wbd["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    wbd = wbd.sort_values("Date")

    # ── Warner Bros movies ─────────────────────────
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[
        [
            "title",
            "release_date",
            "revenue",
            "budget",
            "popularity",
            "vote_average",
            "production_companies",
        ]
    ]
    movies = movies[movies["production_companies"].str.contains("Warner", na=False)]
    movies = movies[movies["revenue"] > 0]
    movies["release_year"] = pd.to_datetime(
        movies["release_date"], errors="coerce"
    ).dt.year

    # ── Netflix churn ──────────────────────────────
    netflix = pd.read_csv("Netflix_customer_churn.csv")
    netflix = netflix.fillna(netflix.median(numeric_only=True))

    # ── Survey ─────────────────────────────────────
    survey = pd.read_csv("Survey.csv")
    survey.columns = survey.columns.str.strip()
    survey.columns = survey.columns.str.replace("'", "'")

    return wbd, movies, netflix, survey


wbd, movies, netflix, survey = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="nav-label"> WBD Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")

    # 🔥 STATE
    if "section" not in st.session_state:
        st.session_state.section = "Overview"

    def nav_button(label):
        if st.button(label, use_container_width=True):
            st.session_state.section = label

    # 🔥 BUTTON NAV
    nav_button("Overview")
    nav_button("Stock Analysis")
    nav_button("Franchise")
    nav_button("Churn Model")
    nav_button("Survey")

    section = st.session_state.section

    st.markdown("---")
    st.markdown(
        '<p style="font-size:.75rem;color:#8888AA;text-align:center;">'
        'Warner Bros Discovery · CRM Study<br/>Data Science Dashboard</p>',
        unsafe_allow_html=True
    )
# ═══════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════
if section == "Overview":
    fig_list = []
    st.markdown(
        """
    <div class="page-hero">
        <div class="hero-title">WBD CRM Analytics</div>
        <div class="hero-sub">Warner Bros Discovery · Comprehensive Business Intelligence Dashboard</div>
        <span class="hero-tag">Stock Market</span>
        <span class="hero-tag">Franchise Revenue</span>
        <span class="hero-tag">Churn Prediction</span>
        <span class="hero-tag">Survey Intelligence</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Stock Price",
            f"${wbd['Close'].iloc[-1]:.2f}",
            f"{wbd['Close'].iloc[-1]-wbd['Close'].iloc[-2]:+.2f}",
        )
    with c2:
        st.metric("Total WB Movies", str(len(movies)))
    with c3:
        st.metric("Churn Rate", f"{netflix['churned'].mean()*100:.1f}%")
    with c4:
        st.metric("Avg Revenue", f"${movies['revenue'].mean()/1e6:.0f}M")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            '<div class="section-header">Stock Closing Price</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(wbd["Date"], wbd["Close"], color=GOLD, lw=1.5)
        ax.fill_between(wbd["Date"], wbd["Close"], alpha=0.08, color=GOLD)
        ax.set_xlabel("Year")
        ax.set_ylabel("USD")
        ax.set_title("WBD Closing Price")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown(
            '<div class="section-header">Top 5 Franchise Revenue</div>',
            unsafe_allow_html=True,
        )
        top5 = movies.nlargest(5, "revenue")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(
            top5["title"],
            top5["revenue"] / 1e6,
            color=PALETTE[:5],
            edgecolor="none",
            height=0.55,
        )
        ax.set_xlabel("Revenue (USD M)")
        ax.set_title("Top 5 WB Movies")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown(
        '<div class="section-header">Dashboard Modules</div>', unsafe_allow_html=True
    )
    m1, m2, m3, m4 = st.columns(4)
    for col, title, desc, color in [
        (
            m1,
            "Stock Analysis",
            "Price, Volume & Volatility trends for WBD equity",
            "card-gold",
        ),
        (
            m2,
            "Franchise",
            "Revenue, ratings & audience metrics for WB films",
            "card-red",
        ),
        (
            m3,
            "Churn Model",
            "ML models predicting Netflix subscriber churn",
            "card-blue",
        ),
        (m4, "Survey", "Brand perception & CRM gap analysis", "card-green"),
    ]:
        with col:
            st.markdown(
                f'<div class="card {color}">'
                f'<strong style="color:#F5C842">{title}</strong><br>'
                f'<span style="font-size:.8rem;color:#8888AA">{desc}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
    pdf = create_pdf(fig_list)
    st.download_button(
        label="Download Report as PDF",
        data=pdf,
        file_name="Overview.pdf",
        mime="application/pdf",
    )

# ═══════════════════════════════════════════════════════
#  STOCK ANALYSIS
# ═══════════════════════════════════════════════════════
if section == "Stock Analysis":
    fig_list = []
    st.markdown(
        """
    <div class="page-hero">
        <div class="hero-title">Stock Market Analysis</div>
        <div class="hero-sub">Warner Bros Discovery · Equity Performance & Volatility</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latest Close", f"${wbd['Close'].iloc[-1]:.2f}")
    with c2:
        st.metric("52-Wk High", f"${wbd['Close'].max():.2f}")
    with c3:
        st.metric("52-Wk Low", f"${wbd['Close'].min():.2f}")
    with c4:
        st.metric("Avg Volatility", f"{wbd['Volatility'].mean():.2f}")

    # ── G1  Closing Price ──────────────────────────────
    st.markdown(
        '<div class="section-header">Closing Price Trend</div>', unsafe_allow_html=True
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(wbd["Date"], wbd["Close"], color=GOLD, lw=1.6)
    ax.fill_between(wbd["Date"], wbd["Close"], alpha=0.1, color=GOLD)
    ax.set_title("Warner Bros Discovery — Stock Closing Price")
    ax.set_xlabel("Year")
    ax.set_ylabel("Stock Price (USD)")
    apply_theme(fig, [ax])
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    # ── G2  Volume ─────────────────────────────────────
    with col1:
        st.markdown(
            '<div class="section-header">Trading Volume</div>', unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(wbd["Date"], wbd["Volume"], color=BLUE, lw=1.3)
        ax.fill_between(wbd["Date"], wbd["Volume"], alpha=0.1, color=BLUE)
        ax.set_title("Trading Volume Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Volume")
        apply_theme(fig, [ax])
        st.pyplot(fig)
        plt.close()

    # ── G3  Moving Averages ────────────────────────────
    with col2:
        st.markdown(
            '<div class="section-header">Moving Averages</div>', unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(
            wbd["Date"],
            wbd["Close"],
            color=TEXT,
            lw=1.0,
            alpha=0.5,
            label="Close Price",
        )
        ax.plot(wbd["Date"], wbd["MA_5"], color=GOLD, lw=1.5, label="5-Day MA")
        ax.plot(wbd["Date"], wbd["MA_20"], color=RED, lw=1.5, label="20-Day MA")
        ax.set_title("Stock Price with Moving Averages")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (USD)")
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        apply_theme(fig, [ax])
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # ── G4  Volatility ─────────────────────────────────
    with col3:
        st.markdown(
            '<div class="section-header">Stock Volatility</div>', unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(wbd["Date"], wbd["Volatility"], color=PURPLE, lw=1.3)
        ax.fill_between(wbd["Date"], wbd["Volatility"], alpha=0.15, color=PURPLE)
        ax.set_title("Stock Volatility Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── G5  Price Distribution ──────────────────────────
    with col4:
        st.markdown(
            '<div class="section-header">Price Distribution</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(
            wbd["Close"],
            bins=30,
            kde=True,
            ax=ax,
            color=GOLD,
            edgecolor=PANEL,
            line_kws={"color": RED, "lw": 2},
        )
        ax.set_title("Distribution of WBD Stock Prices")
        ax.set_xlabel("Stock Price")
        ax.set_ylabel("Frequency")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── G6  Correlation Heatmap ─────────────────────────
    st.markdown(
        '<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        wbd.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        linewidths=0.4,
        linecolor=BORDER,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Between Financial Indicators")
    apply_theme(fig, [ax])
    fig_list.append(fig)
    st.pyplot(fig)
    plt.close()
    pdf = create_pdf(fig_list)
    st.download_button(
        label="Download Report as PDF",
        data=pdf,
        file_name="STOCK ANALYSIS.pdf",
        mime="application/pdf",
    )
# ═══════════════════════════════════════════════════════
#  FRANCHISE
# ═══════════════════════════════════════════════════════
if section == "Franchise":
    fig_list = []
    st.markdown(
        """
    <div class="page-hero">
        <div class="hero-title">Franchise Performance</div>
        <div class="hero-sub">Warner Bros · Box Office Revenue, Ratings & Audience Demand</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Films", len(movies))
    with c2:
        st.metric("Top Revenue", f"${movies['revenue'].max()/1e6:.0f}M")
    with c3:
        st.metric("Avg Rating", f"{movies['vote_average'].mean():.1f} / 10")
    with c4:
        st.metric("Avg Popularity", f"{movies['popularity'].mean():.1f}")

    top10 = movies.sort_values(by="revenue", ascending=False).head(10)

    col1, col2 = st.columns(2)

    # ── G1  Top Movies by Revenue ──────────────────────
    with col1:
        st.markdown(
            '<div class="section-header">Top 10 Movies by Revenue</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(
            top10["title"],
            top10["revenue"] / 1e6,
            color=PALETTE[: len(top10)],
            edgecolor="none",
            height=0.6,
        )
        ax.set_xlabel("Revenue (USD M)")
        ax.set_title("Top Warner Bros Movies by Box Office Revenue")
        for bar, val in zip(bars, top10["revenue"] / 1e6):
            ax.text(
                val + 2,
                bar.get_y() + bar.get_height() / 2,
                f"${val:.0f}M",
                va="center",
                fontsize=7.5,
                color=TEXT,
            )
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── G2  Revenue Trend ──────────────────────────────
    with col2:
        st.markdown(
            '<div class="section-header">Revenue Trend Over Years</div>',
            unsafe_allow_html=True,
        )
        trend = movies.groupby("release_year")["revenue"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            trend["release_year"],
            trend["revenue"] / 1e6,
            color=GOLD,
            lw=2,
            marker="o",
            ms=4,
        )
        ax.fill_between(
            trend["release_year"], trend["revenue"] / 1e6, alpha=0.1, color=GOLD
        )
        ax.set_title("Warner Bros Revenue Trend Over Years")
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Avg Revenue (USD M)")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # ── G3  Popularity vs Revenue ──────────────────────
    with col3:
        st.markdown(
            '<div class="section-header">Popularity vs Revenue</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        scatter = ax.scatter(
            movies["popularity"],
            movies["revenue"] / 1e6,
            c=movies["vote_average"],
            cmap="plasma",
            s=60,
            alpha=0.8,
            edgecolors=PANEL,
            linewidths=0.5,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=TEXT)
        cbar.set_label("Rating", color=TEXT)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
        cbar.outline.set_edgecolor(BORDER)
        ax.set_title("Popularity vs Revenue (Audience Demand)")
        ax.set_xlabel("Popularity Score")
        ax.set_ylabel("Revenue (USD M)")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── G4  Rating Distribution ────────────────────────
    with col4:
        st.markdown(
            '<div class="section-header">Movie Rating Distribution</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            movies["vote_average"],
            bins=20,
            kde=True,
            ax=ax,
            color=GREEN,
            edgecolor=PANEL,
            line_kws={"color": GOLD, "lw": 2},
        )
        ax.axvline(
            movies["vote_average"].mean(),
            color=RED,
            ls="--",
            lw=1.5,
            label=f"Mean: {movies['vote_average'].mean():.1f}",
        )
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        ax.set_title("Distribution of Warner Bros Movie Ratings")
        ax.set_xlabel("Vote Average")
        ax.set_ylabel("Count")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()
        pdf = create_pdf(fig_list)
        st.download_button(
            label="Download Report as PDF",
            data=pdf,
            file_name="FRANCHISE.pdf",
            mime="application/pdf",
        )

# ═══════════════════════════════════════════════════════
#  CHURN MODEL
# ═══════════════════════════════════════════════════════
if section == "Churn Model":
    fig_list = []
    st.markdown(
        """
    <div class="page-hero">
        <div class="hero-title">Netflix Churn Prediction</div>
        <div class="hero-sub">Machine Learning Models · Logistic Regression · Random Forest · XGBoost</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Encode on a copy so @cache_data stays clean
    nf = netflix.copy()
    le = LabelEncoder()
    for col in nf.select_dtypes("object").columns:
        nf[col] = le.fit_transform(nf[col])

    X = nf.drop("churned", axis=1)
    y = nf["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    with st.spinner("Training models…"):
        log = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(
            X_train, y_train
        )
        xgb = XGBClassifier(eval_metric="logloss", verbosity=0).fit(X_train, y_train)

    log_pred = log.predict(X_test)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    log_acc = accuracy_score(y_test, log_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Churn Rate", f"{y.mean()*100:.1f}%")
    with c2:
        st.metric("Logistic Reg Acc", f"{log_acc*100:.1f}%")
    with c3:
        st.metric("Random Forest Acc", f"{rf_acc*100:.1f}%")
    with c4:
        st.metric("XGBoost Acc", f"{xgb_acc*100:.1f}%")

    col1, col2 = st.columns(2)

    # ── Model Accuracy Comparison ──────────────────────
    with col1:
        st.markdown(
            '<div class="section-header">Model Accuracy Comparison</div>',
            unsafe_allow_html=True,
        )
        results = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
                "Accuracy": [log_acc, rf_acc, xgb_acc],
            }
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            results["Model"],
            results["Accuracy"] * 100,
            color=[RED, GOLD, GREEN],
            edgecolor="none",
            width=0.5,
        )
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")
        for bar, val in zip(bars, results["Accuracy"] * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                f"{val:.1f}%",
                ha="center",
                fontsize=10,
                color=TEXT,
                fontweight="bold",
            )
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── Feature Importance ─────────────────────────────
    with col2:
        st.markdown(
            '<div class="section-header">Feature Importance (Random Forest)</div>',
            unsafe_allow_html=True,
        )
        fi = pd.DataFrame(
            {"Feature": X.columns, "Importance": rf.feature_importances_}
        ).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            fi["Feature"],
            fi["Importance"],
            color=PALETTE[: len(fi)],
            edgecolor="none",
            height=0.55,
        )
        ax.set_title("Feature Importance for Netflix Churn")
        ax.set_xlabel("Importance Score")
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── ROC Curve ──────────────────────────────────────
    st.markdown(
        '<div class="section-header">ROC Curve Comparison</div>', unsafe_allow_html=True
    )
    log_prob = log.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    log_fpr, log_tpr, _ = roc_curve(y_test, log_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        log_fpr,
        log_tpr,
        color=RED,
        lw=2,
        label=f"Logistic Regression  (AUC = {roc_auc_score(y_test, log_prob):.2f})",
    )
    ax.plot(
        rf_fpr,
        rf_tpr,
        color=GOLD,
        lw=2,
        label=f"Random Forest        (AUC = {roc_auc_score(y_test, rf_prob):.2f})",
    )
    ax.plot(
        xgb_fpr,
        xgb_tpr,
        color=GREEN,
        lw=2,
        label=f"XGBoost              (AUC = {roc_auc_score(y_test, xgb_prob):.2f})",
    )
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set_title("ROC Curve — All Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    apply_theme(fig, [ax])
    fig_list.append(fig)
    st.pyplot(fig)
    plt.close()
    pdf = create_pdf(fig_list)
    st.download_button(
        label="Download Report as PDF",
        data=pdf,
        file_name="Churn Model.pdf",
        mime="application/pdf",
    )
# ═══════════════════════════════════════════════════════
#  SURVEY (FIXED + ROBUST)
# ═══════════════════════════════════════════════════════
if section == "Survey":
    fig_list = []
    st.markdown(
        """
    <div class="page-hero">
        <div class="hero-title">Survey Intelligence</div>
        <div class="hero-sub">Brand Perception · Personalization · Relationship Metrics</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── CLEAN COLUMN NAMES PROPERLY ────────────────────
    survey.columns = (
        survey.columns.str.strip()
        .str.replace("’", "'")  # fix smart quotes
        .str.replace("\n", " ")
    )

    # ── SMART COLUMN MATCHER (NO MORE KEY ERRORS) ──────
    def get_col(keyword):
        for col in survey.columns:
            if keyword.lower() in col.lower():
                return col
        st.error(f"Column not found for: {keyword}")
        st.write(survey.columns)
        return None

    Q_FAM = get_col("familiar")
    Q_POP = get_col("popularity")
    Q_PERS = get_col("personalize")
    Q_REL = get_col("relationship")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # ── BRAND FAMILIARITY ──────────────────────────────
    with col1:
        st.markdown(
            '<div class="section-header">Brand Familiarity</div>',
            unsafe_allow_html=True,
        )
        fam = survey[Q_FAM].value_counts()

        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.barh(
            fam.index,
            fam.values / fam.sum() * 100,
            color=PALETTE[: len(fam)],
            edgecolor="none",
        )

        ax.set_xlabel("Respondents (%)")
        ax.set_title("Brand Familiarity Distribution")

        plt.subplots_adjust(left=0.45)

        for bar, val in zip(bars, fam.values / fam.sum() * 100):
            ax.text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                color=TEXT,
                fontsize=10,
            )

        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── POPULARITY COMPARISON ──────────────────────────
    with col2:
        st.markdown(
            '<div class="section-header">Popularity vs Rivals</div>',
            unsafe_allow_html=True,
        )
        pop = survey[Q_POP].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax.pie(
            pop.values,
            labels=pop.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=PALETTE[: len(pop)],
            wedgeprops={"edgecolor": PANEL},
            pctdistance=0.75,
            labeldistance=1.05,  # 🔥 labels closer
        )
        for text in texts:
            text.set_color("white")
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_weight("bold")
            ax.set_title("WB vs Netflix & Disney Popularity", color=GOLD)
        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── PERSONALIZATION PERCEPTION ─────────────────────
    with col3:
        st.markdown(
            '<div class="section-header">Personalization Perception</div>',
            unsafe_allow_html=True,
        )

        pers = survey[Q_PERS].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            pers.index,
            pers.values / pers.sum() * 100,
            color=PALETTE[: len(pers)],
            edgecolor="none",
        )

        ax.set_ylabel("Respondents (%)")
        ax.set_title("Do Platforms Personalize Well?")

        for bar, val in zip(bars, pers.values / pers.sum() * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.1f}%",
                ha="center",
                color=TEXT,
            )

        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── RELATIONSHIP BUILDING ──────────────────────────
    with col4:
        st.markdown(
            '<div class="section-header">Long-term Relationship Builder</div>',
            unsafe_allow_html=True,
        )

        rel = survey[Q_REL].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            rel.index,
            rel.values / rel.sum() * 100,
            color=PALETTE[: len(rel)],
            edgecolor="none",
        )

        ax.set_ylabel("Respondents (%)")
        ax.set_title("Strongest Viewer Relationship")

        for bar, val in zip(bars, rel.values / rel.sum() * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.1f}%",
                ha="center",
                color=TEXT,
            )

        apply_theme(fig, [ax])
        fig_list.append(fig)
        st.pyplot(fig)
        plt.close()

    # ── CROSS TAB HEATMAP ──────────────────────────────
    st.markdown(
        '<div class="section-header">Familiarity vs Popularity</div>',
        unsafe_allow_html=True,
    )

    cross_tab = pd.crosstab(survey[Q_FAM], survey[Q_POP], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.3)

    ax.set_title("Familiarity vs Popularity (%)")

    apply_theme(fig, [ax])
    fig_list.append(fig)
    st.pyplot(fig)
    plt.close()
    pdf = create_pdf(fig_list)
    st.download_button(
        label="Download Report as PDF",
        data=pdf,
        file_name="survey_report.pdf",
        mime="application/pdf",
    )

    # ── CRM GAP TABLE (EXTRA INSIGHT) ──────────────────
    st.markdown(
        '<div class="section-header">CRM Gap Analysis</div>', unsafe_allow_html=True
    )

    crm_gap = pd.DataFrame(
        {
            "CRM Feature": [
                "Personalization",
                "Churn Prediction",
                "Customer Data",
                "Recommendations",
            ],
            "Netflix": [
                "Strong AI",
                "Predictive Models",
                "Unified Data",
                "Advanced ML",
            ],
            "Warner Bros": ["Limited", "Weak/None", "Fragmented", "Basic"],
        }
    )

    st.dataframe(crm_gap, use_container_width=True)
