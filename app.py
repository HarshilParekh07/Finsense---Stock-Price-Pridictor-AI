import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import onnxruntime as ort
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os

@st.cache_resource(show_spinner="Loading AI Model...")
def get_model():
    model_path = "finsense_model.onnx"

    if not os.path.exists(model_path):
        st.error("ONNX model not found: finsense_model.onnx")
        st.stop()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    return session


@st.cache_data(ttl=3600)
def get_recommendations(symbol):
    try:
        return yf.Ticker(symbol).recommendations
    except:
        return None


@st.cache_data(ttl=3600)
def get_major_holders(symbol):
    try:
        return yf.Ticker(symbol).major_holders
    except:
        return None

@st.cache_data(ttl=3600)
def get_institutional_holders(symbol):
    try:
        return yf.Ticker(symbol).institutional_holders
    except:
        return None

@st.cache_data(ttl=3600)
def get_recommendations_summary(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.recommendations_summary
    except:
        return None



st.title('Finsense - Stock Analysis')
st.subheader("Finsense Dashboard")
user_input = st.text_input(" Enter Stock Symbol (Example: RELIANCE.NS, AAPL, TCS.NS)")

if not user_input.strip():
    st.info("Please enter a stock symbol to begin analysis.")
    st.stop()


def safe_download(symbol):
    try:
        symbol = symbol.strip().upper()

        # Auto-fix Indian market symbols
        if "." not in symbol and len(symbol) <= 6:
            symbol += ""

        end_date = datetime.today().strftime("2026-01-16")

        df = yf.download(
            tickers=[symbol],
            start="2000-01-01",
            end=end_date,
            group_by="ticker",
            progress=False,
            threads=False
        )

        # Flatten multi-index columns
        if isinstance(df.columns, tuple) or hasattr(df.columns, "levels"):
            df = df[symbol]
            df.columns = [c.capitalize() for c in df.columns]

        if df.empty:
            return None, symbol

        return df, symbol

    except Exception as e:
        return None, str(e)

ticker = user_input
@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info
    except Exception:
        return {}

info = get_company_info(user_input)

df, symbol = safe_download(user_input)

if df is None:
    st.error(f"No market data found for: {symbol}")
    st.info("Try: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT")
    st.stop()

def safe_get(data, key, default="Not Available"):
    try:
        return data.get(key, default)
    except:
        return default
# UI for Glass effect of Company Profile
st.markdown("""
<style>
.profile-card {
    background: rgba(20, 25, 35, 0.75);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
.title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}
.label {
    color: #9aa4b2;
    font-size: 13px;
}
.value {
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


#  UI for COMPANY PROFILE
st.markdown("<div class='title'> Company Profile</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label"> Name</div>
        <div class="value">{safe_get(info, "shortName")}</div>
    </div><br>
    <div class="metric-box">
        <div class="label"> Sector</div>
        <div class="value">{safe_get(info, "sector")}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label"> Industry</div>
        <div class="value">{safe_get(info, "industry")}</div>
    </div><br>
    <div class="metric-box">
        <div class="label"> Country</div>
        <div class="value">{safe_get(info, "country")}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label"> Website</div>
        <div class="value">{safe_get(info, "website")}</div>
    </div><br>
    <div class="metric-box">
        <div class="label"> Currency</div>
        <div class="value">{safe_get(info, "currency")}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("Key Company Metrics")

m1, m2 = st.columns(2)

with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Market Capitalization</div>
        <div class="value">{safe_get(info, "marketCap", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Total Employees</div>
        <div class="value">{safe_get(info, "fullTimeEmployees", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("Business Summary")

summary = safe_get(info, "longBusinessSummary", "No business summary available.")
st.write(summary)


st.markdown("</div>", unsafe_allow_html=True)

# UI for Glass effect of Ratios
st.markdown("""
<style>
.ratio-box {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    transition: 0.2s ease;
}
.ratio-box:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
}
.ratio-label {
    font-size: 13px;
    color: #9aa4b2;
}
.ratio-value {
    font-size: 20px;
    font-weight: 700;
}
.good { color: #00e676; }
.neutral { color: #ffd166; }
.bad { color: #ff5252; }
</style>
""", unsafe_allow_html=True)

# UI for RATIOS
def format_ratio(val, pct=False):
    try:
        val = float(val)
        return f"{val*100:.2f}%" if pct else f"{val:.2f}"
    except:
        return "N/A"

def classify_ratio(label, val):
    try:
        v = float(val)
        if "PE" in label:
            return "good" if v < 20 else "neutral" if v < 35 else "bad"
        if "Return" in label or "Profit" in label:
            return "good" if v > 0.15 else "neutral" if v > 0.05 else "bad"
        if "Debt" in label:
            return "good" if v < 1 else "neutral" if v < 2 else "bad"
    except:
        pass
    return "neutral"

st.markdown("## Valuation & Performance Ratios")

ratios = {
    "PE Ratio": safe_get(info, "trailingPE"),
    "Forward PE": safe_get(info, "forwardPE"),
    "Price / Book": safe_get(info, "priceToBook"),
    "Dividend Yield": safe_get(info, "dividendYield"),
    "Return on Equity": safe_get(info, "returnOnEquity"),
    "Debt to Equity": safe_get(info, "debtToEquity"),
    "Profit Margin": safe_get(info, "profitMargins")
}

cols = st.columns(3)
i = 0

for label, raw in ratios.items():
    col = cols[i % 3]
    pct = label in ["Dividend Yield", "Return on Equity", "Profit Margin"]
    value = format_ratio(raw, pct)
    css_class = classify_ratio(label, raw)

    with col:
        st.markdown(f"""
        <div class="ratio-box">
            <div class="ratio-label">{label}</div>
            <div class="ratio-value {css_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    i += 1

# UI for Shareholding Pattern
st.markdown("""
<style>
.holder-card {
    background: rgba(20,25,35,0.75);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)


#UI for SHAREHOLDING
st.markdown("Shareholding Pattern")

summary_cols = st.columns(3)

summary_cols = st.columns(3)

with summary_cols[0]:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label"> Institutional Ownership %</div>
        <div class="value">{safe_get(info, "heldPercentInstitutions", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)

with summary_cols[1]:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Insider Ownership %</div>
        <div class="value">{safe_get(info, "heldPercentInsiders", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)

with summary_cols[2]:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Float Shares</div>
        <div class="value">{safe_get(info, "floatShares", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)


# Major Holders
major_df = get_major_holders(user_input)

if major_df is not None:
    major_df = major_df.copy()


    st.markdown("Major Holders Overview")

    # Normalize structure safely
    if major_df.shape[1] == 1:
        major_df = major_df.reset_index()
        major_df.columns = ["Category", "Value"]
    elif major_df.shape[1] >= 2:
        major_df = major_df.iloc[:, :2]
        major_df.columns = ["Category", "Value"]
    else:
        st.warning("Major holders data format not supported.")
        major_df = None

    if major_df is not None:
        import plotly.express as px

        fig = px.bar(
            major_df,
            x="Category",
            y="Value",
            title="Major Ownership Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(major_df, use_container_width=True, hide_index=True)
else:
    st.info("Major holders data not available.")


# Pie Chart for Holders
inst_df = get_institutional_holders(user_input)

if inst_df is not None:
    inst_df = inst_df.copy()

    st.markdown("Institutional Holders Distribution")

    if "Shares" in inst_df.columns and "Holder" in inst_df.columns:
        fig2 = px.pie(
            inst_df.head(10),
            names="Holder",
            values="Shares",
            title="Top 10 Institutional Holders",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        inst_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Institutional holders data not available.")

#UI for Glass effect of Ananlyst Recommendation
st.markdown("""
<style>
.verdict-card {
    background: rgba(20,25,35,0.8);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}
.verdict-buy { color: #00e676; }
.verdict-hold { color: #ffd166; }
.verdict-sell { color: #ff5252; }
</style>
""", unsafe_allow_html=True)

# UI for ANALYST RATING
import plotly.express as px
import pandas as pd

st.markdown("Analyst Recommendations")

# CASE 1: Detailed analyst logs
rec_df = get_recommendations(user_input)

if rec_df is not None:
    rec_df = rec_df.copy()

    if not rec_df.empty:
        rec_df = rec_df.tail(10).reset_index()

        # Normalize rating column
        rating_col = None
        for c in rec_df.columns:
            if c.lower() in ["to grade", "to", "rating", "recommendation", "action"]:
                rating_col = c
                break

        if rating_col:
            rec_df.rename(columns={rating_col: "Rating"}, inplace=True)

            summary = rec_df["Rating"].astype(str).str.title().value_counts()

            buy = sum(summary.get(x, 0) for x in ["Buy", "Strong Buy"])
            hold = summary.get("Hold", 0)
            sell = sum(summary.get(x, 0) for x in ["Sell", "Strong Sell"])

            total = max(buy + hold + sell, 1)

            verdict = "BUY" if buy > max(hold, sell) else "SELL" if sell > max(buy, hold) else "HOLD"
            confidence = round((max(buy, hold, sell) / total) * 100, 1)

            st.markdown(f"""
            <div class="verdict-card">
                <h2>{verdict}</h2>
                <p>Analyst Confidence: {confidence}%</p>
                <p>Based on last {total} analyst updates</p>
            </div>
            """, unsafe_allow_html=True)

            fig = px.histogram(rec_df, x="Rating", title="Recommendation Trend", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(rec_df, use_container_width=True, hide_index=True)
            st.stop()

# CASE 2: Summary trend table 
trend_df = get_recommendations_summary(user_input)

if trend_df is not None:
    trend_df = trend_df.copy()

    st.markdown("Analyst Consensus Trend (Monthly)")

    # Latest row = current month
    latest = trend_df.iloc[0]

    buy = latest["strongBuy"] + latest["buy"]
    hold = latest["hold"]
    sell = latest["sell"] + latest["strongSell"]

    total = max(buy + hold + sell, 1)

    verdict = "BUY" if buy > max(hold, sell) else "SELL" if sell > max(buy, hold) else "HOLD"
    confidence = round((max(buy, hold, sell) / total) * 100, 1)

    st.markdown(f"""
    <div class="verdict-card">
        <h2>{verdict}</h2>
        <p>Analyst Confidence: {confidence}%</p>
        <p>Based on {total} analyst opinions (latest month)</p>
    </div>
    """, unsafe_allow_html=True)

    # Trend chart
    melted = trend_df.melt(
        id_vars=["period"],
        value_vars=["strongBuy", "buy", "hold", "sell", "strongSell"],
        var_name="Rating",
        value_name="Count"
    )

    fig2 = px.line(
        melted,
        x="period",
        y="Count",
        color="Rating",
        title="Analyst Recommendation Trend Over Time",
        template="plotly_dark",
        markers=True
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(trend_df, use_container_width=True, hide_index=True)

else:
    st.info("Analyst recommendation data not available for this ticker.")

#Describing Data
st.subheader('Current Stock Data')
st.write(df.describe())

#Visualizations of MA 

# Moving Averages
df["MA100"] = df["Close"].rolling(100).mean()
df["MA200"] = df["Close"].rolling(200).mean()

st.subheader("Closing Price with Technical Indicators")

fig = go.Figure()

# Closing Price Line
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    name="Closing Price",
    mode="lines",
    line=dict(color="#00E5FF", width=2.5),
    hovertemplate="Close: ₹%{y:.2f}<extra></extra>"
))

# 100 MA
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MA100"],
    name="100 MA",
    mode="lines",
    line=dict(color="#7F00FF", width=2, dash="dash"),
    hovertemplate="100 MA: ₹%{y:.2f}<extra></extra>"
))

# 200 MA
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MA200"],
    name="200 MA",
    mode="lines",
    line=dict(color="#FF3D3D", width=2),
    hovertemplate="200 MA: ₹%{y:.2f}<extra></extra>"
))

# Modern Finance Layout
fig.update_layout(
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    title=dict(
        text="Closing Price with MA",
        x=0,
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Time",
        showgrid=False
    ),
    yaxis=dict(
        title="Price (₹)",
        gridcolor="rgba(255,255,255,0.1)"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=50, b=20)
)

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)
    
# Spliting Data into Tranning and Testing

# AI Price Prediction Engine
st.markdown("AI Closing Price Prediction Model")
run_ai = st.button("Run Price Prediction")

if run_ai:

    # Data Preparation
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_scaler = scaler.fit_transform(data_training)

    # Load Model
    model = get_model()
    st.success("AI Model Loaded — Running Predictions")

    # Testing Dataset
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Prediction

    input_name = model.get_inputs()[0].name
    x_test_onnx = x_test.astype(np.float32)

    y_predicted = model.run(
        None,
        {input_name: x_test_onnx}
    )[0]


    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Visualization
    st.subheader("Actual Price vs Predicted Price")

    x_axis = np.arange(len(y_test))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_test.flatten(),
        name="Actual Price",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_predicted.flatten(),
        name="Predicted Price",
        mode="lines",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        title="Actual vs Predicted Price"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Run Price Prediction' to load the AI model and generate predictions.")

