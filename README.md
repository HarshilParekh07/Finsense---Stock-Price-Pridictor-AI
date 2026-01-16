# Finsense — AI-Powered Stock Analyzer

Finsense is a **fintech-grade, AI-powered stock analysis and prediction web application** built with **Streamlit, Yahoo Finance, Plotly, and deep learning (LSTM)**. It delivers a professional dashboard experience with real-time market data, valuation ratios, shareholding insights, analyst consensus, and price prediction using trained neural network models.

---

## Features

###  Market Data & Company Intelligence

* Real-time stock price download via **Yahoo Finance (yfinance)**
* Automated **Company Profile** (Sector, Industry, Country, Website, Currency)
* **Business Summary** from Yahoo Finance

### Valuation & Performance Analytics

* KPI-style **ratio dashboard**

  * PE Ratio
  * Forward PE
  * Price-to-Book
  * Dividend Yield
  * Return on Equity
  * Debt-to-Equity
  * Profit Margins
* Smart color classification (Good / Neutral / Risk)

### Shareholding Insights

* Institutional ownership %
* Insider ownership %
* Float shares
* **Major holders bar chart**
* **Top institutional holders pie chart**

### Analyst Recommendations

* Smart detection of Yahoo Finance data schema
* AI-style **BUY / HOLD / SELL verdict engine**
* Confidence score
* Monthly consensus trend visualization
* Interactive recommendation tables

### Technical Analysis

* 100-day & 200-day moving averages
* Interactive **Plotly candlestick-style price charts**

### AI Price Prediction

* LSTM-based deep learning model
* Predicts future closing prices
* Visual comparison:

  * Actual vs Predicted price

---

## Tech Stack

| Layer         | Technology                     |
| ------------- | ------------------------------ |
| Frontend      | Streamlit                      |
| Market Data   | yfinance (Yahoo Finance API)   |
| Visualization | Plotly, Matplotlib             |
| AI / ML       | TensorFlow, Keras, LSTM        |
| Data          | Pandas, NumPy                  |
| Deployment    | Streamlit Cloud / VPS / Docker |

---

## Project Structure

```
Finsense/
├── app.py                        # Main Streamlit application
├── keras_model.h5              # LSTM model for Close price prediction
├── keras_model_open.h5        # LSTM model for Open price prediction
├── LSTM Model Close.ipynb     # Training notebook (Close price)
├── LSTM Model Open.ipynb      # Training notebook (Open price)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

##  Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/finsense.git
cd finsense
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

##  Required Python Packages

If you don’t use `requirements.txt`, install manually:

```bash
pip install streamlit yfinance pandas numpy matplotlib plotly tensorflow scikit-learn keras
```

---

##  Supported Stock Symbols

| Market      | Example Symbols                    |
| ----------- | ---------------------------------- |
| India (NSE) | RELIANCE.NS, TCS.NS, INFY.NS       |
| USA         | AAPL, MSFT, TSLA                   |
| Global      | Any Yahoo Finance-supported ticker |

---

## AI Model Details

* Model Type: **LSTM (Long Short-Term Memory Neural Network)**
* Framework: **TensorFlow / Keras**
* Input Window: 100 days
* Training Data: Historical closing prices (2000–present)
* Output: Predicted future closing price

The model is pre-trained and loaded dynamically:

```python
model = load_model('keras_model.h5')
```

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository
4. Select `app.py`
5. Click **Deploy**

### VPS / Docker

```bash
docker build -t finsense .
docker run -d -p 8501:8501 finsense
```

---

## Screenshots

> Add screenshots here for:

* Dashboard
* AI Predictions
* Analyst Verdict Panel
* Shareholding Charts

---

##  Disclaimer

This application is for **educational and research purposes only**.

It does **NOT provide financial advice**. Market data and predictions may be inaccurate. Always consult a licensed financial advisor before making investment decisions.

---

##  Future Enhancements

* Price target vs analyst consensus
* AI-generated stock risk analysis
* Mobile app deployment
*  Multi-language UI
*  Stock alerts system

---

##  Author

**Harshil Parekh**

---

## Support

If you found this project useful:

> Star this repository on GitHub

---

## License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.
