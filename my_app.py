from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ALPHAVANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_PREMIUM_API_KEY"]

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def infer_stock_symbols(company_names):
    """
    Uses the LLM to infer stock symbols from a list of company names.
    """
    prompt = f"What is the primary stock symbol for the following companies: {', '.join(company_names)}? Respond ONLY with one symbol per company, in a comma-separated list. If there are multiple symbols for a company, return the most common or primary symbol only."
    response = llm.invoke(prompt)

    return [symbol.strip() for symbol in response.content.split(",")]


def parse_time_period(time_text):
    """
    Uses an LLM to convert a natural language time period into structured start and end dates.
    If the end date is not provided, it is set to today.
    """
    today = datetime.today().strftime('%Y-%m-%d')

    prompt = (
        f"Today's date is {today}. Convert the following time period into a start and end date for a stock price query: "
        f"'{time_text}'. Respond ONLY with the start and end dates in YYYY-MM-DD format, separated by a comma. "
        f"If the end date is not provided, use today's date as the end date. No other text.")

    response = llm.invoke(prompt)
    dates = response.content.strip().split(',')

    if len(dates) == 1:
        start_date = dates[0].strip()
        end_date = today
    else:
        start_date, end_date = dates[0].strip(), dates[1].strip()

    return start_date, end_date


def get_inputs(query):
    prompt = f"Extract the company names and time period from the following query: '{query}'. Respond in the format: 'Companies: <name1>, <name2>, ...; Period: <period>'."
    response = llm.invoke(prompt)
    companies_part, period = response.content.split("; ")
    company_names = [name.strip() for name in companies_part.replace("Companies: ", "").split(",")]
    period = period.replace("Period: ", "")

    stock_symbols = infer_stock_symbols(company_names)
    if not stock_symbols:
        return f"Could not infer stock symbols for {company_names}."

    start_date, end_date = parse_time_period(period)
    if not start_date or not end_date:
        return f"Could not parse time period: {period}."

    return stock_symbols, start_date, end_date


def preprocess_stock(data):
    # Extract the dates and the '4. close' prices
    dates = list(data.keys())
    close_prices = [data[date]['4. close'] for date in dates]
    df = pd.DataFrame({'date': dates, 'close': close_prices})

    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    df = df.sort_values(by='date').reset_index(drop=True)

    return df


def _fetch_stock_data(ticker, start_date):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}'
    r = requests.get(url)
    data = r.json()

    ts = data.get('Time Series (Daily)', [])
    df = preprocess_stock(ts)
    filtered_df = df[df["date"] >= start_date]

    print(ticker)
    print(filtered_df.head())
    return filtered_df


def fetch_stock_prices(query):
    """
    Fetches historical stock prices for multiple companies and time periods.
    Uses the LLM to infer stock symbols and parse the time period.
    """
    stock_symbols, start_date, end_date = get_inputs(query)
    fig = go.Figure()

    for ticker in stock_symbols:
        df = _fetch_stock_data(ticker, start_date)

        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines+markers", name=ticker))

    fig.update_layout(title="Stock Price Over Time", xaxis_title="Date", yaxis_title="Stock Price (USD)",
                      xaxis=dict(tickangle=45), legend_title="Stock Symbols", template="plotly_white")

    st.plotly_chart(fig)
    return f"Stock prices for {stock_symbols} from {start_date} plotted successfully."


def transform_date(input_date, time="0000"):
    """
    Transforms a date string from 'YYYY-MM-DD' to 'YYYYMMDDTHHMM' format.
    """
    date_obj = datetime.strptime(input_date, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%Y%m%d") + "T" + time
    return formatted_date


def compute_daily_avg_sentiment(df):
    df.columns = ['time_published', 'sentiment_score']
    df.loc[:, 'date'] = df['time_published'].str[:8]  # Extract YYYYMMDD part
    daily_avg = df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_avg['date'] = pd.to_datetime(daily_avg['date'], format='%Y%m%d').dt.date
    return daily_avg


def _sentiment_data(ticker, start_date):
    function = 'NEWS_SENTIMENT'
    start_date = transform_date(start_date)

    url = f'https://www.alphavantage.co/query?function={function}&tickers={ticker}&time_from={start_date}&limit={1000}&apikey={ALPHAVANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data.get('feed', [])
    try:
        df = pd.DataFrame(articles)
        df_subset = df[['time_published', 'overall_sentiment_score']]
        daily_avg = compute_daily_avg_sentiment(df_subset)
        return daily_avg
    except:
        "No articles found."


def classify_sentiment(score):
    """
    Classifies sentiment score into categories.
    """
    if score <= -0.35:
        return "bearish"
    elif -0.35 < score <= -0.15:
        return "somewhat bearish"
    elif -0.15 < score < 0.15:
        return "neutral"
    elif 0.15 <= score < 0.35:
        return "somewhat bullish"
    else:
        return "bullish"


def news_sentiment(query):
    stock_symbols, start_date, end_date = get_inputs(query)
    dic = defaultdict(str)

    for ticker in stock_symbols:
        daily_avg = _sentiment_data(ticker, start_date)
        average_sentiment = daily_avg["sentiment_score"].mean()
        sentiment_category = classify_sentiment(average_sentiment)
        dic[ticker] = sentiment_category

    return ', '.join([f"The sentiment for {ticker} is {sentiment}" for ticker, sentiment in dic.items()])


def analyze_sentiment_vs_price(query):
    stock_symbols, start_date, end_date = get_inputs(query)
    df1 = _sentiment_data(stock_symbols[0], start_date)
    df2 = _fetch_stock_data(stock_symbols[0], start_date)

    if df1 is None:
        return "STOP! Request ENDED, it cannot be completed."
    df1["date"] = pd.to_datetime(df1["date"])
    df2["date"] = pd.to_datetime(df2["date"])
    merged_df = pd.merge(df1, df2, on='date')

    merged_df['sentiment_category'] = merged_df['sentiment_score'].apply(classify_sentiment)
    fig = px.scatter(merged_df, x="sentiment_score", y="close", color="sentiment_category",
                     labels={"sentiment_score": "Sentiment Score", "close": "Close Price"}, )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), legend=dict(x=1.05, y=1),
                      # Move legend outside the plot
                      margin=dict(l=0, r=0, t=30, b=0)  # Adjust margins
                      )
    st.plotly_chart(fig, use_container_width=True)
    return f"Sentiment vs. Price trends for {stock_symbols} from {start_date} to {end_date} plotted successfully."


def stocks_correlation(query):
    stock_symbols, start_date, end_date = get_inputs(query)

    stock_string = ','.join(stock_symbols)
    url = f'https://www.alphavantage.co/query?function=ANALYTICS_FIXED_WINDOW&SYMBOLS={stock_string}&RANGE={start_date}&RANGE={end_date}&INTERVAL=DAILY&OHLC=close&CALCULATIONS=CORRELATION&apikey={ALPHAVANTAGE_API_KEY}'
    r = requests.get(url)
    data = r.json()

    print(data)
    print(stock_symbols)

    n = len(stock_symbols)

    lower_triangular = data['payload']['RETURNS_CALCULATIONS']['CORRELATION']['correlation']
    # Convert the lower triangular matrix to a full symmetric correlation matrix
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            corr_matrix[i, j] = lower_triangular[i][j]
            corr_matrix[j, i] = lower_triangular[i][j]

    fig = px.imshow(corr_matrix, text_auto=True, zmin=-1, zmax=1, x=stock_symbols, y=stock_symbols, )
    st.plotly_chart(fig, theme="streamlit")
    return f"Correlation matrix for {stock_symbols} from {start_date} to {end_date} plotted successfully."


news_sentiment_tool = Tool("News Sentiment Agent", func=news_sentiment,
                           description="Fetches sentiment scores from Alpha Vantage for multiple companies. DO NOT USE FOR correlation matrix. Uses LLM to infer stock symbols and parse time periods.")

stock_tool = Tool(name="Stock Price Agent", func=fetch_stock_prices,
                  description="Fetches stock from Alpha Vantage for multiple companies. DO NOT USE FOR correlation matrix. Uses LLM to infer stock symbols and parse time periods.")

sentiment_price_tool = Tool(name="Sentiment vs Stock Agent", func=analyze_sentiment_vs_price,
                            description="Sentiment vs Stock Price from Alpha Vantage for a single company. Use this tool to compare sentiment scores with stock price movements over a specific time period. Uses LLM to infer stock symbols and parse time periods."

                            )

correlation_tool = Tool(name="Stock Correlation Agent", func=stocks_correlation,
                        description="Compute correlation matrix for multiple companies. USE FOR correlation matrix. Uses LLM to infer stock symbols and parse time periods.")

agents = [stock_tool, news_sentiment_tool, sentiment_price_tool, correlation_tool]
multi_agent = initialize_agent(tools=agents, llm=llm, agent="zero-shot-react-description",
                               # Enables auto-selection of tools
                               )

# Add nice layout

st.title("MarketMinds")

with st.sidebar:
    st.write("""MarketMinds utilizes **AI agents** to analyze financial sentiment and stock price trends, helping you gain deeper market insights.""")

    with st.expander("Features"):
        st.write("""
        - **📈 Sentiment Analysis**: AI agents track sentiment trends for companies like Tesla, Nvidia, and Palantir.
        - **💰 Stock Price Tracking**: View stock prices for Tesla, IBM, Apple, and more over custom periods.
        - **📊 Correlation Matrix**: Calculate relationships between daily closing prices of multiple stocks.
        - **🔍 Sentiment vs. Price Trends**: Compare sentiment data with stock price movements over time.
        """)

    with st.expander("Example Usage"):
        st.write("""
        - **What are the sentiments for Nvidia and Palantir?**
        - **What are the stocks for IBM and Apple over the last two weeks?**
        - **For Apple, Ford, and Amazon, calculate the correlation matrix based on daily close prices between 2024-07-01 and 2024-12-31.**
        - **Show the sentiment vs. price trends for Tesla over the last 3 weeks.**
        """)


user_query = st.text_input("💬 **Enter your query:**", placeholder="What are the sentiments for Nvidia and Palantir?")

try:
    if user_query:
        st.cache_data.clear()
        st.write(f"🧠 Analyzing query: '{user_query}'")
        response = multi_agent.invoke(user_query)
        st.write(response.get("output", "No output key found in the response"))

        st.cache_data.clear()
except Exception as e:
    st.error(f"An error occurred.")
