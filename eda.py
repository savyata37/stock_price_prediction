import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
except ImportError:
    px = None
    go = None
    pio = None
    make_subplots = None


# ---------------------------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------------------------

def data_summary(df):
    """Print shape, dtypes, and descriptive statistics."""
    print("Shape of the data:", df.shape)
    print("\nData types and missing values:")
    df.info()
    print("\nSummary Statistics:\n", df.describe())


def _existing_columns(df, columns):
    return [column for column in columns if column in df.columns]


def _require_plotly():
    if px is None or go is None or pio is None or make_subplots is None:
        raise ImportError(
            "Plotly is required for EDA plotting functions. Install plotly to use interactive notebook charts."
        )


def _show_figure(fig):
    _require_plotly()
    pio.show(fig, renderer='notebook_connected')

def plot_boxplots(df):
    """Interactive boxplots with 3 plots per row."""
    _require_plotly()

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    cols_per_row = 3
    rows = math.ceil(len(numeric_columns) / cols_per_row)

    fig = make_subplots(
        rows=rows,
        cols=cols_per_row,
        subplot_titles=numeric_columns
    )

    for i, column in enumerate(numeric_columns):
        row = i // cols_per_row + 1
        col_pos = i % cols_per_row + 1

        fig.add_trace(
            go.Box(x=df[column],
            name="",         
            showlegend=False),
            row=row,
            col=col_pos
        )

    fig.update_layout(
        height=350 * rows,
        width=1000,
        title_text="Boxplots of Numerical Features",
        template="plotly_white",
        showlegend=False
    )

    fig.show()

def plot_correlation_heatmap(df):
    """Interactive correlation heatmap for key numerical columns."""
    _require_plotly()
    numerical_cols = _existing_columns(df, ['Close', 'High', 'Low', 'Open', 'Volume', 'fedrete'])
    if len(numerical_cols) < 2:
        return

    corr_matrix = df[numerical_cols].corr().round(2)
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlation Heatmap'
    )
    fig.update_layout(template='plotly_white', height=600)
    _show_figure(fig)



def plot_market_trends(df):
    """Interactive close-price trend plot with moving averages."""
    _require_plotly()
    if 'Close' not in df.columns:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    if '50-day MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['50-day MA'],
            mode='lines',
            name='50-day MA',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    if '200-day MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['200-day MA'],
            mode='lines',
            name='200-day MA',
            line=dict(color='#2ca02c', width=2, dash='dot')
        ))
    fig.update_layout(
        template='plotly_white',
        title='Market Trends with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=500
    )
    _show_figure(fig)

def plot_distributions(df):
    """Interactive histograms with 2 plots per row."""
    _require_plotly()

    numerical_cols = [c for c in ['Close', 'High', 'Low', 'Open', 'Volume', 'fedrete']
                      if c in df.columns]

    cols_per_row = 2
    rows = math.ceil(len(numerical_cols) / cols_per_row)

    fig = make_subplots(
        rows=rows,
        cols=cols_per_row,
        subplot_titles=numerical_cols
    )

    for i, col in enumerate(numerical_cols):
        row = i // cols_per_row + 1
        col_pos = i % cols_per_row + 1

        fig.add_trace(
            go.Histogram(x=df[col], nbinsx=30, name=col),
            row=row,
            col=col_pos
        )

    fig.update_layout(
        height=350 * rows,
        width=900,
        title_text="Distributions of Numerical Columns",
        template="plotly_white",
        showlegend=False
    )

    fig.show()

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def engineer_features(df):
    """Add all derived features used for modelling."""
    # --- Trend & Time Series ---
    df['7-day MA'] = df['Close'].rolling(window=7).mean()
    df['30-day MA'] = df['Close'].rolling(window=30).mean()
    df['Rolling_Median_7'] = df['Close'].rolling(window=7).median()
    df['Rolling Std (30-day)'] = df['Close'].rolling(window=30).std()
    df['50-day MA'] = df['Close'].rolling(window=50).mean()
    df['200-day MA'] = df['Close'].rolling(window=200).mean()
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['MA_7_50_diff'] = df['7-day MA'] - df['50-day MA']
    df['MA_30_200_diff'] = df['30-day MA'] - df['200-day MA']
    df['MA_7_slope'] = df['7-day MA'].diff()
    df['MA_30_slope'] = df['30-day MA'].diff()

    # --- Market Sentiment & Shock Events ---
    df['Pct Change'] = df['Close'].pct_change() * 100
    df['Price_Change_Open_Close'] = df['Close'] - df['Open']
    df['Pct_Change_Open_Close'] = df['Price_Change_Open_Close'] / df['Open'] * 100
    threshold = df['Pct Change'].std() * 2
    df['Shock Event'] = (df['Pct Change'].abs() > threshold).astype(int)

    # --- Market Volatility & Stability ---
    df['Rolling Volatility'] = df['Close'].rolling(window=30).std()
    df['Rolling_Volatility_7'] = df['Close'].rolling(window=7).std()
    df['Rolling_Volatility_14'] = df['Close'].rolling(window=14).std()
    df['ATR'] = df['High'] - df['Low']
    df['ATR_14'] = df['ATR'].rolling(window=14).mean()
    df['Close_Open_diff'] = df['Close'] - df['Open']
    df['Rolling_Mean_Close_Open'] = df['Close_Open_diff'].rolling(window=30).mean()
    df['Rolling_Std_Close_Open'] = df['Close_Open_diff'].rolling(window=30).std()

    # --- Market Performance ---
    df['Cumulative Returns'] = (1 + df['Pct Change'] / 100).cumprod()
    df['Cumulative_Returns_7'] = df['Close'].pct_change(7).cumsum()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Cumulative_Return_100'] = (1 + df['Pct Change'] / 100).cumprod()
    df['Price_Momentum_100'] = df['Close'].pct_change(100)
    df['Price_to_Cumulative_Returns'] = df['Close'] / df['Cumulative Returns']

    # --- Predictive Features ---
    for lag in [1, 2, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag)

    df['Price_Momentum_7'] = df['Close'].pct_change(7)
    df['Price_Momentum_30'] = df['Close'].pct_change(30)
    df['RSI'] = compute_rsi(df, window=14)
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['5-day MA'] = df['Close'].rolling(window=5).mean()
    df['10-day MA'] = df['Close'].rolling(window=10).mean()
    df['Bollinger_Upper'] = (df['Close'].rolling(window=20).mean()
                             + 2 * df['Close'].rolling(window=20).std())
    df['Bollinger_Lower'] = (df['Close'].rolling(window=20).mean()
                             - 2 * df['Close'].rolling(window=20).std())
    df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['Volume_Rate_of_Change'] = df['Volume'].pct_change()
    df['Volume_to_Open_Ratio'] = df['Volume'] / df['Open']
    df['Price_to_7MA'] = df['Close'] / df['7-day MA']
    df['Price_to_30MA'] = df['Close'] / df['30-day MA']
    df['Volatility-to-Change'] = df['Rolling Std (30-day)'] / df['Pct Change'].abs()
    df['Open_Close_Momentum_7'] = (df['Open'] - df['Close']).pct_change(7)
    df['Open_Close_Momentum_30'] = (df['Open'] - df['Close']).pct_change(30)
    df['Open_to_Close_Ratio'] = df['Open'] / df['Close']
    df['Close_to_Open_Ratio'] = df['Close'] / df['Open']
    df['Rolling_Open_MA_7'] = df['Open'].rolling(window=7).mean()
    df['Rolling_Open_MA_30'] = df['Open'].rolling(window=30).mean()
    df['Rolling_Open_MA_50'] = df['Open'].rolling(window=50).mean()

    return df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_eda(df):
    """Run full EDA (summary + plots) and return dataframe with engineered features."""
    data_summary(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    df = engineer_features(df)
    plot_market_trends(df)
    # Boxplot for outlier visualisation (shown BEFORE IQR capping in the cleaning step)
    plot_boxplots(df)
    return df
