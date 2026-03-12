import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------------------------

def data_summary(df):
    """Print shape, dtypes, and descriptive statistics."""
    print("Shape of the data:", df.shape)
    print("\nData types and missing values:")
    df.info()
    print("\nSummary Statistics:\n", df.describe())


def plot_distributions(df):
    """Histogram of key numerical columns."""
    numerical_cols = [c for c in ['Close', 'High', 'Low', 'Open', 'Volume', 'fedrete']
                      if c in df.columns]
    df[numerical_cols].hist(bins=20, figsize=(10, 8))
    plt.suptitle('Distributions of Numerical Columns')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """Correlation heatmap for key numerical columns."""
    numerical_cols = [c for c in ['Close', 'High', 'Low', 'Open', 'Volume', 'fedrete']
                      if c in df.columns]
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()


def plot_market_trends(df):
    """Plot close price with 50-day and 200-day moving averages."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.6)
    if '50-day MA' in df.columns:
        plt.plot(df.index, df['50-day MA'], label='50-day MA', linestyle='dashed')
    if '200-day MA' in df.columns:
        plt.plot(df.index, df['200-day MA'], label='200-day MA', linestyle='dashed')
    plt.legend()
    plt.title('Market Trends with Moving Averages')
    plt.show()


def plot_boxplots(df):
    """Boxplots for all numeric columns to visualise outliers."""
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = len(numeric_columns)
    num_rows = (num_cols // 5) + (num_cols % 5 > 0)
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
    axes = axes.flatten()
    for i, column in enumerate(numeric_columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


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
