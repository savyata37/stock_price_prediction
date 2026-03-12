import pandas as pd
import numpy as np


def clean_columns(df):
    """Strip whitespace from all column names."""
    df.columns = df.columns.str.strip()
    return df


def drop_nulls_and_duplicates(df):
    """Drop fully-null columns and duplicate rows."""
    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    return df


def convert_date(df):
    """Parse the Date column, sort chronologically, and set as index."""
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df


def fill_missing_values(df):
    """Fill NaN values in engineered columns with their column medians."""
    columns_to_fill = [
        '7-day MA', '30-day MA', 'Rolling_Median_7', 'Rolling Std (30-day)', '50-day MA',
        '200-day MA', 'MA_7_50_diff', 'MA_30_200_diff', 'MA_7_slope', 'MA_30_slope',
        'Pct Change', 'Rolling Volatility', 'Rolling_Volatility_7', 'Rolling_Volatility_14',
        'ATR_14', 'Rolling_Mean_Close_Open', 'Rolling_Std_Close_Open', 'Cumulative Returns',
        'Cumulative_Returns_7', 'Log_Return', 'Cumulative_Return_100', 'Price_Momentum_100',
        'Price_to_Cumulative_Returns', 'Close_lag_1', 'Close_lag_2', 'Close_lag_5', 'Close_lag_10',
        'Open_lag_1', 'Open_lag_2', 'Open_lag_3', 'Open_lag_5', 'Open_lag_10', 'Price_Momentum_7',
        'Price_Momentum_30', 'RSI', '5-day MA', '10-day MA', 'Bollinger_Upper', 'Bollinger_Lower',
        'Volume_MA_7', 'Volume_Rate_of_Change', 'Price_to_7MA', 'Price_to_30MA',
        'Volatility-to-Change', 'Open_Close_Momentum_7', 'Open_Close_Momentum_30',
        'Rolling_Open_MA_7', 'Rolling_Open_MA_30', 'Rolling_Open_MA_50',
    ]
    # Only fill columns that actually exist in the dataframe
    existing = [c for c in columns_to_fill if c in df.columns]
    medians = df[existing].median().to_numpy()
    for i, col in enumerate(existing):
        df[col] = df[col].fillna(medians[i])
    return df


def handle_skewness(df):
    """Apply log1p transformation to highly skewed features (skewness > 1)."""
    skewness = df.skew()
    cols_to_transform = skewness[skewness > 1].index.tolist()
    print(f"Applying log1p to skewed columns: {cols_to_transform}")
    df[cols_to_transform] = df[cols_to_transform].apply(np.log1p)

    # Also apply log1p to these core columns to reduce skewness
    core_cols = ['Volume', 'Close', 'High', 'Low', 'Pct Change']
    existing_core = [c for c in core_cols if c in df.columns]
    for col in existing_core:
        df[col] = np.log1p(df[col])
    return df


def remove_outliers(df):
    """Cap extreme outliers using the IQR method."""
    columns_to_check = [c for c in ['Volume', 'Close', 'High', 'Low', 'Pct Change',
                                     'Rolling Std (30-day)', 'Cumulative Returns']
                        if c in df.columns]
    Q1 = df[columns_to_check].quantile(0.25)
    Q3 = df[columns_to_check].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[columns_to_check] = np.where(df[columns_to_check] < lower_bound, lower_bound,
                                    df[columns_to_check])
    df[columns_to_check] = np.where(df[columns_to_check] > upper_bound, upper_bound,
                                    df[columns_to_check])
    return df


def remove_highly_correlated(df, threshold=0.90):
    """Drop features whose absolute pairwise correlation exceeds the threshold."""
    corr_matrix = df.corr()
    high_corr_var = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_var.add(corr_matrix.columns[i])
    df = df.drop(columns=high_corr_var)
    print("Remaining features after removing highly correlated ones:")
    print(df.columns.tolist())
    return df


def drop_inf_nan(df):
    """Replace infinite values with NaN and drop remaining NaN rows."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def run_initial_cleaning(df):
    """Steps applied before feature engineering: columns, nulls, duplicates, date."""
    df = clean_columns(df)
    df = drop_nulls_and_duplicates(df)
    print(df.dtypes)
    df = convert_date(df)
    return df


def run_post_feature_cleaning(df):
    """Steps applied after feature engineering: fill NaNs, skewness, outliers, correlation."""
    df = fill_missing_values(df)
    df = handle_skewness(df)
    df = remove_outliers(df)
    df = remove_highly_correlated(df)
    df = drop_inf_nan(df)
    return df
