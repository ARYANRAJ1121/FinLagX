"""
Dashboard Helper Functions for FinLagX Streamlit App
Utilities for data loading, processing, and visualization
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sqlalchemy import text
import streamlit as st
from src.data_storage.database_setup import get_engine
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Constants
project_root = Path(__file__).parent.parent
DATA_FOLDER = project_root / "data" / "results"
ASSETS = [
    "bitcoin", "crude_oil", "dax40", "dowjones", "eurusd", 
    "ftse100", "gbpusd", "gold", "nasdaq100", "nifty50", 
    "nikkei225", "silver", "sp500", "us10y", "usdcny"
]

ASSET_DISPLAY_NAMES = {
    "bitcoin": "Bitcoin (BTC)",
    "crude_oil": "Crude Oil (WTI)",
    "dax40": "DAX 40",
    "dowjones": "Dow Jones",
    "eurusd": "EUR/USD",
    "ftse100": "FTSE 100",
    "gbpusd": "GBP/USD",
    "gold": "Gold",
    "nasdaq100": "NASDAQ 100",
    "nifty50": "NIFTY 50",
    "nikkei225": "NIKKEI 225",
    "silver": "Silver",
    "sp500": "S&P 500",
    "us10y": "US 10Y Treasury",
    "usdcny": "USD/CNY"
}

CATEGORY_COLORS = {
    'EQUITIES': '#667EEA',
    'COMMODITIES': '#F6AD55',
    'FX': '#48BB78',
    'VOL_BONDS': '#FC8181',
    'CRYPTO': '#9F7AEA',
    'MACRO': '#ECC94B',
    'OTHER': '#A0AEC0'
}


# ==================== DATA LOADING ====================

def get_database_engine():
    """Get database engine without caching (engines can't be pickled)"""
    return get_engine()


@st.cache_data(ttl=300)
def load_granger_results_from_db():
    """Load latest Granger causality results from database"""
    try:
        engine = get_database_engine()
        query = """
        SELECT asset_x, asset_y, granger_score, p_value, optimal_lag
        FROM granger_results
        WHERE computed_date = (SELECT MAX(computed_date) FROM granger_results)
        AND is_significant = TRUE
        ORDER BY granger_score DESC
        """
        df = pd.read_sql(query, engine)
        engine.dispose()  # Clean up connection
        return df
    except Exception as e:
        st.error(f"Error loading Granger results: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_market_features_from_db(symbols=None, start_date=None, end_date=None):
    """Load market features from database"""
    try:
        from sqlalchemy import text
        engine = get_database_engine()
        
        # Build query with proper parameter binding
        conditions = ["1=1"]
        params = {}
        
        if symbols:
            # For PostgreSQL arrays, use IN clause instead of ANY
            placeholders = ','.join([f':symbol_{i}' for i in range(len(symbols))])
            conditions.append(f"symbol IN ({placeholders})")
            for i, symbol in enumerate(symbols):
                params[f'symbol_{i}'] = symbol
        
        if start_date:
            conditions.append("time >= :start_date")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("time <= :end_date")
            params['end_date'] = end_date
        
        query = f"SELECT * FROM market_features WHERE {' AND '.join(conditions)} ORDER BY symbol, time"
        
        df = pd.read_sql(text(query), engine, params=params)
        engine.dispose()  # Clean up connection
        return df
    except Exception as e:
        st.error(f"Error loading market features: {e}")
        return pd.DataFrame()


@st.cache_data
def load_asset_predictions(asset):
    """Load LSTM predictions CSV for a specific asset"""
    file_path = DATA_FOLDER / f"{asset}_predictions.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()


@st.cache_data
def load_asset_metrics(asset):
    """Load performance metrics CSV for a specific asset"""
    file_path = DATA_FOLDER / f"{asset}_metrics.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_data
def load_asset_relationships(asset):
    """Load lead-lag relationships CSV for a specific asset"""
    file_path = DATA_FOLDER / f"{asset}_leadlag_relationships.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_data
def load_asset_summary(asset):
    """Load summary CSV for a specific asset"""
    file_path = DATA_FOLDER / f"{asset}_summary.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_data
def load_all_metrics():
    """Load metrics for all assets"""
    all_metrics = []
    for asset in ASSETS:
        metrics = load_asset_metrics(asset)
        if not metrics.empty:
            metrics['Asset'] = ASSET_DISPLAY_NAMES.get(asset, asset)
            metrics['asset_id'] = asset
            all_metrics.append(metrics)
    
    if all_metrics:
        return pd.concat(all_metrics, ignore_index=True)
    return pd.DataFrame()


# ==================== SUMMARY CALCULATIONS ====================

def calculate_summary_stats():
    """Calculate overall summary statistics"""
    all_metrics = load_all_metrics()
    
    if all_metrics.empty:
        return {
            'total_assets': 0,
            'avg_accuracy': 0,
            'total_predictions': 0,
            'avg_rmse': 0,
            'best_asset': 'N/A',
            'worst_asset': 'N/A'
        }
    
    stats = {
        'total_assets': len(all_metrics),
        'avg_accuracy': all_metrics['Directional_Accuracy_%'].mean(),
        'total_predictions': all_metrics['Total_Predictions'].sum() if 'Total_Predictions' in all_metrics.columns else 0,
        'avg_rmse': all_metrics['RMSE'].mean(),
        'best_asset': all_metrics.loc[all_metrics['Directional_Accuracy_%'].idxmax(), 'Asset'] if len(all_metrics) > 0 else 'N/A',
        'worst_asset': all_metrics.loc[all_metrics['Directional_Accuracy_%'].idxmin(), 'Asset'] if len(all_metrics) > 0 else 'N/A'
    }
    
    return stats


def get_recent_predictions(n=10):
    """Get N most recent predictions across all assets"""
    recent = []
    
    for asset in ASSETS:
        preds = load_asset_predictions(asset)
        if not preds.empty:
            latest = preds.tail(n).copy()
            latest['Asset'] = ASSET_DISPLAY_NAMES.get(asset, asset)
            recent.append(latest)
    
    if recent:
        combined = pd.concat(recent, ignore_index=True)
        combined = combined.sort_values('Date', ascending=False).head(n)
        return combined
    
    return pd.DataFrame()


# ==================== VISUALIZATION HELPERS ====================

def create_prediction_chart(df, asset_name):
    """Create Plotly chart for actual vs predicted returns"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual_Return'],
        mode='lines',
        name='Actual',
        line=dict(color='#667EEA', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted_Return'],
        mode='lines',
        name='Predicted',
        line=dict(color='#F6AD55', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=f"{asset_name} - Actual vs Predicted Returns",
        xaxis_title="Date",
        yaxis_title="Return",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_error_distribution(df):
    """Create histogram of prediction errors"""
    fig = px.histogram(
        df,
        x='Prediction_Error',
        nbins=50,
        title='Prediction Error Distribution',
        labels={'Prediction_Error': 'Prediction Error'},
        color_discrete_sequence=['#9F7AEA']
    )
    
    fig.update_layout(
        template='plotly_white',
        height=350
    )
    
    return fig


def create_correlation_heatmap(assets):
    """Create correlation heatmap for selected assets"""
    returns_data = {}
    
    for asset in assets:
        preds = load_asset_predictions(asset)
        if not preds.empty:
            returns_data[ASSET_DISPLAY_NAMES.get(asset, asset)] = preds.set_index('Date')['Actual_Return']
    
    if len(returns_data) < 2:
        return None
    
    df = pd.DataFrame(returns_data)
    corr = df.corr()
    
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Asset Correlation Heatmap'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500
    )
    
    return fig


def create_performance_comparison(assets):
    """Create bar chart comparing asset performance"""
    metrics = []
    
    for asset in assets:
        m = load_asset_metrics(asset)
        if not m.empty:
            m['Asset'] = ASSET_DISPLAY_NAMES.get(asset, asset)
            metrics.append(m)
    
    if not metrics:
        return None
    
    df = pd.concat(metrics, ignore_index=True)
    
    fig = px.bar(
        df,
        x='Asset',
        y='Directional_Accuracy_%',
        title='Directional Accuracy Comparison',
        labels={'Directional_Accuracy_%': 'Accuracy (%)'},
        color='Directional_Accuracy_%',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    return fig


def create_time_series_chart(df, symbols, feature='returns'):
    """Create time series chart for market features"""
    fig = go.Figure()
    
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol]
        fig.add_trace(go.Scatter(
            x=symbol_data['time'],
            y=symbol_data[feature],
            mode='lines',
            name=symbol
        ))
    
    fig.update_layout(
        title=f'{feature.replace("_", " ").title()} Over Time',
        xaxis_title='Date',
        yaxis_title=feature.replace("_", " ").title(),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


# ==================== UTILITY FUNCTIONS ====================

def format_metric(value, metric_type='number'):
    """Format metric values for display"""
    if metric_type == 'percentage':
        return f"{value:.2f}%"
    elif metric_type == 'decimal':
        return f"{value:.4f}"
    elif metric_type == 'integer':
        return f"{int(value):,}"
    else:
        return f"{value:.2f}"


def get_available_assets():
    """Get list of assets with available data"""
    available = []
    for asset in ASSETS:
        if (DATA_FOLDER / f"{asset}_predictions.csv").exists():
            available.append(asset)
    return available


def apply_date_filter(df, start_date, end_date, date_column='Date'):
    """Filter dataframe by date range"""
    if start_date and end_date:
        mask = (df[date_column] >= pd.Timestamp(start_date)) & (df[date_column] <= pd.Timestamp(end_date))
        return df[mask]
    return df


def color_correct_prediction(val):
    """Color coding for correct/incorrect predictions"""
    if val == True or val == 'True':
        return 'background-color: #48BB78; color: white'
    else:
        return 'background-color: #FC8181; color: white'
