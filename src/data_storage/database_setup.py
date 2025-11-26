import os
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# TimescaleDB Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'finlagx'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'finlagx_password')
}

def get_db_url():
    """Generate PostgreSQL connection URL"""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def get_mlflow_db_url():
    """Generate MLflow PostgreSQL connection URL"""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/mlflow"
    )

def get_engine():
    """Get SQLAlchemy engine for main database operations"""
    return create_engine(get_db_url())

def get_mlflow_engine():
    """Get SQLAlchemy engine for MLflow database"""
    return create_engine(get_mlflow_db_url())

def test_connection():
    """Test database connections."""
    print("\n--- Testing Database Connections ---")
    
    # Test main TimescaleDB
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"  Connected to FinLagX DB: {version.split(',')[0]}")
            
            # Check for TimescaleDB extension
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';"))
            if result.fetchone():
                print(f"  TimescaleDB extension is installed")
            else:
                print(f"⚠️ TimescaleDB extension not found")
    except Exception as e:
        print(f"  FinLagX Database Connection Failed: {e}")
        raise
    
    # Test MLflow database
    try:
        mlflow_engine = get_mlflow_engine()
        with mlflow_engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            print(f"  Connected to MLflow DB")
    except Exception as e:
        print(f"⚠️ MLflow Database not ready (will be created by MLflow): {e}")
    
    print("------------------------------------")

def check_tables():
    """Check which tables exist in the database"""
    engine = get_engine()
    
    print("\n--- Checking Database Tables ---")
    
    with engine.connect() as conn:
        # Get all tables
        result = conn.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """))
        
        tables = [row[0] for row in result]
        
        if tables:
            print(f"  Found {len(tables)} tables:")
            # Hypertables such as var_features can hold many chunks; counting them
            # requires grabbing locks on every chunk and can exhaust shared memory.
            heavy_tables = {"var_features"}
            
            for table in tables:
                try:
                    if table in heavy_tables:
                        estimate_sql = text("""
                            SELECT reltuples::bigint 
                            FROM pg_class 
                            WHERE relname = :table_name
                        """)
                        estimate = conn.execute(estimate_sql, {"table_name": table}).scalar()
                        count_display = f"~{estimate:,} (estimate)"
                    else:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = count_result.fetchone()[0]
                        count_display = f"{count:,}"
                    
                    print(f"   • {table}: {count_display} rows")
                except Exception as e:
                    print(f"   • {table}: ⚠️ count unavailable ({e})")
        else:
            print("⚠️ No tables found in database")
    
    print("------------------------------------")

def clean_raw_data():
    """Truncate raw data tables (market_data, macro_data)"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE market_data RESTART IDENTITY CASCADE;"))
            conn.execute(text("TRUNCATE TABLE macro_data RESTART IDENTITY CASCADE;"))
            conn.commit()
            print("🧹 Cleaned raw data tables (market_data, macro_data)")
    except Exception as e:
        print(f"  Error cleaning raw data: {e}")

def clean_processed_features():
    """Truncate processed feature tables"""
    engine = get_engine()
    
    feature_tables = [
        'market_features',
        'granger_results', 
        'var_features',
        'lstm_predictions'
    ]
    
    try:
        with engine.connect() as conn:
            for table in feature_tables:
                try:
                    conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
                    print(f"🧹 Cleaned {table}")
                except Exception as e:
                    print(f"⚠️ Could not clean {table}: {e}")
            conn.commit()
            print("  Cleaned all processed feature tables")
    except Exception as e:
        print(f"  Error cleaning processed features: {e}")

def clean_all_data():
    """Clean both raw and processed data"""
    print("\n🧹 Cleaning ALL data from database...")
    clean_raw_data()
    clean_processed_features()
    print("  All data cleaned\n")

def init_feature_store():
    """Initialize feature store tables if they don't exist"""
    from src.feature_store import FeatureStore
    
    print("\n🏗️ Initializing Feature Store...")
    fs = FeatureStore()
    fs.initialize_feature_store()
    print("  Feature Store ready\n")

if __name__ == "__main__":
    print("  FinLagX Database Setup")
    
    # Test connection
    test_connection()
    
    # Check tables
    check_tables()
    
    # Initialize feature store
    init_feature_store()
    
    print("\n  Database setup completed!")