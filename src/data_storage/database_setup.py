import os
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Database configuration
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

def create_database():
    """Create the FinLagX database if it doesn't exist"""
    try:
        # Connect to default postgres database first
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create database
        cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']};")
        print(f"✅ Created database: {DB_CONFIG['database']}")

    except psycopg2.errors.DuplicateDatabase:
        print(f"✅ Database {DB_CONFIG['database']} already exists")
    except Exception as e:
        print(f"❌ Error creating database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def setup_timescaledb():
    """Install TimescaleDB extension and create hypertables for financial data only"""
    engine = create_engine(get_db_url())

    try:
        with engine.connect() as conn:
            # Enable TimescaleDB extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            print("✅ TimescaleDB extension enabled")

            # Create market data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    category VARCHAR(20) NOT NULL,
                    open_price DECIMAL(15,4),
                    high_price DECIMAL(15,4),
                    low_price DECIMAL(15,4),
                    close_price DECIMAL(15,4),
                    adj_close DECIMAL(15,4),
                    volume BIGINT,
                    PRIMARY KEY (time, symbol)
                );
            """))

            # Convert to hypertable (TimescaleDB's special time-series table)
            try:
                conn.execute(text("SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);"))
                print("✅ Created market_data hypertable")
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"⚠️ Market hypertable creation: {e}")

            # Create macro data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    time TIMESTAMPTZ NOT NULL,
                    indicator VARCHAR(50) NOT NULL,
                    value DECIMAL(15,6),
                    PRIMARY KEY (time, indicator)
                );
            """))

            try:
                conn.execute(text("SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);"))
                print("✅ Created macro_data hypertable")
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"⚠️ Macro hypertable creation: {e}")

            conn.commit()
            print("✅ TimescaleDB schema created successfully (Financial data only)")
            print("📰 Note: News data is stored separately in MongoDB")

    except Exception as e:
        print(f"❌ Error setting up TimescaleDB: {e}")

def get_engine():
    """Get SQLAlchemy engine for database operations"""
    return create_engine(get_db_url())

def test_connection():
    """Test database connection and show what's stored here"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version}")

            # Check TimescaleDB
            result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"))
            ts_version = result.fetchone()
            if ts_version:
                print(f"✅ TimescaleDB version: {ts_version[0]}")
            else:
                print("⚠️ TimescaleDB not installed")

            # Show what data types are stored here
            print("\n📊 Data stored in TimescaleDB:")
            print("  • Market Data (OHLCV prices)")
            print("  • Macro Economic Indicators")
            print("📰 News data is stored in MongoDB")

    except Exception as e:
        print(f"❌ Connection failed: {e}")

def drop_news_tables_if_exist():
    """Clean up any existing news tables from PostgreSQL"""
    engine = get_engine()

    try:
        with engine.connect() as conn:
            # Drop news-related tables if they exist
            conn.execute(text("DROP TABLE IF EXISTS news_data CASCADE;"))
            print("🧹 Cleaned up any existing news tables from PostgreSQL")
            conn.commit()
    except Exception as e:
        print(f"⚠️ Error cleaning news tables: {e}")

def clean_database_tables():
    """Truncate all data from market and macro tables."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE market_data, macro_data RESTART IDENTITY CASCADE;"))
            conn.commit()
            print("🧹 Cleaned all market and macro data from TimescaleDB.")
    except Exception as e:
        print(f"❌ Error cleaning TimescaleDB tables: {e}")


if __name__ == "__main__":
    print("🚀 Setting up FinLagX TimescaleDB (Financial Data Only)...\n")

    create_database()
    drop_news_tables_if_exist()  # Clean up any old news tables
    setup_timescaledb()
    test_connection()

    print("\n✅ TimescaleDB setup completed!")
    print("📊 Ready for market and macro data")
    print("📰 Use MongoDB for news data")