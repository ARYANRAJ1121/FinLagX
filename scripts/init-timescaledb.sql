-- Initialize TimescaleDB extension and create schema for FINANCIAL DATA ONLY
-- News data is stored in MongoDB - not here!

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create market data table
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

-- Convert to hypertable
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_category ON market_data (category, time DESC);

-- Create macro data table
CREATE TABLE IF NOT EXISTS macro_data (
    time TIMESTAMPTZ NOT NULL,
    indicator VARCHAR(50) NOT NULL,
    value DECIMAL(15,6),
    PRIMARY KEY (time, indicator)
);

-- Convert to hypertable
SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);

-- Create index for macro data
CREATE INDEX IF NOT EXISTS idx_macro_indicator ON macro_data (indicator, time DESC);

-- Create a view for latest market prices
CREATE OR REPLACE VIEW latest_market_prices AS
SELECT DISTINCT ON (symbol)
    symbol,
    category,
    time,
    close_price,
    volume
FROM market_data
ORDER BY symbol, time DESC;

-- Create a view for recent market and macro summary
CREATE OR REPLACE VIEW recent_data_summary AS
SELECT 
    'market_data' as data_type,
    COUNT(*) as record_count,
    MAX(time) as latest_timestamp
FROM market_data 
WHERE time >= NOW() - INTERVAL '7 days'
UNION ALL
SELECT 
    'macro_data' as data_type,
    COUNT(*) as record_count,
    MAX(time) as latest_timestamp
FROM macro_data 
WHERE time >= NOW() - INTERVAL '30 days'
ORDER BY data_type;

-- Create a view for macro indicators summary
CREATE OR REPLACE VIEW macro_indicators_summary AS
SELECT DISTINCT ON (indicator)
    indicator,
    time as latest_date,
    value as latest_value
FROM macro_data
ORDER BY indicator, time DESC;

-- Insert some sample data for testing
INSERT INTO market_data (time, symbol, category, close_price, volume) 
VALUES (NOW(), 'SP500', 'equities', 4500.00, 1000000)
ON CONFLICT DO NOTHING;

INSERT INTO macro_data (time, indicator, value)
VALUES (NOW(), 'CPI', 3.2)
ON CONFLICT DO NOTHING;

-- Clean up any old news tables if they exist (migration helper)
DROP TABLE IF EXISTS news_data CASCADE;

-- Show created hypertables
SELECT hypertable_name, hypertable_schema FROM timescaledb_information.hypertables;

-- Add helpful comment
COMMENT ON DATABASE finlagx IS 'FinLagX Financial Database - Market & Macro Data Only (News data in MongoDB)';
COMMENT ON TABLE market_data IS 'OHLCV market data from multiple asset classes';
COMMENT ON TABLE macro_data IS 'Economic indicators from FRED and other sources';