#!/usr/bin/env python3
"""
Test script to verify FinLagX setup
"""
import sys
import os

def test_imports():
    """Test all imports"""
    print("🔍 Testing imports...")
    try:
        from src.data_storage.database_setup import test_connection, get_engine
        from src.data_ingestion.market_data import download_asset_to_db
        from src.data_ingestion.macro_data import download_macro_indicator_to_db
        from src.data_ingestion.news_data import get_mongo_client
        print("  All imports successful!")
        return True
    except ImportError as e:
        print(f"  Import error: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing database connection...")
    try:
        from src.data_storage.database_setup import test_connection
        test_connection()
        return True
    except Exception as e:
        print(f"  Database connection failed: {e}")
        return False

def test_config_files():
    """Test config files"""
    print("\n🔍 Testing config files...")
    import yaml
    
    config_files = [
        "configs/config_market.yaml",
        "configs/config_macro.yaml",
        "configs/config_news.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            print(f"  {config_file} loaded successfully")
        except Exception as e:
            print(f"  {config_file} failed: {e}")
            return False
    return True

def test_single_download():
    """Test downloading a single asset"""
    print("\n🔍 Testing single asset download...")
    try:
        from src.data_ingestion.market_data import download_asset_to_db
        from src.data_storage.database_setup import get_engine
        from datetime import datetime, timedelta
        
        engine = get_engine()
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        result = download_asset_to_db("AAPL", "APPLE", "TEST", start_date, end_date, engine)
        if result is not None:
            print(f"  Test download successful: {len(result)} rows")
            return True
        else:
            print("  Test download returned None")
            return False
    except Exception as e:
        print(f"  Test download failed: {e}")
        return False

def main():
    """Run all tests"""
    print("  FinLagX Setup Test\n")
    
    tests = [
        test_imports,
        test_config_files,
        test_database_connection,
        test_single_download
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("  All tests passed! FinLagX is ready to run!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
