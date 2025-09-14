#!/usr/bin/env python3
"""
FinLagX Setup Script
"""
import os
import subprocess
import sys

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finlagx
DB_USER=postgres
DB_PASSWORD=finlagx_password

# MongoDB Configuration
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USER=admin
MONGO_PASSWORD=finlagx_mongo
MONGO_DB=finlagx_news

# API Keys (add your keys here)
FRED_API_KEY=your_fred_api_key_here
""")
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def setup_docker():
    """Setup Docker containers"""
    print("🐳 Setting up Docker containers...")
    
    # Stop and remove existing containers
    subprocess.run(['docker-compose', 'down'], capture_output=True)
    
    # Remove volumes for fresh start
    subprocess.run(['docker', 'volume', 'prune', '-f'], capture_output=True)
    
    # Start containers
    result = subprocess.run(['docker-compose', 'up', '-d', 'timescaledb', 'mongodb'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Docker containers started successfully")
        return True
    else:
        print(f"❌ Docker setup failed: {result.stderr}")
        return False

def wait_for_containers():
    """Wait for containers to be healthy"""
    print("⏳ Waiting for containers to be ready...")
    import time
    time.sleep(15)  # Wait 15 seconds for containers to start
    
    # Check if containers are running
    result = subprocess.run(['docker', 'ps', '--filter', 'name=finlagx'], 
                          capture_output=True, text=True)
    
    if 'finlagx_timescaledb' in result.stdout and 'finlagx_mongodb' in result.stdout:
        print("✅ Containers are running")
        return True
    else:
        print("❌ Containers not running properly")
        return False

def setup_database():
    """Setup database schema"""
    print("🗄️ Setting up database schema...")
    try:
        from src.data_storage.database_setup import create_database, setup_timescaledb, test_connection
        
        create_database()
        setup_timescaledb()
        test_connection()
        print("✅ Database setup completed")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 FinLagX Setup Script\n")
    
    steps = [
        ("Creating environment file", create_env_file),
        ("Setting up Docker", setup_docker),
        ("Waiting for containers", wait_for_containers),
        ("Setting up database", setup_database)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return False
    
    print("\n🎉 FinLagX setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your API keys to .env file (FRED_API_KEY)")
    print("2. Run: python test_setup.py")
    print("3. Run: python -m src.data_ingestion.run_data_pipeline")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
