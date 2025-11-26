#!/usr/bin/env python3
"""
FinLagX Setup Script
Sets up the complete environment: Docker, Database, Feature Store
"""
import os
import subprocess
import sys
import time

def create_env_file():
    """Create .env file if it doesn't exist"""
    try:
        if not os.path.exists('.env'):
            print("📝 Creating .env file...")
            with open('.env', 'w') as f:
                f.write("""# FinLagX Environment Configuration

# TimescaleDB Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finlagx
DB_USER=postgres
DB_PASSWORD=finlagx_password

# MLflow Configuration  
MLFLOW_TRACKING_URI=http://localhost:5000

# API Keys
FRED_API_KEY=your_fred_api_key_here
""")
            print("  Created .env file")
            print("⚠️  IMPORTANT: Add your FRED_API_KEY to the .env file!")
        else:
            print("  .env file already exists")
        return True
    except Exception as e:
        print(f"  Error creating .env file: {e}")
        return False

def check_docker():
    """Check if Docker is installed and running"""
    print("🐳 Checking Docker...")
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Docker installed: {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("  Docker daemon is running")
                return True
            else:
                print("  Docker daemon is not running. Please start Docker Desktop.")
                return False
        else:
            print("  Docker not found. Please install Docker Desktop.")
            return False
    except FileNotFoundError:
        print("  Docker not found. Please install Docker Desktop.")
        return False

def setup_docker():
    """Setup Docker containers"""
    print("\n🐳 Setting up Docker containers...")

    # Stop and remove existing containers
    print("   Stopping existing containers...")
    subprocess.run(['docker-compose', 'down'], capture_output=True)

    # Remove volumes for fresh start
    print("   Cleaning volumes...")
    subprocess.run(['docker', 'volume', 'prune', '-f'], capture_output=True)

    # Start containers
    print("   Starting containers...")
    result = subprocess.run(['docker-compose', 'up', '-d'],
                          capture_output=True, text=True)

    if result.returncode == 0:
        print("  Docker containers started successfully")
        return True
    else:
        print(f"  Docker setup failed: {result.stderr}")
        return False

def wait_for_containers():
    """Wait for containers to be healthy"""
    print("\n⏳ Waiting for containers to be ready...")
    
    max_wait = 60  # Maximum wait time in seconds
    wait_interval = 5
    elapsed = 0
    
    while elapsed < max_wait:
        time.sleep(wait_interval)
        elapsed += wait_interval
        
        result = subprocess.run(['docker', 'ps', '--filter', 'name=finlagx'],
                              capture_output=True, text=True)
        
        if 'finlagx_timescaledb' in result.stdout and 'finlagx_mlflow' in result.stdout:
            # Check if containers are healthy
            health_check = subprocess.run(
                ['docker', 'ps', '--filter', 'health=healthy', '--filter', 'name=finlagx'],
                capture_output=True, text=True
            )
            
            if 'finlagx_timescaledb' in health_check.stdout:
                print(f"  Containers are running and healthy (waited {elapsed}s)")
                return True
        
        print(f"   Still waiting... ({elapsed}s)")
    
    print("⚠️ Containers started but health check timed out")
    print("   This is usually fine - continuing with setup...")
    return True

def setup_database():
    """Setup database schema and feature store"""
    print("\n🗄️ Setting up database schema...")
    try:
        from src.data_storage.database_setup import test_connection, init_feature_store
        
        # Test connection
        test_connection()
        
        # Initialize feature store
        init_feature_store()
        
        print("  Database setup completed")
        return True
    except Exception as e:
        print(f"  Database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_mlflow():
    """Verify MLflow is accessible"""
    print("\n📊 Verifying MLflow...")
    try:
        import requests
        import time
        
        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.get('http://localhost:5000/health', timeout=5)
                if response.status_code == 200:
                    print("  MLflow server is running at http://localhost:5000")
                    return True
            except:
                if i < max_retries - 1:
                    print(f"   MLflow not ready yet, retrying... ({i+1}/{max_retries})")
                    time.sleep(10)
        
        print("⚠️ MLflow server not responding")
        print("   You can start it manually later with: docker-compose up mlflow")
        return True  # Don't fail setup for this
        
    except Exception as e:
        print(f"⚠️ Could not verify MLflow: {e}")
        return True  # Don't fail setup for this

def show_next_steps():
    """Show next steps to user"""
    print("\n" + "="*80)
    print("  FINLAGX SETUP COMPLETED!")
    print("="*80)
    
    print("\n  Next Steps:")
    print("   1. Add your FRED API key to .env file:")
    print("      FRED_API_KEY=your_actual_key_here")
    print()
    print("   2. Run the complete pipeline:")
    print("      python run_complete_pipeline.py")
    print()
    print("   3. Access the tools:")
    print("      • MLflow UI:      http://localhost:5000")
    print("      • PgAdmin:        http://localhost:8080 (admin@finlagx.com / admin123)")
    print()
    print("   4. Start modeling:")
    print("      python -m src.modeling.granger_causality")
    print("      python -m src.modeling.lstm_model")
    print()
    print("="*80)

def main():
    """Main setup function"""
    print("\n" + "  "*20)
    print("FINLAGX SETUP SCRIPT")
    print("  "*20 + "\n")

    steps = [
        ("Creating environment file", create_env_file),
        ("Checking Docker", check_docker),
        ("Setting up Docker containers", setup_docker),
        ("Waiting for containers", wait_for_containers),
        ("Setting up database", setup_database),
        ("Verifying MLflow", verify_mlflow)
    ]

    for step_name, step_func in steps:
        print(f"\n{'='*80}")
        print(f"{step_name}...")
        print(f"{'='*80}")
        
        if not step_func():
            print(f"\n  Setup failed at: {step_name}")
            print("   Please fix the error and run setup again.")
            return False

    show_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)