#!/usr/bin/env python3
"""
Database setup script for LandPPT
Installs dependencies and initializes a clean SQLite database
"""

import sys
import os
import subprocess
import asyncio
import logging
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def install_dependencies():
    """Install required database dependencies"""
    print("📦 Installing database dependencies...")
    
    dependencies = [
        "sqlalchemy>=2.0.0",
        "alembic>=1.13.0", 
        "aiosqlite>=0.19.0"
    ]
    
    try:
        # Try using uv first (preferred)
        result = subprocess.run(
            ["uv", "pip", "install"] + dependencies,
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Dependencies installed successfully with uv")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  uv not found, trying pip...")
        
        try:
            # Fallback to pip
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + dependencies,
                capture_output=True,
                text=True,
                check=True
            )
            print("✅ Dependencies installed successfully with pip")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print(f"Error output: {e.stderr}")
            return False

async def initialize_database():
    """Initialize the database tables"""
    print("🗄️  Initializing database...")
    
    try:
        from landppt.database.database import init_db
        
        # Initialize database
        await init_db()
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

async def test_database_connection():
    """Test database connection and basic operations with thorough cleanup"""
    print("🔍 Testing database connection...")

    test_project_id = None
    try:
        from landppt.database.service import DatabaseService
        from landppt.database.database import AsyncSessionLocal
        from landppt.api.models import PPTGenerationRequest

        # Create database session
        async with AsyncSessionLocal() as session:
            db_service = DatabaseService(session)

            # Test creating a project
            test_request = PPTGenerationRequest(
                scenario="test",
                topic="Database Test",
                requirements="Testing database functionality"
            )

            project = await db_service.create_project(test_request)
            test_project_id = project.project_id
            print(f"✅ Test project created: {project.project_id}")

            # Test retrieving the project
            retrieved_project = await db_service.get_project(project.project_id)
            if retrieved_project:
                print("✅ Project retrieval successful")
            else:
                print("❌ Project retrieval failed")
                return False

            # Test listing projects
            project_list = await db_service.list_projects(page=1, page_size=10)
            print(f"✅ Found {project_list.total} projects in database")

            # Clean up test project and all related data
            await db_service.project_repo.delete(project.project_id)
            print("✅ Test project cleaned up")

            # Verify cleanup was successful
            verify_project = await db_service.get_project(project.project_id)
            if verify_project is None:
                print("✅ Test project cleanup verified")
            else:
                print("⚠️  Test project cleanup incomplete")

        print("✅ Database connection test passed")
        return True

    except Exception as e:
        print(f"❌ Database connection test failed: {e}")

        # Emergency cleanup if test failed
        if test_project_id:
            try:
                print("🧹 Attempting emergency cleanup...")
                async with AsyncSessionLocal() as session:
                    db_service = DatabaseService(session)
                    await db_service.project_repo.delete(test_project_id)
                    print("✅ Emergency cleanup completed")
            except Exception as cleanup_error:
                print(f"❌ Emergency cleanup failed: {cleanup_error}")

        return False

def check_existing_database():
    """Check if database already exists"""
    db_path = "landppt.db"
    if os.path.exists(db_path):
        print(f"📋 Found existing database: {db_path}")
        return True
    else:
        print("📋 No existing database found")
        return False

def remove_existing_database():
    """Remove existing database file to ensure clean start"""
    db_path = "landppt.db"
    if os.path.exists(db_path):
        try:
            # Create backup before removal
            backup_path = f"{db_path}.backup.{int(time.time())}"
            os.rename(db_path, backup_path)
            print(f"📋 Existing database backed up to: {backup_path}")
            print(f"🗑️  Removed existing database: {db_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove existing database: {e}")
            return False
    else:
        print("📋 No existing database to remove")
        return True

async def clear_all_data():
    """Clear all data from database tables while keeping structure"""
    print("🧹 Clearing all existing data from database...")

    try:
        from landppt.database.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            # Get all table names
            result = await session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ))
            tables = [row[0] for row in result.fetchall()]

            if not tables:
                print("📋 No tables found to clear")
                return True

            # Disable foreign key constraints temporarily
            await session.execute(text("PRAGMA foreign_keys = OFF"))

            # Clear all tables
            cleared_count = 0
            for table in tables:
                if table != 'schema_migrations':  # Keep migration history
                    result = await session.execute(text(f"DELETE FROM {table}"))
                    cleared_count += result.rowcount
                    print(f"🧹 Cleared table: {table}")

            # Re-enable foreign key constraints
            await session.execute(text("PRAGMA foreign_keys = ON"))

            await session.commit()
            print(f"✅ Cleared {cleared_count} records from {len(tables)} tables")
            return True

    except Exception as e:
        print(f"❌ Failed to clear database data: {e}")
        return False

async def verify_clean_database():
    """Verify that database tables are empty"""
    try:
        from landppt.database.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            # Get all table names except migration table
            result = await session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'schema_migrations'"
            ))
            tables = [row[0] for row in result.fetchall()]

            total_records = 0
            for table in tables:
                result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                total_records += count
                if count > 0:
                    print(f"⚠️  Table {table} contains {count} records")

            if total_records == 0:
                print("✅ All data tables are empty")
                return True
            else:
                print(f"⚠️  Found {total_records} total records in database")
                return False

    except Exception as e:
        print(f"❌ Failed to verify database cleanliness: {e}")
        return False

async def main(clean_start=True):
    """Main setup function"""
    print("🚀 LandPPT Database Setup")
    print("=" * 50)

    # Check if database already exists
    db_exists = check_existing_database()

    # Handle existing database
    if db_exists and clean_start:
        print("\n🧹 Clean start requested - removing existing database...")
        if not remove_existing_database():
            print("❌ Failed to remove existing database")
            return False
        print()
    elif db_exists:
        print("\n🧹 Clearing existing data from database...")
        if not await clear_all_data():
            print("❌ Failed to clear existing data")
            return False
        print()

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please install manually:")
        print("uv pip install sqlalchemy>=2.0.0 alembic>=1.13.0 aiosqlite>=0.19.0")
        return False

    print()

    # Initialize database
    if not await initialize_database():
        print("\n❌ Database initialization failed")
        return False

    print()

    # Test database connection (with cleanup)
    if not await test_database_connection():
        print("\n❌ Database connection test failed")
        return False

    # Import default templates from examples (force import for setup)
    print("\n📋 Importing default templates from examples...")
    try:
        from landppt.database.create_default_template import ensure_default_templates_exist_first_time
        template_ids = await ensure_default_templates_exist_first_time()
        if template_ids:
            print(f"✅ Successfully imported {len(template_ids)} templates")
        else:
            print("⚠️  No templates were imported")
    except Exception as e:
        print(f"❌ Failed to import templates: {e}")

    # Final verification - ensure database is clean
    print("\n🔍 Final verification - ensuring clean database...")
    if not await verify_clean_database():
        print("⚠️  Database may contain residual data")
    else:
        print("✅ Database is clean and ready")

    print()
    print("🎉 Database setup completed successfully!")
    print("📋 Database initialized with clean, empty tables")
    print()
    print("📝 Next steps:")
    print("1. Start the server: python start_server.py")
    print("2. Access the web interface: http://localhost:8000/web")
    print("3. Check API docs: http://localhost:8000/docs")
    print()
    print("💾 Database file: landppt.db")
    print("🔧 Configuration: .env")

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LandPPT Database Setup")
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep existing data instead of creating clean database"
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear existing data without full reset"
    )

    args = parser.parse_args()

    try:
        if args.clear_only:
            # Only clear data from existing database
            print("🧹 Clearing existing database data...")
            success = asyncio.run(clear_all_data())
            if success:
                print("✅ Database data cleared successfully")
                success = asyncio.run(verify_clean_database())
            else:
                print("❌ Failed to clear database data")
        else:
            # Full setup with optional clean start
            clean_start = not args.keep_data
            success = asyncio.run(main(clean_start=clean_start))

        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
