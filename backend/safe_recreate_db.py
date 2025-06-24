#!/usr/bin/env python
"""
Safely recreate the database with the correct schema
"""
import os
import time
import shutil
from sqlalchemy import create_engine, inspect
from database_base import Base

def safe_recreate_database():
    """Safely drop all tables and recreate them"""
    print("🔄 Starting safe database recreation...")
    
    # Create new database with different name first
    new_db_file = "modelship_new.db"
    backup_db_file = "modelship_backup.db"
    main_db_file = "modelship.db"
    
    # Remove new db file if it exists
    if os.path.exists(new_db_file):
        os.remove(new_db_file)
        print(f"🗑️  Removed existing {new_db_file}")
    
    try:
        # Create engine for new database
        engine = create_engine(f"sqlite:///./{new_db_file}", 
                             connect_args={"check_same_thread": False})
        
        # Import all models to register with Base
        print("📦 Importing models...")
        from models import (
            User, File, Job, Result, Project, Organization, LabelSchema, 
            ProjectAssignment, Review, Analytics
        )
        
        # Create all tables in new database
        print("🏗️  Creating tables...")
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"✅ Created tables: {tables}")
        
        # Close the engine
        engine.dispose()
        
        # Backup existing database if it exists
        if os.path.exists(main_db_file):
            if os.path.exists(backup_db_file):
                os.remove(backup_db_file)
            shutil.copy2(main_db_file, backup_db_file)
            print(f"💾 Backed up existing database to {backup_db_file}")
            
            # Wait a moment
            time.sleep(1)
            
            # Remove the old database
            try:
                os.remove(main_db_file)
                print(f"🗑️  Removed old database: {main_db_file}")
            except PermissionError:
                print(f"⚠️  Could not remove {main_db_file} - it may be in use")
                print("❌ Please close any applications using the database and try again")
                return False
        
        # Rename new database to main name
        shutil.move(new_db_file, main_db_file)
        print(f"✅ Successfully created new database: {main_db_file}")
        
        # Test the new database
        test_engine = create_engine(f"sqlite:///./{main_db_file}", 
                                  connect_args={"check_same_thread": False})
        test_inspector = inspect(test_engine)
        final_tables = test_inspector.get_table_names()
        test_engine.dispose()
        
        print(f"🧪 Final verification - tables in database: {final_tables}")
        print("🎉 Database recreation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during database recreation: {e}")
        # Cleanup on error
        if os.path.exists(new_db_file):
            os.remove(new_db_file)
        return False

if __name__ == "__main__":
    success = safe_recreate_database()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Close any database browsers or applications")
        print("2. Stop any running FastAPI servers")
        print("3. Try running the script again")
        exit(1)
    else:
        print("\n✅ Database is ready! You can now start the server.") 