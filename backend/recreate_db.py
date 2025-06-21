#!/usr/bin/env python
"""
Recreate the database with the correct schema
"""
import os
from database import engine
from database_base import Base
from models import User, File, Job, Result

def recreate_database():
    """Drop all tables and recreate them"""
    print("Recreating database...")
    
    # Delete database file if it exists
    db_file = "modelship.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Deleted existing database: {db_file}")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Created all tables with updated schema")
    
    # Verify tables were created
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Created tables: {tables}")
    
    # Check columns for users table
    if 'users' in tables:
        columns = inspector.get_columns('users')
        column_names = [col['name'] for col in columns]
        print(f"Users table columns: {column_names}")
    
    print("✅ Database recreation complete!")

if __name__ == "__main__":
    recreate_database() 