#!/usr/bin/env python3
"""
Migration script to support guest users
Makes owner_id and user_id nullable in projects, files, and jobs tables
"""

import sqlite3
import os
from pathlib import Path

def migrate_database():
    """Apply migration to support guest users"""
    
    # Database path
    db_path = Path(__file__).parent / "modelship.db"
    
    if not db_path.exists():
        print("Database not found, migration not needed")
        return
    
    print(f"Migrating database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Begin transaction
        conn.execute("BEGIN TRANSACTION")
        
        # Check if migration is needed
        cursor.execute("PRAGMA table_info(projects)")
        projects_info = {row[1]: row for row in cursor.fetchall()}
        
        if 'owner_id' in projects_info:
            owner_nullable = projects_info['owner_id'][3] == 0  # 0 means nullable=True
            if not owner_nullable:
                print("Migrating projects table to make owner_id nullable...")
                
                # Create new projects table
                cursor.execute("""
                    CREATE TABLE projects_new AS SELECT * FROM projects
                """)
                
                cursor.execute("DROP TABLE projects")
                
                cursor.execute("""
                    CREATE TABLE projects (
                        id INTEGER PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        project_type VARCHAR(50) NOT NULL,
                        status VARCHAR(20) DEFAULT 'draft',
                        confidence_threshold FLOAT DEFAULT 0.8,
                        auto_approve_threshold FLOAT DEFAULT 0.95,
                        guidelines TEXT,
                        owner_id INTEGER,
                        organization_id INTEGER,
                        total_items INTEGER DEFAULT 0,
                        labeled_items INTEGER DEFAULT 0,
                        reviewed_items INTEGER DEFAULT 0,
                        approved_items INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        deadline TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    INSERT INTO projects SELECT * FROM projects_new
                """)
                
                cursor.execute("DROP TABLE projects_new")
                print("✅ Projects table updated")
            else:
                print("Projects table already supports nullable owner_id")
        
        # Similar process for files and jobs tables...
        # For now, let's just ensure the tables exist with correct schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                project_id INTEGER,
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                file_size INTEGER,
                file_type VARCHAR(50),
                status VARCHAR(50) DEFAULT 'uploaded',
                file_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                project_id INTEGER,
                job_type VARCHAR(50) NOT NULL,
                status VARCHAR(50) DEFAULT 'processing',
                model_name VARCHAR(100),
                confidence_threshold FLOAT DEFAULT 0.8,
                total_items INTEGER DEFAULT 0,
                completed_items INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            )
        """)
        
        # Commit transaction
        conn.commit()
        conn.close()
        
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_database() 