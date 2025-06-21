#!/usr/bin/env python
"""
Fix database schema by dropping and recreating tables
"""
import os
import sqlite3

def fix_database():
    """Fix the database schema"""
    db_file = "backend/modelship.db"
    
    print("🔧 Fixing database schema...")
    
    # Connect to database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Drop all existing tables
    print("Dropping existing tables...")
    cursor.execute("DROP TABLE IF EXISTS results")
    cursor.execute("DROP TABLE IF EXISTS jobs") 
    cursor.execute("DROP TABLE IF EXISTS files")
    cursor.execute("DROP TABLE IF EXISTS users")
    
    # Create users table with correct schema
    print("Creating users table...")
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            subscription_tier VARCHAR(50) DEFAULT 'free',
            credits_remaining INTEGER DEFAULT 100,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create files table
    print("Creating files table...")
    cursor.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            filename VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size INTEGER,
            file_type VARCHAR(50),
            status VARCHAR(50) DEFAULT 'uploaded',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create jobs table
    print("Creating jobs table...")
    cursor.execute("""
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            job_type VARCHAR(50) NOT NULL,
            status VARCHAR(50) DEFAULT 'processing',
            total_items INTEGER DEFAULT 0,
            completed_items INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create results table
    print("Creating results table...")
    cursor.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY,
            job_id INTEGER NOT NULL,
            file_id INTEGER,
            filename VARCHAR(255),
            predicted_label VARCHAR(255),
            confidence REAL,
            processing_time REAL,
            reviewed BOOLEAN DEFAULT 0,
            ground_truth VARCHAR(255),
            status VARCHAR(50) DEFAULT 'success',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id),
            FOREIGN KEY (file_id) REFERENCES files (id)
        )
    """)
    
    # Commit changes and close
    conn.commit()
    conn.close()
    
    print("✅ Database schema fixed successfully!")
    print("📋 Tables created: users, files, jobs, results")

if __name__ == "__main__":
    fix_database() 