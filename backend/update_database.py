"""
Database Update Script for ModelShip MVP
Migrates existing database to support complete feature set
"""

import sqlite3
import os
from datetime import datetime
import logging
from sqlalchemy import create_engine
from database import Base, DATABASE_URL
from models import *  # Import all existing models

# Import Phase 2 models
from data_versioning import DatasetVersion, AnnotationSnapshot
from gold_standard_testing import GoldStandardSample, GoldStandardTest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_database():
    """Update database schema to support all MVP features"""
    
    db_path = "modelship.db"
    
    if not os.path.exists(db_path):
        logger.error("Database file not found. Please run the main app first to create initial database.")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("Starting database migration...")
        
        # 1. Add new columns to existing users table
        logger.info("Updating users table...")
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'labeler'")
            cursor.execute("ALTER TABLE users ADD COLUMN organization_id INTEGER")
            cursor.execute("ALTER TABLE users ADD COLUMN last_login_at TIMESTAMP")
            cursor.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
            logger.info("Users table updated successfully")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("Users table already has new columns")
            else:
                logger.error(f"Error updating users table: {e}")
        
        # 2. Create organizations table
        logger.info("Creating organizations table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                plan_type VARCHAR(50) DEFAULT 'team',
                credits_pool INTEGER DEFAULT 1000,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. Create projects table
        logger.info("Creating projects table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                project_type VARCHAR(50) NOT NULL,
                status TEXT DEFAULT 'draft',
                confidence_threshold REAL DEFAULT 0.8,
                auto_approve_threshold REAL DEFAULT 0.95,
                guidelines TEXT,
                owner_id INTEGER NOT NULL,
                organization_id INTEGER,
                total_items INTEGER DEFAULT 0,
                labeled_items INTEGER DEFAULT 0,
                reviewed_items INTEGER DEFAULT 0,
                approved_items INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deadline TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users (id),
                FOREIGN KEY (organization_id) REFERENCES organizations (id)
            )
        """)
        
        # 4. Create label_schemas table
        logger.info("Creating label_schemas table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS label_schemas (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                name VARCHAR(255) NOT NULL,
                label_type TEXT NOT NULL,
                categories TEXT NOT NULL,
                hierarchy TEXT,
                attributes TEXT,
                is_multi_label BOOLEAN DEFAULT 0,
                is_hierarchical BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        # 5. Create project_assignments table
        logger.info("Creating project_assignments table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_assignments (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                assigned_items INTEGER DEFAULT 0,
                completed_items INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 6. Create analytics table
        logger.info("Creating analytics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                user_id INTEGER,
                metric_type VARCHAR(100) NOT NULL,
                metric_value REAL NOT NULL,
                metric_data TEXT,
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 7. Update existing tables with new columns
        logger.info("Updating files table...")
        try:
            cursor.execute("ALTER TABLE files ADD COLUMN project_id INTEGER")
            cursor.execute("ALTER TABLE files ADD COLUMN file_metadata TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("Files table already updated")
            else:
                logger.error(f"Error updating files table: {e}")
        
        logger.info("Updating jobs table...")
        try:
            cursor.execute("ALTER TABLE jobs ADD COLUMN project_id INTEGER")
            cursor.execute("ALTER TABLE jobs ADD COLUMN model_name VARCHAR(100)")
            cursor.execute("ALTER TABLE jobs ADD COLUMN confidence_threshold REAL DEFAULT 0.8")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("Jobs table already updated")
            else:
                logger.error(f"Error updating jobs table: {e}")
        
        logger.info("Updating results table...")
        try:
            cursor.execute("ALTER TABLE results ADD COLUMN all_predictions TEXT")
            cursor.execute("ALTER TABLE results ADD COLUMN model_version VARCHAR(100)")
            cursor.execute("ALTER TABLE results ADD COLUMN reviewed_by INTEGER")
            cursor.execute("ALTER TABLE results ADD COLUMN reviewed_at TIMESTAMP")
            cursor.execute("ALTER TABLE results ADD COLUMN review_action VARCHAR(50)")
            cursor.execute("ALTER TABLE results ADD COLUMN correction_reason TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("Results table already updated")
            else:
                logger.error(f"Error updating results table: {e}")
        
        # 8. Create indexes for better performance
        logger.info("Creating performance indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_organization ON users(organization_id)",
            "CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id)",
            "CREATE INDEX IF NOT EXISTS idx_projects_organization ON projects(organization_id)",
            "CREATE INDEX IF NOT EXISTS idx_project_assignments_project ON project_assignments(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_project_assignments_user ON project_assignments(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_results_job ON results(job_id)",
            "CREATE INDEX IF NOT EXISTS idx_results_reviewed_by ON results(reviewed_by)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_project ON analytics(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics(user_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        # 9. Insert default organization for existing users
        logger.info("Creating default organization...")
        cursor.execute("""
            INSERT OR IGNORE INTO organizations (id, name, description, plan_type)
            VALUES (1, 'Default Organization', 'Default organization for existing users', 'team')
        """)
        
        # Update existing users to be part of default organization
        cursor.execute("""
            UPDATE users SET organization_id = 1 WHERE organization_id IS NULL
        """)
        
        # 10. Create migration record
        logger.info("Recording migration...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY,
                version VARCHAR(50) NOT NULL,
                description TEXT,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT INTO migrations (version, description)
            VALUES ('2024.01.mvp', 'Complete MVP database schema with projects, organizations, analytics, and enhanced features')
        """)
        
        conn.commit()
        logger.info("Database migration completed successfully!")
        
        # Print summary
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM projects")
        project_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM organizations")
        org_count = cursor.fetchone()[0]
        
        logger.info(f"Migration Summary:")
        logger.info(f"- Users: {user_count}")
        logger.info(f"- Organizations: {org_count}")
        logger.info(f"- Projects: {project_count}")
        logger.info(f"- Database version: 2024.01.mvp")
        
        return True
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def verify_migration():
    """Verify that migration was successful"""
    
    try:
        conn = sqlite3.connect("modelship.db")
        cursor = conn.cursor()
        
        # Check that all tables exist
        required_tables = [
            'users', 'organizations', 'projects', 'label_schemas', 
            'project_assignments', 'analytics', 'files', 'jobs', 'results'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            logger.error(f"Missing tables after migration: {missing_tables}")
            return False
        
        # Check critical columns exist
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [col[1] for col in cursor.fetchall()]
        
        required_user_columns = ['role', 'organization_id', 'last_login_at', 'is_active']
        missing_user_columns = set(required_user_columns) - set(user_columns)
        
        if missing_user_columns:
            logger.error(f"Missing user columns: {missing_user_columns}")
            return False
        
        logger.info("Migration verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

def create_all_tables():
    """Create all database tables including Phase 2 additions"""
    engine = create_engine(DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("✅ All Phase 1 + Phase 2 database tables created successfully!")
    print("\nPhase 1 Tables:")
    print("- users")
    print("- organizations") 
    print("- projects")
    print("- jobs")
    print("- results")
    print("- reviews")
    print("- user_organizations")
    
    print("\nPhase 2 Tables:")
    print("- dataset_versions")
    print("- annotation_snapshots")
    print("- gold_standard_samples")
    print("- gold_standard_tests")

if __name__ == "__main__":
    print("ModelShip Database Migration Tool")
    print("="*50)
    
    # Backup existing database
    if os.path.exists("modelship.db"):
        backup_name = f"modelship_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        os.system(f"cp modelship.db {backup_name}")
        logger.info(f"Database backed up to {backup_name}")
    
    # Run migration
    if update_database():
        if verify_migration():
            print("\n✅ Database migration completed successfully!")
            print("All MVP features are now available.")
            print("\nNew features enabled:")
            print("- Project management and organization")
            print("- Team collaboration with roles")
            print("- Label schema management")
            print("- Advanced export formats (COCO, YOLO, Pascal VOC)")
            print("- Active learning and intelligent sampling")
            print("- Comprehensive analytics dashboard")
            print("- Enhanced user and project management")
        else:
            print("\n❌ Migration verification failed!")
    else:
        print("\n❌ Database migration failed!")

    # Create all tables
    create_all_tables() 