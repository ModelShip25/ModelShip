#!/usr/bin/env python
"""
Database Access Guide - Multiple ways to access your ModelShip database
"""
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User, Project, Organization, LabelSchema, Job, Result
import json

class DatabaseManager:
    """Helper class for database operations"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def list_all_users(self):
        """List all users in the database"""
        print("👥 USERS:")
        users = self.db.query(User).all()
        for user in users:
            print(f"  ID: {user.id} | Email: {user.email} | Role: {user.role} | Credits: {user.credits_remaining}")
        return users
    
    def list_all_projects(self):
        """List all projects in the database"""
        print("\n📁 PROJECTS:")
        projects = self.db.query(Project).all()
        for project in projects:
            print(f"  ID: {project.id} | Name: {project.name} | Type: {project.project_type} | Status: {project.status}")
            print(f"      Progress: {project.labeled_items}/{project.total_items} labeled")
        return projects
    
    def list_all_organizations(self):
        """List all organizations in the database"""
        print("\n🏢 ORGANIZATIONS:")
        orgs = self.db.query(Organization).all()
        for org in orgs:
            print(f"  ID: {org.id} | Name: {org.name} | Plan: {org.plan_type} | Credits: {org.credits_pool}")
        return orgs
    
    def list_all_jobs(self):
        """List all jobs in the database"""
        print("\n⚙️ JOBS:")
        jobs = self.db.query(Job).all()
        for job in jobs:
            print(f"  ID: {job.id} | Type: {job.job_type} | Status: {job.status} | Progress: {job.completed_items}/{job.total_items}")
        return jobs
    
    def get_database_stats(self):
        """Get database statistics"""
        print("📊 DATABASE STATISTICS:")
        print(f"  Users: {self.db.query(User).count()}")
        print(f"  Organizations: {self.db.query(Organization).count()}")
        print(f"  Projects: {self.db.query(Project).count()}")
        print(f"  Jobs: {self.db.query(Job).count()}")
        print(f"  Results: {self.db.query(Result).count()}")
        print(f"  Label Schemas: {self.db.query(LabelSchema).count()}")
    
    def search_user_by_email(self, email):
        """Search for a user by email"""
        user = self.db.query(User).filter(User.email == email).first()
        if user:
            print(f"\n🔍 Found user: {user.email}")
            print(f"   ID: {user.id}")
            print(f"   Role: {user.role}")
            print(f"   Credits: {user.credits_remaining}")
            print(f"   Active: {user.is_active}")
            return user
        else:
            print(f"❌ User with email '{email}' not found")
            return None

def database_browser_cli():
    """Command-line database browser"""
    print("🗄️  ModelShip Database Browser")
    print("=" * 50)
    
    with DatabaseManager() as db_manager:
        while True:
            print("\nChoose an option:")
            print("1. View all data")
            print("2. Database statistics")
            print("3. List users")
            print("4. List projects")
            print("5. List organizations")
            print("6. List jobs")
            print("7. Search user by email")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                print("\n" + "="*60)
                db_manager.get_database_stats()
                db_manager.list_all_organizations()
                db_manager.list_all_users()
                db_manager.list_all_projects()
                db_manager.list_all_jobs()
                print("="*60)
            
            elif choice == "2":
                print("\n" + "="*30)
                db_manager.get_database_stats()
                print("="*30)
            
            elif choice == "3":
                print("\n" + "="*30)
                db_manager.list_all_users()
                print("="*30)
            
            elif choice == "4":
                print("\n" + "="*30)
                db_manager.list_all_projects()
                print("="*30)
            
            elif choice == "5":
                print("\n" + "="*30)
                db_manager.list_all_organizations()
                print("="*30)
            
            elif choice == "6":
                print("\n" + "="*30)
                db_manager.list_all_jobs()
                print("="*30)
            
            elif choice == "7":
                email = input("Enter email to search: ").strip()
                db_manager.search_user_by_email(email)
            
            elif choice == "8":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")

def print_access_guide():
    """Print comprehensive database access guide"""
    print("""
🗄️  MODELSHIP DATABASE ACCESS GUIDE
=====================================

📍 DATABASE LOCATION:
    File: C:\\Users\\shine\\Desktop\\ModelShip\\backend\\modelship.db
    Type: SQLite Database
    Size: ~220KB

🛠️  ACCESS METHODS:

1. 📱 BUILT-IN CLI BROWSER (This Script)
   ----------------------------------------
   Run: python database_access_guide.py
   
   Features:
   - View all data
   - Search users
   - Browse tables
   - Database statistics

2. 🖥️  DB BROWSER FOR SQLITE (Recommended GUI)
   -------------------------------------------
   Download: https://sqlitebrowser.org/
   
   Steps:
   1. Download and install DB Browser for SQLite
   2. Open the application
   3. Click "Open Database"
   4. Navigate to: C:\\Users\\shine\\Desktop\\ModelShip\\backend\\modelship.db
   5. Browse tables, run queries, view data
   
   Features:
   - Visual table browsing
   - SQL query interface
   - Data editing
   - Export capabilities

3. 💻 COMMAND LINE SQLITE
   ----------------------
   Run: sqlite3 modelship.db
   
   Common commands:
   .tables                    -- List all tables
   .schema users             -- Show table structure
   SELECT * FROM users;      -- View all users
   SELECT * FROM projects;   -- View all projects
   .quit                     -- Exit

4. 🐍 PYTHON SCRIPTS
   -----------------
   Create custom scripts using the DatabaseManager class above
   
   Example:
   with DatabaseManager() as db:
       users = db.list_all_users()

5. 🌐 WEB INTERFACE (Via FastAPI)
   -------------------------------
   When your server is running:
   - API Docs: http://localhost:8000/docs
   - Database endpoints: /api/users, /api/projects, etc.

📋 DATABASE TABLES:
   - users: User accounts and authentication
   - organizations: Company/team structures  
   - projects: ML labeling projects
   - jobs: Annotation jobs and batches
   - results: ML predictions and human labels
   - files: Uploaded images and documents
   - label_schemas: Classification categories
   - reviews: Human review and corrections
   - analytics: Usage and performance metrics

💡 TIPS:
   - Always backup before making changes
   - Use transactions for bulk operations
   - The API provides the safest access method
   - SQLite Browser is great for exploring data
   - Use this CLI tool for quick lookups

🔒 SECURITY:
   - Database contains user credentials (hashed)
   - Keep database file secure
   - Don't share credentials in plain text
   - Use proper authentication in production
""")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        print_access_guide()
    else:
        print_access_guide()
        print("\n" + "="*50)
        input("Press Enter to open the interactive database browser...")
        database_browser_cli() 