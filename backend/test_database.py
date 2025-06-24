#!/usr/bin/env python
"""
Test the database setup and add sample data
"""
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import User, Project, Organization, LabelSchema
from datetime import datetime
import json

def test_database_connection():
    """Test that we can connect to the database"""
    print("🔌 Testing database connection...")
    
    try:
        db = SessionLocal()
        
        # Test basic connection
        result = db.execute("SELECT 1").fetchone()
        print(f"✅ Database connection successful: {result}")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def create_sample_data():
    """Create some sample data for testing"""
    print("📝 Creating sample data...")
    
    db = SessionLocal()
    try:
        # Create organization
        org = Organization(
            name="ModelShip Demo Corp",
            description="Demo organization for testing",
            plan_type="pro",
            credits_pool=5000
        )
        db.add(org)
        db.commit()
        db.refresh(org)
        print(f"✅ Created organization: {org.name} (ID: {org.id})")
        
        # Create admin user
        admin_user = User(
            email="admin@modelship.com",
            password_hash="$2b$12$dummy_hash_for_testing",  # In production, use proper hashing
            subscription_tier="pro",
            credits_remaining=1000,
            role="admin",
            organization_id=org.id,
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        print(f"✅ Created admin user: {admin_user.email} (ID: {admin_user.id})")
        
        # Create demo user
        demo_user = User(
            email="demo@modelship.com",
            password_hash="$2b$12$dummy_hash_for_testing",
            subscription_tier="free",
            credits_remaining=100,
            role="labeler",
            organization_id=org.id,
            is_active=True
        )
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        print(f"✅ Created demo user: {demo_user.email} (ID: {demo_user.id})")
        
        # Create sample project
        project = Project(
            name="Image Classification Demo",
            description="Demo project for image classification tasks",
            project_type="image_classification",
            status="active",
            confidence_threshold=0.8,
            auto_approve_threshold=0.95,
            guidelines="Please classify images accurately according to the provided categories.",
            owner_id=admin_user.id,
            organization_id=org.id,
            total_items=0,
            labeled_items=0,
            reviewed_items=0,
            approved_items=0
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        print(f"✅ Created project: {project.name} (ID: {project.id})")
        
        # Create label schema
        schema = LabelSchema(
            project_id=project.id,
            name="Animal Classification",
            label_type="classification",
            categories=json.dumps([
                {"id": 1, "name": "cat", "description": "Domestic cats", "color": "#FF6B6B"},
                {"id": 2, "name": "dog", "description": "Domestic dogs", "color": "#4ECDC4"},
                {"id": 3, "name": "bird", "description": "Various bird species", "color": "#45B7D1"},
                {"id": 4, "name": "other", "description": "Other animals", "color": "#96CEB4"}
            ]),
            is_multi_label=False,
            is_hierarchical=False
        )
        db.add(schema)
        db.commit()
        db.refresh(schema)
        print(f"✅ Created label schema: {schema.name} (ID: {schema.id})")
        
        print(f"\n🎉 Sample data created successfully!")
        print(f"📊 Summary:")
        print(f"   - Organization: {org.name}")
        print(f"   - Users: {admin_user.email}, {demo_user.email}")
        print(f"   - Project: {project.name}")
        print(f"   - Label Schema: {schema.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def verify_data():
    """Verify the data was created correctly"""
    print("\n🔍 Verifying data...")
    
    db = SessionLocal()
    try:
        # Count records
        user_count = db.query(User).count()
        org_count = db.query(Organization).count()
        project_count = db.query(Project).count()
        schema_count = db.query(LabelSchema).count()
        
        print(f"📈 Database contents:")
        print(f"   - Users: {user_count}")
        print(f"   - Organizations: {org_count}")
        print(f"   - Projects: {project_count}")
        print(f"   - Label Schemas: {schema_count}")
        
        # Get first user to test relationships
        user = db.query(User).first()
        if user:
            print(f"👤 First user: {user.email} (Role: {user.role})")
            if user.organization:
                print(f"🏢 Organization: {user.organization.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        return False
    finally:
        db.close()

def main():
    """Main test function"""
    print("🚀 Starting database tests...\n")
    
    # Test 1: Connection
    if not test_database_connection():
        print("❌ Database tests failed - connection issue")
        return False
    
    # Test 2: Create sample data
    if not create_sample_data():
        print("❌ Database tests failed - data creation issue")
        return False
    
    # Test 3: Verify data
    if not verify_data():
        print("❌ Database tests failed - data verification issue")
        return False
    
    print("\n✅ All database tests passed!")
    print("🎯 Your database is ready for the ModelShip application!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 