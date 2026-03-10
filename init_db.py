#!/usr/bin/env python
"""
Quick Setup Script for Sign
Initializes database and creates admin user
"""

import os
import sys
from app import app, db
from models import User, SignDataset

def init_database():
    """Initialize database with tables"""
    with app.app_context():
        print("🗄️  Creating database tables...")
        db.create_all()
        print("✅ Database tables created")

def create_admin_user():
    """Create default admin user"""
    with app.app_context():
        if User.query.filter_by(username='admin').first():
            print("⚠️  Admin user already exists, skipping...")
            return
        
        admin = User(
            username='admin',
            email='admin@sign.com',
            full_name='Administrator',
            role='admin'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user created")
        print("   Username: admin")
        print("   Password: admin123")

def seed_sign_data():
    """Seed sample sign language data"""
    with app.app_context():
        sample_signs = [
            ('A', 'Letter A in ASL', 'letter'),
            ('B', 'Letter B in ASL', 'letter'),
            ('C', 'Letter C in ASL', 'letter'),
            ('HELLO', 'Greeting sign', 'phrase'),
            ('THANK YOU', 'Expression of gratitude', 'phrase'),
            ('YES', 'Affirmative response', 'phrase'),
            ('NO', 'Negative response', 'phrase'),
            ('PLEASE', 'Polite request', 'phrase'),
            ('I LOVE YOU', 'Expression of affection', 'phrase'),
            ('GOOD MORNING', 'Time-specific greeting', 'phrase'),
        ]
        
        count = 0
        for sign_name, description, gesture_type in sample_signs:
            if not SignDataset.query.filter_by(sign_name=sign_name).first():
                sign = SignDataset(
                    sign_name=sign_name,
                    description=description,
                    gesture_type=gesture_type
                )
                db.session.add(sign)
                count += 1
        
        if count > 0:
            db.session.commit()
            print(f"✅ Seeded {count} sign entries")
        else:
            print("⚠️  Sign data already seeded, skipping...")

def main():
    """Run initialization sequence"""
    print("=" * 50)
    print("Sign - Initialization Script")
    print("=" * 50)
    print()
    
    try:
        init_database()
        create_admin_user()
        seed_sign_data()
        
        print()
        print("=" * 50)
        print("✅ Initialization Complete!")
        print("=" * 50)
        print()
        print("🚀 You can now start the application:")
        print("   python app.py")
        print()
        print("📝 Access at: http://localhost:5000")
        print()
        print("⚠️  Important: Change the default password in production!")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
