"""
Verify user email in database using SQLAlchemy.
Run this script to mark enterprise user as verified.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.user import User
from app.core.config import settings

# ============================================================
# METHOD 1: Using your app's database connection
# ============================================================

def verify_user_by_id(user_id: int):
    """Verify user by ID."""
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            print(f"❌ User with ID {user_id} not found")
            return False
        
        user.is_verified = True
        db.commit()
        
        print(f"✅ User {user.email} (ID: {user.id}) verified successfully!")
        print(f"   - Email: {user.email}")
        print(f"   - Subscription: {user.subscription}")
        print(f"   - Verified: {user.is_verified}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
        return False
    
    finally:
        db.close()


def verify_user_by_email(email: str):
    """Verify user by email address."""
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            print(f"❌ User with email {email} not found")
            return False
        
        user.is_verified = True
        db.commit()
        
        print(f"✅ User {user.email} verified successfully!")
        print(f"   - User ID: {user.id}")
        print(f"   - Subscription: {user.subscription}")
        print(f"   - Verified: {user.is_verified}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
        return False
    
    finally:
        db.close()


def verify_all_enterprise_users():
    """Verify all unverified enterprise users."""
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Find all unverified enterprise users
        unverified = db.query(User).filter(
            User.subscription == "enterprise",
            User.is_verified == False
        ).all()
        
        if not unverified:
            print("✅ All enterprise users already verified!")
            return True
        
        count = len(unverified)
        
        # Verify them
        for user in unverified:
            user.is_verified = True
            print(f"   ✓ Verifying {user.email}")
        
        db.commit()
        
        print(f"✅ {count} enterprise users verified successfully!")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
        return False
    
    finally:
        db.close()


def check_verification_status(user_id: int):
    """Check current verification status of a user."""
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            print(f"❌ User with ID {user_id} not found")
            return None
        
        print(f"\n📋 User Verification Status:")
        print(f"   - ID: {user.id}")
        print(f"   - Email: {user.email}")
        print(f"   - Username: {user.username}")
        print(f"   - Subscription: {user.subscription}")
        print(f"   - Is Active: {user.is_active}")
        print(f"   - Is Verified: {user.is_verified}")
        print(f"   - Created: {user.created_at}")
        print(f"   - Last Login: {user.last_login_at}\n")
        
        return user
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None
    
    finally:
        db.close()


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("USER EMAIL VERIFICATION SCRIPT")
    print("=" * 60)
    
    # Example 1: Check status before verification
    print("\n1️⃣ Checking user status BEFORE verification...")
    check_verification_status(user_id=10)
    
    # Example 2: Verify by ID
    print("2️⃣ Verifying user by ID...")
    verify_user_by_id(user_id=10)
    
    # Example 3: Check status after verification
    print("\n3️⃣ Checking user status AFTER verification...")
    check_verification_status(user_id=10)
    
    # Example 4: Verify by email
    print("\n4️⃣ Verifying user by email...")
    verify_user_by_email(email="enterprise@example.com")
    
    # Example 5: Verify all enterprise users
    print("\n5️⃣ Verifying all enterprise users...")
    verify_all_enterprise_users()
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)
