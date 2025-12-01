from app.core.security import get_password_hash
from app.database import SessionLocal
from app.models.user import User, UserRole, UserSubscription

db = SessionLocal()

password = "admin123"
hashed = get_password_hash(password)

user = User(
    email="admin@test.com",
    username="enterprise_admin",
    hashed_password=hashed,
    full_name="Enterprise Admin",
    role=UserRole.ADMIN,
    subscription=UserSubscription.ENTERPRISE,
    is_active=True,
    is_verified=True,
)

db.add(user)
db.commit()
db.refresh(user)

print("✅ Admin created successfully!")
print("Username: enterprise_admin")
print("Password:", password)
print("User ID:", user.id)

db.close()
