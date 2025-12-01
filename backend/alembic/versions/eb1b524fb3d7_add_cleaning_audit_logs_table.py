"""initial full schema

Revision ID: 0001_initial
Revises: 
Create Date: 2025-12-01
"""

from alembic import op
import sqlalchemy as sa


# ✅ Required by Alembic
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # =========================
    # ✅ USERS TABLE
    # =========================
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("username", sa.String(50), nullable=False, unique=True, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("profile_picture_url", sa.String(500)),
        sa.Column("phone_number", sa.String(20)),
        sa.Column("bio", sa.Text),
        sa.Column("role", sa.String(20), nullable=False, server_default="user"),
        sa.Column("subscription", sa.String(20), nullable=False, server_default="free"),
        sa.Column("subscription_expires_at", sa.DateTime(timezone=True)),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("email_verified_at", sa.DateTime(timezone=True)),
        sa.Column("last_login_at", sa.DateTime(timezone=True)),
        sa.Column("login_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed_login_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime(timezone=True)),
        sa.Column("password_changed_at", sa.DateTime(timezone=True)),
        sa.Column("api_key_hash", sa.String(255), unique=True),
        sa.Column("api_key_created_at", sa.DateTime(timezone=True)),
        sa.Column("timezone", sa.String(50), nullable=False, server_default="UTC"),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
        sa.Column("storage_used_bytes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("datasets_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # =========================
    # ✅ DATASETS TABLE
    # =========================
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_path", sa.String(500), nullable=False, unique=True),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=False),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("file_hash", sa.String(64)),
        sa.Column("status", sa.String(20), nullable=False, server_default="uploading"),
        sa.Column("processing_started_at", sa.DateTime(timezone=True)),
        sa.Column("processing_completed_at", sa.DateTime(timezone=True)),
        sa.Column("processing_error", sa.Text),
        sa.Column("processing_duration_seconds", sa.Float),
        sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("row_count", sa.Integer),
        sa.Column("column_count", sa.Integer),
        sa.Column("columns_info", sa.JSON),
        sa.Column("missing_values_count", sa.Integer),
        sa.Column("duplicate_rows_count", sa.Integer),
        sa.Column("data_quality_score", sa.Float),
        sa.Column("eda_report_url", sa.String(500)),
        sa.Column("eda_report_generated_at", sa.DateTime(timezone=True)),
        sa.Column("tags", sa.JSON),
        sa.Column("metadata_json", sa.JSON),
        sa.Column("is_public", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("share_token", sa.String(64), unique=True),
        sa.Column("view_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("download_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("orders_path", sa.String(500)),
        sa.Column("customers_path", sa.String(500)),
        sa.Column("products_path", sa.String(500)),
        sa.Column("marketing_path", sa.String(500)),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True)),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # =========================
    # ✅ DATASET INSIGHTS
    # =========================
    op.create_table(
        "dataset_insights",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE")),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("insight_type", sa.String(50), nullable=False),
        sa.Column("confidence_score", sa.Float),
        sa.Column("model_used", sa.String(100)),
        sa.Column("generation_time_seconds", sa.Float),
        sa.Column("is_helpful", sa.Boolean),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # =========================
    # ✅ DATASET VISUALIZATIONS
    # =========================
    op.create_table(
        "dataset_visualizations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE")),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("chart_type", sa.String(50), nullable=False),
        sa.Column("chart_url", sa.String(500)),
        sa.Column("config", sa.JSON),
        sa.Column("data_columns", sa.JSON),
        sa.Column("order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # =========================
    # ✅ CLEANING AUDIT LOGS
    # =========================
    op.create_table(
        "cleaning_audit_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE")),
        sa.Column("session_id", sa.String(64), unique=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_seconds", sa.Float),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("config_used", sa.JSON),
        sa.Column("original_shape", sa.JSON),
        sa.Column("final_shape", sa.JSON),
        sa.Column("rows_removed", sa.Integer),
        sa.Column("columns_removed", sa.Integer),
        sa.Column("transformations_applied", sa.JSON),
        sa.Column("cleaning_steps", sa.JSON),
        sa.Column("quality_metrics", sa.JSON),
        sa.Column("quality_alerts", sa.JSON),
        sa.Column("original_file_path", sa.String(500)),
        sa.Column("cleaned_file_path", sa.String(500)),
        sa.Column("error_message", sa.Text),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("cleaning_audit_logs")
    op.drop_table("dataset_visualizations")
    op.drop_table("dataset_insights")
    op.drop_table("datasets")
    op.drop_table("users")
