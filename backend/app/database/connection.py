
from sqlalchemy import create_engine, text
from app.core.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=2,
    connect_args={"sslmode": "require"}
)


def check_database_connection():
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT 1")
        ).scalar_one()
