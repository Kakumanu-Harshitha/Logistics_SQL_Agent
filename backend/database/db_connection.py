"""
database/db_connection.py
SQLAlchemy engine and session management.
"""
import os
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Ensure we use the asyncpg driver
raw_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/logistics_db")
if raw_url.startswith("postgresql://"):
    ASYNC_DATABASE_URL = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = raw_url

_async_engine = None
_sync_engine = None

def get_engine():
    """Return synchronous SQLAlchemy engine for ML/Ingestion."""
    global _sync_engine
    if _sync_engine is None:
        # Use sync URL (remove asyncpg if present, although raw_url usually doesn't have it)
        sync_url = raw_url.replace("+asyncpg", "")
        _sync_engine = create_engine(sync_url, pool_pre_ping=True)
    return _sync_engine

def get_async_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            ASYNC_DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _async_engine

def get_async_session() -> AsyncSession:
    engine = get_async_engine()
    AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
    return AsyncSessionLocal()

async def test_connection() -> bool:
    """Test async database connectivity."""
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
