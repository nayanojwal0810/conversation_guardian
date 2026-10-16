"""
MongoDB connection and lifecycle management for ConversationGuardian.

Provides configured MongoClient and Database instances, connection validation,
and explicit failure handling without silent mock fallbacks.
"""

import os
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError


DEFAULT_MONGODB_HOST = "127.0.0.1"
DEFAULT_MONGODB_PORT = 27018
DEFAULT_MONGODB_URI = f"mongodb://{DEFAULT_MONGODB_HOST}:{DEFAULT_MONGODB_PORT}"
DEFAULT_DATABASE_NAME = "conversation_guardian"
DEFAULT_TIMEOUT_MS = 3000


class DatabaseConnectionError(RuntimeError):
    """Raised when MongoDB connection cannot be established or verified."""
    pass


def get_mongodb_port() -> int:
    """Resolve published MongoDB host port from environment or default to 27018."""
    port_str = os.getenv("MONGODB_HOST_PORT", str(DEFAULT_MONGODB_PORT)).strip()
    try:
        return int(port_str)
    except ValueError:
        return DEFAULT_MONGODB_PORT


def get_mongodb_uri() -> str:
    """
    Resolve MongoDB connection string.
    
    Precedence:
    1. MONGODB_URI environment variable if set and non-empty.
    2. Default constructed URI using MONGODB_HOST_PORT (defaults to 27018).
    """
    explicit_uri = os.getenv("MONGODB_URI", "").strip()
    if explicit_uri:
        return explicit_uri
    port = get_mongodb_port()
    return f"mongodb://{DEFAULT_MONGODB_HOST}:{port}"


def get_database_name() -> str:
    """Resolve target database name from environment or use default."""
    return os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE_NAME).strip() or DEFAULT_DATABASE_NAME


def create_mongo_client(
    uri: Optional[str] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    max_pool_size: int = 50,
) -> MongoClient:
    """
    Construct a PyMongo client configured with explicit timeouts.
    
    Args:
        uri: MongoDB connection URI. Defaults to MONGODB_URI env var or localhost.
        timeout_ms: Server selection and connection timeout in milliseconds.
        max_pool_size: Maximum connection pool size.
        
    Returns:
        Configured MongoClient.
    """
    resolved_uri = uri or get_mongodb_uri()
    return MongoClient(
        resolved_uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        maxPoolSize=max_pool_size,
    )


def ping_connection(client: MongoClient, endpoint: Optional[str] = None) -> bool:
    """
    Verify live MongoDB connectivity using the admin ping command.
    
    Raises:
        DatabaseConnectionError: If MongoDB is unreachable or refuses connection.
    """
    try:
        client.admin.command("ping")
        return True
    except Exception as exc:
        target = endpoint or "configured endpoint"
        raise DatabaseConnectionError(
            f"Failed to connect to MongoDB at {target}: {str(exc)}. "
            f"Ensure MongoDB is running (e.g. via 'docker compose up -d') "
            f"and reachable at the configured MONGODB_URI."
        ) from exc


def get_database(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Database:
    """
    Connect to MongoDB, verify connectivity via ping, and return the database.
    
    Args:
        uri: Optional MongoDB URI override.
        db_name: Optional database name override.
        timeout_ms: Timeout in milliseconds.
        
    Returns:
        Verified pymongo Database instance.
        
    Raises:
        DatabaseConnectionError: When MongoDB cannot be contacted.
    """
    resolved_uri = uri or get_mongodb_uri()
    resolved_db_name = db_name or get_database_name()
    client = create_mongo_client(uri=resolved_uri, timeout_ms=timeout_ms)
    ping_connection(client, endpoint=resolved_uri)
    return client[resolved_db_name]
