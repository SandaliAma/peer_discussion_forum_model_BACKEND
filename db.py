"""
MongoDB connection module for MathRAG
Saves question/answer responses to MongoDB Atlas
"""

from pymongo import MongoClient
from datetime import datetime
import logging

import config

logger = logging.getLogger(__name__)

# Global client (reused across requests)
_client = None
_db = None


def get_database():
    """Get MongoDB database connection"""
    global _client, _db

    if _db is not None:
        return _db

    if not config.MONGO_URL:
        logger.warning("MONGO_URL not set, database disabled")
        return None

    try:
        _client = MongoClient(
            config.MONGO_URL,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            retryWrites=True,
        )
        # Test connection
        _client.admin.command('ping')
        _db = _client[config.MONGO_DB_NAME]
        logger.info(f"MongoDB connected: {config.MONGO_DB_NAME}")
        return _db
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return None


def save_response(question, answer, student_id, source, model_used,
                  response_time_ms, validation=None, similar_problems_count=0):
    """Save a question/answer response to MongoDB"""

    db = get_database()
    if db is None:
        return None

    document = {
        "student_id": student_id,
        "question": question,
        "answer": answer,
        "source": source,
        "model_used": model_used,
        "response_time_ms": response_time_ms,
        "validation": validation or {},
        "similar_problems_count": similar_problems_count,
        "created_at": datetime.utcnow()
    }

    try:
        result = db[config.MONGO_COLLECTION].insert_one(document)
        logger.info(f"Saved to MongoDB: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        logger.error(f"Failed to save to MongoDB: {e}")
        return None
