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
                  response_time_ms, chat_id=None, validation=None, similar_problems_count=0):
    """Save a question/answer response to MongoDB"""

    db = get_database()
    if db is None:
        return None

    document = {
        "student_id": student_id,
        "chat_id": chat_id or "default",
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


def get_history(student_id, limit=50):
    """Get chat history for a student, grouped by chat_id"""

    db = get_database()
    if db is None:
        return []

    try:
        pipeline = [
            {"$match": {"student_id": student_id}},
            {"$sort": {"created_at": 1}},
            {"$group": {
                "_id": {"$ifNull": ["$chat_id", "default"]},
                "messages": {"$push": {
                    "question": "$question",
                    "answer": "$answer",
                    "created_at": "$created_at"
                }},
                "first_question": {"$first": "$question"},
                "last_active": {"$max": "$created_at"}
            }},
            {"$sort": {"last_active": -1}},
            {"$limit": limit}
        ]

        result = list(db[config.MONGO_COLLECTION].aggregate(pipeline))

        chats = []
        for chat in result:
            chats.append({
                "chat_id": chat["_id"],
                "title": chat["first_question"][:30] + "...",
                "last_active": chat["last_active"].isoformat() if chat["last_active"] else "",
                "messages": [
                    {
                        "question": m["question"],
                        "answer": m["answer"],
                        "created_at": m["created_at"].isoformat() if m.get("created_at") else ""
                    }
                    for m in chat["messages"]
                ]
            })

        # Merge custom titles from chat_titles collection
        titles = {}
        try:
            title_docs = db["chat_titles"].find({"student_id": student_id})
            for doc in title_docs:
                titles[doc["chat_id"]] = doc["title"]
        except Exception:
            pass

        for chat in chats:
            if chat["chat_id"] in titles:
                chat["title"] = titles[chat["chat_id"]]

        return chats
    except Exception as e:
        logger.error(f"Failed to get history from MongoDB: {e}")
        return []


def delete_chat(chat_id, student_id):
    """Delete all messages for a chat session"""

    db = get_database()
    if db is None:
        return False

    try:
        result = db[config.MONGO_COLLECTION].delete_many({
            "chat_id": chat_id,
            "student_id": student_id
        })
        # Also remove custom title
        db["chat_titles"].delete_one({"chat_id": chat_id, "student_id": student_id})
        logger.info(f"Deleted {result.deleted_count} messages for chat {chat_id}")
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Failed to delete chat: {e}")
        return False


def rename_chat(chat_id, student_id, title):
    """Rename a chat session"""

    db = get_database()
    if db is None:
        return False

    try:
        db["chat_titles"].update_one(
            {"chat_id": chat_id, "student_id": student_id},
            {"$set": {"title": title, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        logger.info(f"Renamed chat {chat_id} to: {title}")
        return True
    except Exception as e:
        logger.error(f"Failed to rename chat: {e}")
        return False
