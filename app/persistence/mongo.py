"""
Async MongoDB client + collection helpers via ``motor``.

Connection string defaults to ``mongodb://localhost:27017/`` (override via
``MONGO_URL`` env). Database name defaults to ``biolistener_sleep_dev`` (override
via ``MONGO_DB`` env). Use a separate DB for each researcher / each run
to keep multi-user experiments isolated.

The schema documents (`app.persistence.schema`) are converted to dicts
via ``.model_dump()`` on write. Reads return raw dicts - callers can
wrap them back into Pydantic models if they want type-checking.
"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger("persistence.mongo")


MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "biolistener_sleep_dev")


_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
        logger.info("Mongo client -> %s", MONGO_URL)
    return _client


def get_db(db_name: Optional[str] = None) -> AsyncIOMotorDatabase:
    return get_client()[db_name or MONGO_DB]


async def init_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes - cheap if they already exist."""
    await db.subjects.create_index([("code", ASCENDING)], unique=True)
    await db.sessions.create_index([("subject_id", ASCENDING),
                                    ("started_at", ASCENDING)])
    await db.agent_decisions.create_index([("session_id", ASCENDING),
                                           ("t_ms", ASCENDING)])
    await db.stim_events.create_index([("session_id", ASCENDING),
                                       ("t_ms", ASCENDING)])
    await db.safety_events.create_index([("session_id", ASCENDING),
                                         ("t_ms", ASCENDING)])
    await db.feature_snapshots.create_index([("session_id", ASCENDING),
                                             ("t_ms", ASCENDING)])
    await db.policy_checkpoints.create_index([("algo", ASCENDING),
                                              ("version", ASCENDING)])


async def insert_one(db: AsyncIOMotorDatabase, collection: str,
                      doc: Dict[str, Any]) -> str:
    res = await db[collection].insert_one(doc)
    return str(res.inserted_id)


async def insert_many(db: AsyncIOMotorDatabase, collection: str,
                       docs: List[Dict[str, Any]]) -> List[str]:
    if not docs:
        return []
    res = await db[collection].insert_many(docs)
    return [str(i) for i in res.inserted_ids]


async def find(db: AsyncIOMotorDatabase, collection: str,
               filt: Dict[str, Any], limit: int = 0,
               sort: Optional[List] = None) -> AsyncIterator[Dict[str, Any]]:
    cursor = db[collection].find(filt)
    if sort:
        cursor = cursor.sort(sort)
    if limit:
        cursor = cursor.limit(limit)
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        yield doc


async def get_by_id(db: AsyncIOMotorDatabase, collection: str,
                    doc_id: str) -> Optional[Dict[str, Any]]:
    try:
        oid = ObjectId(doc_id)
    except Exception:
        return None
    doc = await db[collection].find_one({"_id": oid})
    if doc is not None:
        doc["_id"] = str(doc["_id"])
    return doc


async def update_one(db: AsyncIOMotorDatabase, collection: str,
                      filt: Dict[str, Any], update: Dict[str, Any]) -> int:
    res = await db[collection].update_one(filt, {"$set": update})
    return int(res.modified_count)
