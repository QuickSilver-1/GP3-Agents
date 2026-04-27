import uuid
from datetime import datetime
from typing import List
from enum import Enum
import grpc
from structlog import BoundLogger
import src.weaviate as weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter
from src.config import Weaviate

class MemoryType(Enum):
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    CONTEXT = "CONTEXT"
    CONVERSATION = "CONVERSATION"
    COMMAND = "COMMAND"
    RULE = "RULE"
    TEMPLATE = "TEMPLATE"

class ImportanceLevel(Enum):
    TRIVIAL = 1
    LOW = 2
    NORMAL = 3
    MEDIUM = 4
    HIGH = 5
    CRITICAL = 6

class WeaviateClient:
    def __init__(self, cfg: Weaviate, logger: BoundLogger, collection_name: str = "LongTermMemory"):
        self.collection_name = collection_name
        self.logger = logger
        self.client = weaviate.connect_to_local(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
        )
        
        if self.client.collections.exists(self.collection_name):
            return
        
        self.client.collections.create(
            name=self.collection_name,
            description="Long term memory for AI agent",
            properties=[
                Property(name="type", data_type=DataType.TEXT, description="Memory type"),
                Property(name="content", data_type=DataType.TEXT, description="Main content for memory"),
                Property(name="importance", data_type=DataType.INT, description="Level of data value"),
                Property(name="created_at", data_type=DataType.DATE, description="Data of entry creating"),
                Property(name="updated_at", data_type=DataType.DATE, description="Data of entry last updating")
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers()
        )
        
        self.logger.info("Weaviate client successfully created")
    
    def create_memory(self, type: MemoryType, importance: ImportanceLevel, content: str) -> str:
        collection = self.client.collections.get(self.collection_name)
        id = str(uuid.uuid4())
        
        properties = {
            "memory_id": id,
            "type": type.value,
            "content": content,
            "importance": importance.value,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        collection.data.insert(
            uuid=id,
            properties=properties,
        )
        
        return id
        
    def get_memories(self, types: List[MemoryType], importances: List[ImportanceLevel], query: str, limit: int = 10) -> List[dict]:
        collection = self.client.collections.get(self.collection_name)
        filters = []

        if types:
            filters.append(Filter.by_property("type").contains_any([t.value for t in types]))

        if importances:
            filters.append(Filter.by_property("importance").contains_any([i.value for i in importances]))

        where_filter = Filter.all_of(filters) if filters else None

        if not query or query.strip() == "":
            response = collection.query.fetch_objects(
                limit=limit,
                filters=where_filter,
                return_metadata=MetadataQuery.full()
            )

        else:
            response = collection.query.near_text(
                query=query,
                limit=limit,
                filters=where_filter,
                return_metadata=MetadataQuery.full()
            )
            
        memories = []
        for obj in response.objects:
            distance = getattr(obj.metadata, "distance", None)
            props = obj.properties
            memories.append({
                "id": props.get("memory_id"),
                "content": props.get("content"),
                "type": props.get("type"),
                "importance": props.get("importance"),
                "created_at": props.get("created_at"),
                "similarity_score": (1 - distance) if distance is not None else None
            })
        
        return memories
    
    def close(self):
        self.client.close()