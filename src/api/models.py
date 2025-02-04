from pydantic import BaseModel
from typing import Optional, Dict, Any

class FunctionExecuteRequest(BaseModel):
    function_name: str
    arguments: Dict[str, Any]
    call_id: str

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    supporting_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
