from pydantic import BaseModel, Field
from typing import Optional

class AgentMemoryItem(BaseModel):

    content: str = Field(..., description="The content of the memory item.")
    identifier: str = Field(..., description="Who generate the memory item.")
    timestamp: Optional[str] = Field(None, description="The timestamp when the memory item was created.")
    