from pydantic import BaseModel
from typing import List, Optional, Any

class RecommendRequest(BaseModel):
    user_input: str
    
class ItemScore(BaseModel):
    item_id: str
    item_name: str
    score: float
    explanation: Optional[str] = None
    image: Optional[str] = None
    price: Optional[float] = None
    average_rating: Optional[float] = None
    rating_number: Optional[int] = None
    store: Optional[str] = None
    
class RecommendResponse(BaseModel):
    recommendations: List[ItemScore]
    latency: Optional[dict] = None
