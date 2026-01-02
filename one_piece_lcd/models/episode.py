"""Episode model for scraped episode data."""

from typing import List

from pydantic import BaseModel, Field


class EpisodeModel(BaseModel):
    """Pydantic model for episode data scraped from the One Piece Wiki."""
    
    episode_id: int = Field(..., description="Episode number")
    title: str = Field(..., description="Episode title")
    airdate: str = Field(default="", description="Original airdate")
    characters_in_order_of_appearance: List[str] = Field(
        default_factory=list,
        description="List of normalized character IDs in order of appearance"
    )

