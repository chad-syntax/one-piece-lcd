"""Character model for normalized character data."""

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from ..utils.normalization import normalize_name, normalize_field_name, parse_affiliations


class CharacterModel(BaseModel):
    """Pydantic model for normalized character data."""
    character_id: str = Field(..., description="Normalized character ID")
    name: str = Field(..., description="Original character name")
    images: str = Field(default="", description="Original images string")
    image_urls: List[str] = Field(default_factory=list, description="Parsed list of image URLs")
    portraits: str = Field(default="", description="Original portraits string")
    image_paths: List[str] = Field(default_factory=list, description="Local paths to downloaded images")
    face_image_paths: List[str] = Field(default_factory=list, description="Local paths to cropped face images")
    affiliations: List[str] = Field(default_factory=list, description="List of parsed affiliation names")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Normalized statistics")
    portrayal: Dict[str, Any] = Field(default_factory=dict, description="Normalized portrayal data")
    devil_fruit: Optional[Dict[str, Any]] = Field(default=None, description="Devil fruit information if present")
    
    @classmethod
    def from_wiki_data(cls, character_name: str, wiki_data: Dict[str, Any]) -> "CharacterModel":
        """Create a CharacterModel from wiki JSON data."""
        character_id = normalize_name(character_name)
        
        # Normalize statistics
        statistics = {}
        if "Statistics" in wiki_data:
            for key, value in wiki_data["Statistics"].items():
                normalized_key = normalize_field_name(key)
                # Handle duplicate keys (e.g., multiple empty string keys or same normalized name)
                if normalized_key in statistics:
                    # If key already exists, make it a list or append
                    if not isinstance(statistics[normalized_key], list):
                        statistics[normalized_key] = [statistics[normalized_key]]
                    statistics[normalized_key].append(value)
                else:
                    statistics[normalized_key] = value
        
        # Normalize portrayal
        portrayal = {}
        if "Portrayal" in wiki_data:
            for key, value in wiki_data["Portrayal"].items():
                normalized_key = normalize_field_name(key)
                # Handle duplicate keys (e.g., multiple empty string keys)
                if normalized_key in portrayal:
                    # If key already exists, make it a list or append
                    if not isinstance(portrayal[normalized_key], list):
                        portrayal[normalized_key] = [portrayal[normalized_key]]
                    portrayal[normalized_key].append(value)
                else:
                    portrayal[normalized_key] = value
        
        # Check for devil fruit (stored under "null" key sometimes)
        # Normalize devil fruit fields if present
        devil_fruit = None
        if "null" in wiki_data and isinstance(wiki_data["null"], dict):
            devil_fruit = {}
            for key, value in wiki_data["null"].items():
                normalized_key = normalize_field_name(key)
                devil_fruit[normalized_key] = value
        
        # Parse affiliations
        affiliations_str = statistics.get("affiliations", "") or statistics.get("affiliation", "")
        affiliations = parse_affiliations(affiliations_str) if affiliations_str else []
        
        # Parse image URLs
        images_str = wiki_data.get("images", "")
        image_urls = [url.strip() for url in images_str.split(",") if url.strip()] if images_str else []
        
        return cls(
            character_id=character_id,
            name=character_name,
            images=images_str,
            image_urls=image_urls,
            portraits=wiki_data.get("portraits", ""),
            affiliations=affiliations,
            statistics=statistics,
            portrayal=portrayal,
            devil_fruit=devil_fruit,
        )

