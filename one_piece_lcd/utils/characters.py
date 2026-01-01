"""Character data access utilities."""

import json
from pathlib import Path
from typing import Optional

from ..constants.paths import (
    INDIVIDUALS_DIR,
    AFFILIATIONS_DIR,
    CHARACTER_JSON_FILENAME,
    DEFAULT_IMAGE_EXTENSION,
)


def get_character_json(character_id: str) -> Optional[dict]:
    """
    Load character JSON data by character ID.
    
    Args:
        character_id: The normalized character ID (e.g., "monkey_d_luffy")
        
    Returns:
        Character data dict, or None if not found
    """
    character_dir = INDIVIDUALS_DIR / character_id
    json_path = character_dir / CHARACTER_JSON_FILENAME
    
    if not json_path.exists():
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_affiliation_character_ids(affiliation_id: str) -> list[str]:
    """
    Get list of character IDs belonging to an affiliation.
    
    Args:
        affiliation_id: The affiliation ID (e.g., "straw_hat_pirates")
        
    Returns:
        List of character IDs in the affiliation
    """
    affiliation_path = AFFILIATIONS_DIR / f"{affiliation_id}.json"
    
    if not affiliation_path.exists():
        return []
    
    with open(affiliation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_characters_by_affiliations(affiliation_ids: list[str]) -> list[dict]:
    """
    Get all character data for characters in the specified affiliations.
    
    Args:
        affiliation_ids: List of affiliation IDs to include
        
    Returns:
        List of character data dicts (duplicates removed by character_id)
    """
    seen_ids: set[str] = set()
    characters: list[dict] = []
    
    for affiliation_id in affiliation_ids:
        character_ids = get_affiliation_character_ids(affiliation_id)
        
        for character_id in character_ids:
            if character_id in seen_ids:
                continue
            
            character_data = get_character_json(character_id)
            if character_data:
                characters.append(character_data)
                seen_ids.add(character_id)
    
    return characters


def get_character_face_paths(character_id: str) -> list[Path]:
    """
    Get paths to all face images for a character.
    
    Args:
        character_id: The normalized character ID
        
    Returns:
        List of Path objects to image files
    """
    character_dir = INDIVIDUALS_DIR / character_id
    
    if not character_dir.exists():
        return []
    
    # Find all image files in the character directory
    image_paths: list[Path] = []
    
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        image_paths.extend(character_dir.glob(f"*{ext}"))
    
    # Sort by filename to ensure consistent ordering
    return sorted(image_paths)


def get_all_character_ids() -> list[str]:
    """
    Get all available character IDs.
    
    Returns:
        List of all character IDs in the individuals directory
    """
    if not INDIVIDUALS_DIR.exists():
        return []
    
    return [
        d.name for d in INDIVIDUALS_DIR.iterdir()
        if d.is_dir() and (d / CHARACTER_JSON_FILENAME).exists()
    ]

