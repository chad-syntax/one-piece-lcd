"""Normalization utilities for names and field names."""

import re
import unicodedata
from typing import List


def normalize_name(name: str) -> str:
    """Normalize a name to only alphanumeric characters and underscores."""
    # Normalize Unicode characters to NFD, then remove accents
    name = unicodedata.normalize('NFD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    # Lowercase
    name = name.lower()
    # Replace spaces and dashes with underscores
    name = re.sub(r"[ \-]", "_", name)
    # Remove all non-alphanumeric or underscore
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def normalize_field_name(field_name: str) -> str:
    """Convert a field name like 'Romanized Name' to 'romanized_name'."""
    # Handle empty string keys (often used for bounty)
    if not field_name or field_name.strip() == "":
        return "bounty_or_unknown"
    
    # Normalize Unicode
    field_name = unicodedata.normalize('NFD', field_name)
    field_name = "".join([c for c in field_name if not unicodedata.combining(c)])
    # Lowercase
    field_name = field_name.lower()
    # Replace spaces and special chars with underscores
    field_name = re.sub(r"[^a-z0-9]", "_", field_name)
    # Remove multiple underscores
    field_name = re.sub(r"_+", "_", field_name)
    # Remove leading/trailing underscores
    field_name = field_name.strip("_")
    
    # Ensure we have a valid field name
    if not field_name:
        return "unknown_field"
    
    return field_name


def parse_affiliations(affiliations_str: str) -> List[str]:
    """Parse affiliations string into a list of normalized affiliation names."""
    if not affiliations_str or not isinstance(affiliations_str, str):
        return []
    
    # Split by common separators (semicolon, comma, etc.)
    affiliations = []
    for sep in [';', ',', '\n']:
        if sep in affiliations_str:
            affiliations = [aff.strip() for aff in affiliations_str.split(sep)]
            break
    
    # If no separator found, treat as single affiliation
    if not affiliations:
        affiliations = [affiliations_str.strip()]
    
    # Clean up affiliations - remove empty strings and references like [1], [2], etc.
    cleaned = []
    for aff in affiliations:
        aff = aff.strip()
        # Remove citation references like [1], [2], etc.
        aff = re.sub(r'\[\d+\]', '', aff).strip()
        # Remove other common patterns
        aff = re.sub(r'\s*\([^)]*\)\s*$', '', aff).strip()  # Remove trailing parentheses
        if aff:
            # Normalize the affiliation name
            normalized_aff = normalize_name(aff)
            if normalized_aff and normalized_aff not in cleaned:
                cleaned.append(normalized_aff)
    
    return cleaned

