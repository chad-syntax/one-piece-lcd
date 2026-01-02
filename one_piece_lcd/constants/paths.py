"""File and directory path constants."""

from pathlib import Path

# Directory names
ASSETS_DIR = Path("assets")
CHARACTERS_DIR = ASSETS_DIR / "characters"
VIDEOS_DIR = ASSETS_DIR / "videos"
INDIVIDUALS_DIR_NAME = "individuals"
AFFILIATIONS_DIR_NAME = "affiliations"

# Directory paths
INDIVIDUALS_DIR = CHARACTERS_DIR / INDIVIDUALS_DIR_NAME
AFFILIATIONS_DIR = CHARACTERS_DIR / AFFILIATIONS_DIR_NAME

# File paths
WIKI_DATA_PATH = ASSETS_DIR / "one-piece-characters-wiki.json"
ALL_CHARACTERS_JSON_PATH = CHARACTERS_DIR / AFFILIATIONS_DIR_NAME / "all_characters.json"

# File names (relative to directories)
CHARACTER_JSON_FILENAME = "character.json"

# Data files
EPISODES_WIKI_JSON_PATH = ASSETS_DIR / "one-piece-episodes-wiki.json"

# Image-related constants
IMAGE_FILENAME_PATTERN = "image_{idx}{ext}"
DEFAULT_IMAGE_EXTENSION = ".png"

# Video-related constants
SAMPLE_VIDEO_DIR = VIDEOS_DIR / "sample"
SAMPLE_VIDEO_PATH = SAMPLE_VIDEO_DIR / "sample.mp4"
FACE_DETECTIONS_FILENAME_PATTERN = "face-detections-{timestamp}.json"

