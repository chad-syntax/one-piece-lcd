"""Character processing functions."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import aiofiles
import aiohttp

from ..constants.config import MAX_CONCURRENT_PROCESSING
from ..constants.paths import (
    CHARACTERS_DIR,
    INDIVIDUALS_DIR,
    INDIVIDUALS_DIR_NAME,
    AFFILIATIONS_DIR,
    WIKI_DATA_PATH,
    ALL_CHARACTERS_JSON_PATH,
    IMAGE_FILENAME_PATTERN,
    DEFAULT_IMAGE_EXTENSION,
    CHARACTER_JSON_FILENAME,
)
from ..models.character import CharacterModel


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    filepath: Path,
    force_refresh: bool = False
) -> tuple[bool, bool]:
    """Download an image from a URL and save it to a filepath.
    
    Args:
        session: aiohttp session
        url: URL to download from
        filepath: Path to save to
        force_refresh: If True, re-download even if file exists
    
    Returns:
        tuple[bool, bool]: (success, was_downloaded) - success indicates if file exists/valid,
                          was_downloaded indicates if an actual download occurred
    """
    # Check if file already exists (skip if not forcing refresh)
    if filepath.exists() and not force_refresh:
        return (True, False)  # Success, but not downloaded
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                return (True, True)  # Success and downloaded
            else:
                print(f"Failed to download {url}: HTTP {response.status}")
                return (False, False)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return (False, False)


def generate_character_image_paths(character: CharacterModel, character_dir: Path) -> List[str]:
    """Generate image paths for a character based on their image URLs."""
    image_paths = []
    
    if not character.image_urls:
        return image_paths
    
    # Get the character_id from the directory name
    character_id = character_dir.name
    
    # Generate paths for each image URL
    for idx, url in enumerate(character.image_urls, start=1):
        # Determine file extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1] or DEFAULT_IMAGE_EXTENSION
        # Remove query parameters from extension
        if "?" in ext:
            ext = ext.split("?")[0]
        
        filename = IMAGE_FILENAME_PATTERN.format(idx=idx, ext=ext)
        # Store relative path from project root: ./assets/characters/individuals/{normalized_character_name}/image_1.png
        relative_path = f"./{CHARACTERS_DIR}/{INDIVIDUALS_DIR_NAME}/{character_id}/{filename}"
        image_paths.append(relative_path)
    
    return image_paths


async def download_character_images(
    character: CharacterModel,
    character_dir: Path,
    session: aiohttp.ClientSession,
    force_refresh: bool = False
) -> int:
    """Download all images for a character concurrently.
    
    Args:
        character: Character model with image URLs
        character_dir: Directory to save images to
        session: aiohttp session
        force_refresh: If True, re-download even if files exist
    
    Returns:
        Number of images actually downloaded (not skipped)
    """
    if not character.image_urls:
        return 0
    
    # Prepare download tasks
    download_tasks = []
    for idx, url in enumerate(character.image_urls, start=1):
        # Determine file extension from URL (same logic as generate_character_image_paths)
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1] or DEFAULT_IMAGE_EXTENSION
        if "?" in ext:
            ext = ext.split("?")[0]
        
        filename = IMAGE_FILENAME_PATTERN.format(idx=idx, ext=ext)
        filepath = character_dir / filename
        download_tasks.append(download_image(session, url, filepath, force_refresh))
    
    # Download all images concurrently
    results = await asyncio.gather(*download_tasks, return_exceptions=True)
    
    # Count actual downloads (not skipped)
    actual_downloads = 0
    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            _, was_downloaded = result
            if was_downloaded:
                actual_downloads += 1
    
    return actual_downloads


async def process_character(
    character_name: str,
    wiki_data: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    session: Optional[aiohttp.ClientSession] = None,
    force_refresh: bool = False
) -> Optional[CharacterModel]:
    """Process a single character: create model, generate image paths, download images, save JSON."""
    async with semaphore:  # Limit concurrent processing
        try:
            # Create character model
            character = CharacterModel.from_wiki_data(character_name, wiki_data)
            
            # Create character directory in individuals folder
            character_dir = INDIVIDUALS_DIR / character.character_id
            character_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate image paths
            image_paths = generate_character_image_paths(character, character_dir)
            character.image_paths = image_paths
            
            # Download images if session is provided
            if session and character.image_urls:
                num_downloaded = await download_character_images(
                    character, character_dir, session, force_refresh
                )
                if num_downloaded > 0:
                    print(f"Downloaded {num_downloaded} images for {character.name}")
            
            # Save character.json
            character_json_path = character_dir / CHARACTER_JSON_FILENAME
            async with aiofiles.open(character_json_path, 'w', encoding='utf-8') as f:
                await f.write(character.model_dump_json(indent=2))
            
            return character
        except Exception as e:
            print(f"Error processing character {character_name}: {e}")
            return None


def collect_affiliations(characters: List[CharacterModel]) -> Dict[str, List[str]]:
    """Collect affiliations from a list of CharacterModels and build a map of normalized_affiliation -> character_ids."""
    affiliation_map: Dict[str, List[str]] = {}
    
    for character in characters:
        # Add character to each affiliation's collection
        # Note: affiliations are already normalized from parse_affiliations
        for affiliation in character.affiliations:
            if affiliation not in affiliation_map:
                affiliation_map[affiliation] = []
            if character.character_id not in affiliation_map[affiliation]:
                affiliation_map[affiliation].append(character.character_id)
    
    return affiliation_map


async def process_all_characters(force_refresh: bool = False):
    """Process all characters in parallel, collect affiliations, and save all output files.
    
    Args:
        force_refresh: If True, re-download images even if they exist.
    """
    # Load wiki data
    with open(WIKI_DATA_PATH, "r", encoding="utf-8") as f:
        wiki_characters = json.load(f)
    
    print(f"Found {len(wiki_characters)} characters in wiki data")
    
    all_characters: List[CharacterModel] = []
    
    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSING)
    
    # Create HTTP session for image downloads
    async with aiohttp.ClientSession() as session:
        # Create tasks for all characters
        tasks = []
        for wiki_character in wiki_characters:
            character_name = list(wiki_character.keys())[0]
            character_data = wiki_character[character_name]
            tasks.append(process_character(
                character_name, character_data, semaphore, session, force_refresh
            ))
        
        # Process all characters (await inside session context to ensure session stays open)
        print(f"Processing {len(tasks)} characters...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    for result in results:
        if isinstance(result, CharacterModel):
            all_characters.append(result)
        elif isinstance(result, Exception):
            print(f"Exception occurred: {result}")
    
    # Collect affiliations
    affiliation_map = collect_affiliations(all_characters)
    
    # Extract character IDs for all_characters.json
    all_character_ids = [char.character_id for char in all_characters]
    
    # Save all_characters.json
    with open(ALL_CHARACTERS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_character_ids, f, indent=2, ensure_ascii=False)
    
    # Save affiliation files for each affiliation
    print(f"\nCreating {len(affiliation_map)} affiliation files...")
    for normalized_affiliation, character_ids in affiliation_map.items():
        affiliation_file = AFFILIATIONS_DIR / f"{normalized_affiliation}.json"
        with open(affiliation_file, 'w', encoding='utf-8') as f:
            json.dump(character_ids, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(all_characters)} characters")
    print(f"Created {len(affiliation_map)} affiliation files")
    print(f"Individuals saved to: {INDIVIDUALS_DIR}")
    print(f"Affiliations saved to: {AFFILIATIONS_DIR}")
    
    return all_characters


def _process_single_character(
    args: tuple
) -> Optional[tuple[str, List[str], int]]:
    """Process faces for a single character. Used by ThreadPoolExecutor."""
    char_dir, force_refresh, processor, idx, total_chars = args
    
    character_id = char_dir.name
    character_json_path = char_dir / CHARACTER_JSON_FILENAME
    
    if not character_json_path.exists():
        return None
    
    # Load character data
    with open(character_json_path, 'r', encoding='utf-8') as f:
        char_data = json.load(f)
    
    character_name = char_data.get("name", character_id)
    
    # Get image paths
    image_paths = char_data.get("image_paths", [])
    face_image_paths = []
    char_faces_detected = 0
    
    for img_path in image_paths:
        # Convert relative path to absolute
        abs_path = Path(img_path.lstrip("./"))
        if not abs_path.exists():
            continue
        
        # Check if face images already exist
        existing_faces = list(abs_path.parent.glob(f"{abs_path.stem}_face_*.png"))
        
        if existing_faces and not force_refresh:
            # Use existing face images
            for face_path in existing_faces:
                rel_path = f"./{face_path}"
                face_image_paths.append(rel_path)
                
                # Generate embedding if not cached
                processor.get_or_create_embedding(face_path, force_refresh=force_refresh)
            char_faces_detected += len(existing_faces)
        else:
            # Detect and crop faces
            try:
                saved_faces = processor.crop_and_save_faces(abs_path)
                for face_path in saved_faces:
                    rel_path = f"./{face_path}"
                    face_image_paths.append(rel_path)
                    
                    # Generate and cache embedding
                    processor.get_or_create_embedding(face_path, force_refresh=True)
                char_faces_detected += len(saved_faces)
            except Exception as e:
                print(f"[{idx}/{total_chars}] Error processing {character_name}: {e}")
    
    # Log results for this character
    if char_faces_detected > 0:
        print(f"[{idx}/{total_chars}] Detected {char_faces_detected} face(s) for {character_name}")
    
    if face_image_paths:
        # Update character.json with face paths
        char_data["face_image_paths"] = face_image_paths
        with open(character_json_path, 'w', encoding='utf-8') as f:
            json.dump(char_data, f, indent=2, ensure_ascii=False)
        return (character_id, face_image_paths, char_faces_detected)
    
    return None


def process_character_faces(
    character_ids: Optional[List[str]] = None,
    force_refresh: bool = False,
    max_workers: int = 4
) -> Dict[str, List[str]]:
    """
    Process faces for characters: detect, crop, and generate embeddings.
    Uses ThreadPoolExecutor for parallel processing.
    
    Args:
        character_ids: List of character IDs to process. If None, processes all.
        force_refresh: If True, reprocess even if face images exist.
        max_workers: Number of parallel workers (default: 4)
        
    Returns:
        Dict mapping character_id to list of face image paths
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .faces import AnimeFaceProcessor
    
    # Initialize processor (this will load models before parallelization)
    processor = AnimeFaceProcessor()
    # Pre-load models to avoid race conditions
    _ = processor.detector
    _ = processor.clip_model
    
    # Get character directories to process
    if character_ids:
        char_dirs = [INDIVIDUALS_DIR / cid for cid in character_ids if (INDIVIDUALS_DIR / cid).exists()]
    else:
        char_dirs = [d for d in INDIVIDUALS_DIR.iterdir() if d.is_dir()]
    
    total_chars = len(char_dirs)
    print(f"Processing faces for {total_chars} characters using {max_workers} workers...")
    
    # Prepare arguments for each character
    args_list = [
        (char_dir, force_refresh, processor, idx, total_chars)
        for idx, char_dir in enumerate(char_dirs, 1)
    ]
    
    results: Dict[str, List[str]] = {}
    total_faces = 0
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_character, args) for args in args_list]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                character_id, face_paths, face_count = result
                results[character_id] = face_paths
                total_faces += face_count
    
    print(f"\nCompleted: {total_faces} faces for {len(results)} characters")
    return results

