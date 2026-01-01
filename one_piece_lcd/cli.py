"""CLI entry point for processing wiki data."""

import asyncio
import json
import sys

import click

from .constants.paths import INDIVIDUALS_DIR, AFFILIATIONS_DIR, SAMPLE_VIDEO_PATH
from .processors.character import process_all_characters, process_character_faces
from .processors.video import VideoFaceRecognition
from .utils.characters import (
    get_characters_by_affiliations,
    get_all_character_ids,
)


@click.group()
def cli():
    """One Piece LCD - Character data processing CLI."""
    pass


@cli.command("process-wiki")
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Re-download images and reprocess faces even if they exist"
)
@click.option(
    "--force-refresh-faces",
    is_flag=True,
    help="Only reprocess faces (skip image downloads)"
)
@click.option(
    "--workers",
    default=4,
    type=int,
    help="Number of parallel workers for face processing (default: 4)"
)
def process_wiki(force_refresh: bool, force_refresh_faces: bool, workers: int):
    """Process wiki JSON data into organized dataset.
    
    Downloads character images, detects and crops anime faces,
    and generates CLIP embeddings for face matching.
    
    Examples:
    
        oplcd process-wiki
        
        oplcd process-wiki --force-refresh
        
        oplcd process-wiki --force-refresh-faces
    """
    # If only refreshing faces, skip the image download step
    if not force_refresh_faces:
        click.echo("Processing wiki to dataset...")
        
        # Create directories
        INDIVIDUALS_DIR.mkdir(parents=True, exist_ok=True)
        AFFILIATIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Process all characters (downloads images)
        asyncio.run(process_all_characters(force_refresh=force_refresh))
    else:
        click.echo("Skipping image downloads, only processing faces...")
    
    # Process faces (detect, crop, generate embeddings)
    click.echo("\nProcessing faces (detecting, cropping, generating embeddings)...")
    results = process_character_faces(
        force_refresh=force_refresh or force_refresh_faces,
        max_workers=workers
    )
    click.echo(f"\nProcessed faces for {len(results)} characters")
    
    click.echo("\nDataset processed successfully")


@cli.command("process-video")
@click.option("--sample", is_flag=True, help="Use assets/videos/sample.mp4")
@click.option("--video", type=click.Path(exists=True), help="Path to video file")
@click.option(
    "--affiliations", "-a",
    multiple=True,
    help="Filter characters by affiliation IDs (can specify multiple)"
)
@click.option(
    "--frame-skip",
    default=5,
    type=int,
    help="Process every Nth frame (default: 5)"
)
@click.option(
    "--tolerance",
    default=0.6,
    type=float,
    help="Face matching tolerance, lower = stricter (default: 0.6)"
)
@click.option(
    "--gap-threshold",
    default=1.0,
    type=float,
    help="Max gap in seconds between detections for same segment (default: 1.0)"
)
def process_video(
    sample: bool,
    video: str | None,
    affiliations: tuple[str, ...],
    frame_skip: int,
    tolerance: float,
    gap_threshold: float
):
    """Run facial recognition on a video file.
    
    Outputs JSON to stdout. Progress and summary go to stderr.
    
    Examples:
    
        oplcd process-video --sample -a straw_hat_pirates > output.json
        
        oplcd process-video --video path/to/video.mp4 -a straw_hat_pirates -a beasts_pirates
    """
    # Determine video path
    if sample:
        video_path = SAMPLE_VIDEO_PATH
    elif video:
        video_path = video
    else:
        click.echo("Error: Must specify --sample or --video", err=True)
        sys.exit(1)
    
    # Determine which characters to use
    if affiliations:
        characters = get_characters_by_affiliations(list(affiliations))
        character_ids = [c["character_id"] for c in characters]
        click.echo(f"Using {len(character_ids)} characters from affiliations: {', '.join(affiliations)}", err=True)
    else:
        character_ids = get_all_character_ids()
        click.echo(f"Warning: No affiliations specified, using all {len(character_ids)} characters (this may be slow)", err=True)
    
    if not character_ids:
        click.echo("Error: No characters found to match against", err=True)
        sys.exit(1)
    
    # Initialize and run recognition
    recognizer = VideoFaceRecognition(video_path)
    recognizer.load_known_faces(character_ids)
    recognizer.process_video(frame_skip=frame_skip, tolerance=tolerance)
    
    # Print summary to stderr
    recognizer.print_summary(gap_threshold=gap_threshold)
    
    # Output JSON to stdout
    results = recognizer.get_results(gap_threshold=gap_threshold)
    print(json.dumps(results, indent=2))

