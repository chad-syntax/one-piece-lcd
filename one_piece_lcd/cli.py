"""CLI entry point for processing wiki data."""

import asyncio
import sys

import click

from .constants.paths import INDIVIDUALS_DIR, AFFILIATIONS_DIR, SAMPLE_VIDEO_PATH


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
    from .processors.character import process_all_characters, process_character_faces
    
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
    "--episode", "-e",
    type=int,
    help="Episode number to load characters from (uses scraped episode data)"
)
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
    "--face-tolerance",
    default=0.5,
    type=float,
    help="Face embedding similarity threshold (0-1, default: 0.5)"
)
@click.option(
    "--char-tolerance",
    default=0.5,
    type=float,
    help="Full-image character SigLIP similarity threshold (0-1, default: 0.5)"
)
@click.option(
    "--face-threshold",
    default=0.3,
    type=float,
    help="Face detection confidence threshold (0-1, default: 0.3)"
)
@click.option(
    "--face-iou",
    default=0.5,
    type=float,
    help="Face detection IOU threshold for NMS (0-1, default: 0.5, lower = more overlapping)"
)
@click.option(
    "--face-imgsz",
    default=640,
    type=int,
    help="Face detection input size (default: 640, try 1280 for better accuracy)"
)
@click.option(
    "--face-augment",
    is_flag=True,
    help="Enable test-time augmentation for face detection (slower but more accurate)"
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Number of frames to process in parallel (default: 8, increase for faster GPU processing)"
)
def process_video(
    sample: bool,
    video: str | None,
    episode: int | None,
    affiliations: tuple[str, ...],
    frame_skip: int,
    face_tolerance: float,
    char_tolerance: float,
    face_threshold: float,
    face_iou: float,
    face_imgsz: int,
    face_augment: bool,
    batch_size: int,
):
    """Run character recognition on a video file.
    
    Uses YOLOv8 AnimeFace for detection and SigLIP for embeddings.
    Matches both detected faces AND full-frame against character images.
    
    Saves results to assets/videos/sample/face-detections-{timestamp}.json
    
    Examples:
    
        oplcd process-video --sample --episode 1
        
        oplcd process-video --sample -a straw_hat_pirates
        
        oplcd process-video --video path/to/video.mp4 -a straw_hat_pirates -a beasts_pirates
    """
    from .processors.video import VideoFaceRecognition
    from .utils.characters import (
        get_characters_by_affiliations,
        get_all_character_ids,
        get_episode_character_ids,
    )
    
    # Determine video path
    if sample:
        video_path = SAMPLE_VIDEO_PATH
    elif video:
        video_path = video
    else:
        click.echo("Error: Must specify --sample or --video", err=True)
        sys.exit(1)
    
    # Determine which characters to use (episode takes priority)
    affiliation_list = list(affiliations)
    
    if episode is not None:
        character_ids = get_episode_character_ids(episode)
        if not character_ids:
            click.echo(f"Error: No characters found for episode {episode}. Run 'oplcd scrape-episodes' first.", err=True)
            sys.exit(1)
        click.echo(f"Using {len(character_ids)} characters from episode {episode}", err=True)
    elif affiliation_list:
        characters = get_characters_by_affiliations(affiliation_list)
        character_ids = [c["character_id"] for c in characters]
        click.echo(f"Using {len(character_ids)} characters from affiliations: {', '.join(affiliation_list)}", err=True)
    else:
        character_ids = get_all_character_ids()
        click.echo(f"Warning: No episode or affiliations specified, using all {len(character_ids)} characters (this may be slow)", err=True)
    
    if not character_ids:
        click.echo("Error: No characters found to match against", err=True)
        sys.exit(1)
    
    # Initialize and run recognition
    recognizer = VideoFaceRecognition(
        video_path=video_path,
        character_ids=character_ids,
        affiliations=affiliation_list,
        frame_skip=frame_skip,
        face_tolerance=face_tolerance,
        character_tolerance=char_tolerance,
        face_detection_threshold=face_threshold,
        face_iou_threshold=face_iou,
        face_imgsz=face_imgsz,
        face_augment=face_augment,
        batch_size=batch_size,
    )
    recognizer.process_video()
    
    # Save results to file
    recognizer.save_results()
    
    # Print summary to stderr
    recognizer.print_summary()


@cli.command("scrape-episodes")
@click.option(
    "--start",
    default=1,
    type=int,
    help="Episode number to start from (default: 1)"
)
@click.option(
    "--end",
    default=None,
    type=int,
    help="Episode number to end at (default: scrape until 404)"
)
@click.option(
    "--concurrency",
    default=20,
    type=int,
    help="Number of concurrent requests (default: 20)"
)
def scrape_episodes(start: int, end: int | None, concurrency: int):
    """Scrape episode data from the One Piece Wiki.
    
    Fetches episode titles, airdates, and character appearances.
    Saves data to assets/episodes/<episode_id>/episode.json
    
    Examples:
    
        oplcd scrape-episodes
        
        oplcd scrape-episodes --start 100 --end 200
        
        oplcd scrape-episodes --concurrency 10
    """
    from .scrapers.episodes import scrape_episodes_parallel
    
    scraped_count = 0
    
    def on_episode_scraped(episode):
        nonlocal scraped_count
        scraped_count += 1
        click.echo(f"[{scraped_count}] Episode {episode.episode_id}: {episode.title} ({len(episode.characters_in_order_of_appearance)} characters)")
    
    end_str = str(end) if end else "end"
    click.echo(f"Scraping episodes {start}-{end_str} with {concurrency} concurrent requests...")
    
    episodes = asyncio.run(scrape_episodes_parallel(
        start_episode=start,
        end_episode=end,
        concurrency=concurrency,
        on_episode_scraped=on_episode_scraped,
    ))
    
    click.echo(f"\nScraped {len(episodes)} episodes successfully")


@cli.command("serve")
@click.option(
    "--port",
    default=3000,
    type=int,
    help="Port to run the server on (default: 3000)"
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)"
)
def serve(port: int, host: str):
    """Start the face detection viewer web server.
    
    Opens a web interface to visualize face detection results
    overlaid on video playback.
    
    Examples:
    
        oplcd serve
        
        oplcd serve --port 8080
    """
    from .server import run_server
    run_server(host=host, port=port)
