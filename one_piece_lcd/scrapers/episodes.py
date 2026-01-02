"""Episode scraper for One Piece Wiki."""

import asyncio
import json
import re
from html.parser import HTMLParser
from typing import Callable, Optional

import aiohttp

from ..constants.paths import EPISODES_WIKI_JSON_PATH
from ..models.episode import EpisodeModel
from ..utils.normalization import normalize_name


WIKI_BASE_URL = "https://onepiece.fandom.com/wiki"
STARTING_EPISODE_URL = f"{WIKI_BASE_URL}/Episode_1"
MAX_CONCURRENT_REQUESTS = 20


class EpisodeHTMLParser(HTMLParser):
    """Parse episode page HTML to extract episode data."""
    
    def __init__(self):
        super().__init__()
        self.title: str = ""
        self.airdate: str = ""
        self.characters: list[str] = []
        self.next_episode_url: Optional[str] = None
        
        # State tracking
        self._in_translation_h2 = False  # The title like "I'm Luffy!..."
        self._in_airdate_div = False
        self._in_airdate_value = False
        self._in_characters_section = False  # After "Characters in Order of Appearance" h2
        self._in_characters_list = False
        self._in_character_link = False
        self._current_character_name = ""
        self._list_depth = 0
        
    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        attrs_dict = dict(attrs)
        
        # Episode title: <h2 data-source="Translation">
        if tag == "h2" and attrs_dict.get("data-source") == "Translation":
            self._in_translation_h2 = True
            
        # Airdate section: <div data-source="Airdate">
        if tag == "div" and attrs_dict.get("data-source") == "Airdate":
            self._in_airdate_div = True
            
        # Airdate value is in <div class="pi-data-value">
        if self._in_airdate_div and tag == "div" and "pi-data-value" in (attrs_dict.get("class") or ""):
            self._in_airdate_value = True
            
        # Characters section starts with <h2><span id="Characters_in_Order_of_Appearance">
        if tag == "span" and attrs_dict.get("id") == "Characters_in_Order_of_Appearance":
            self._in_characters_section = True
            
        # Track ul/ol lists after characters section header
        if self._in_characters_section and tag in ("ul", "ol"):
            self._list_depth += 1
            if self._list_depth == 1:
                self._in_characters_list = True
                
        # Character links in the list
        if self._in_characters_list and tag == "a":
            href = attrs_dict.get("href") or ""
            # Only capture wiki links to characters (not anchors, episodes, or external)
            if href.startswith("/wiki/") and not href.startswith("/wiki/Episode_") and not href.startswith("/wiki/Chapter_"):
                self._in_character_link = True
                
    def handle_endtag(self, tag: str):
        if tag == "h2":
            self._in_translation_h2 = False
            # If we were in characters section and hit another h2, we're done
            if self._in_characters_section and self._in_characters_list:
                self._in_characters_section = False
                self._in_characters_list = False
            
        if tag == "div":
            if self._in_airdate_value:
                self._in_airdate_value = False
            elif self._in_airdate_div:
                self._in_airdate_div = False
                
        if self._in_characters_section and tag in ("ul", "ol"):
            self._list_depth -= 1
            if self._list_depth == 0:
                # First list after characters header is done
                self._in_characters_list = False
                self._in_characters_section = False
                
        if tag == "a":
            if self._in_character_link and self._current_character_name:
                # Normalize the character name
                normalized = normalize_name(self._current_character_name)
                if normalized and normalized not in self.characters:
                    self.characters.append(normalized)
                self._current_character_name = ""
            self._in_character_link = False
            
    def handle_data(self, data: str):
        data_stripped = data.strip()
        if not data_stripped:
            return
            
        if self._in_translation_h2 and not self.title:
            self.title = data_stripped
            
        if self._in_airdate_value and not self.airdate:
            self.airdate = data_stripped
            
        if self._in_character_link:
            self._current_character_name += data_stripped


def parse_episode_html(html: str, episode_num: int) -> tuple[EpisodeModel, Optional[str]]:
    """
    Parse episode HTML and return episode data and next episode URL.
    
    Args:
        html: Raw HTML content
        episode_num: Current episode number
        
    Returns:
        Tuple of (EpisodeModel, next_episode_url or None)
    """
    parser = EpisodeHTMLParser()
    parser.feed(html)
    
    # Look for next episode link using regex (more reliable)
    next_url: Optional[str] = None
    next_ep_pattern = rf'href="/wiki/Episode_{episode_num + 1}"'
    if re.search(next_ep_pattern, html):
        next_url = f"{WIKI_BASE_URL}/Episode_{episode_num + 1}"
    
    episode = EpisodeModel(
        episode_id=episode_num,
        title=parser.title or f"Episode {episode_num}",
        airdate=parser.airdate,
        characters_in_order_of_appearance=parser.characters,
    )
    
    return episode, next_url


def load_episodes_data() -> dict[str, dict]:
    """Load existing episodes data from JSON file."""
    if EPISODES_WIKI_JSON_PATH.exists():
        with open(EPISODES_WIKI_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_episodes_data(episodes: dict[str, dict]) -> None:
    """Save all episodes data to a single JSON file."""
    EPISODES_WIKI_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EPISODES_WIKI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)


async def fetch_episode(
    session: aiohttp.ClientSession,
    url: str,
    episode_num: int,
    semaphore: asyncio.Semaphore,
) -> tuple[EpisodeModel, Optional[str]]:
    """
    Fetch and parse a single episode page.
    
    Args:
        session: aiohttp session
        url: Episode URL to fetch
        episode_num: Episode number
        semaphore: Semaphore for rate limiting
        
    Returns:
        Tuple of (EpisodeModel, next_episode_url or None)
    """
    async with semaphore:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch {url}: {response.status}")
            html = await response.text()
            return parse_episode_html(html, episode_num)


async def scrape_episodes_sequential(
    start_episode: int = 1,
    max_episodes: Optional[int] = None,
    on_episode_scraped: Optional[Callable[[EpisodeModel], None]] = None,
) -> list[EpisodeModel]:
    """
    Scrape episodes sequentially following next episode links.
    
    This is slower but ensures we follow the wiki's episode chain.
    
    Args:
        start_episode: Episode number to start from
        max_episodes: Maximum number of episodes to scrape (None for all)
        on_episode_scraped: Callback when an episode is scraped
        
    Returns:
        List of scraped EpisodeModels
    """
    episodes: list[EpisodeModel] = []
    semaphore = asyncio.Semaphore(1)  # Sequential
    
    current_url = f"{WIKI_BASE_URL}/Episode_{start_episode}"
    current_num = start_episode
    
    async with aiohttp.ClientSession() as session:
        while current_url:
            if max_episodes and len(episodes) >= max_episodes:
                break
                
            try:
                episode, next_url = await fetch_episode(
                    session, current_url, current_num, semaphore
                )
                episodes.append(episode)
                
                if on_episode_scraped:
                    on_episode_scraped(episode)
                
                current_url = next_url
                current_num += 1
                
            except Exception as e:
                print(f"Error scraping episode {current_num}: {e}")
                break
    
    # Save all episodes at the end
    episodes_data = load_episodes_data()
    for episode in episodes:
        episodes_data[str(episode.episode_id)] = episode.model_dump()
    save_episodes_data(episodes_data)
    
    return episodes


async def scrape_episodes_parallel(
    start_episode: int = 1,
    end_episode: Optional[int] = None,
    concurrency: int = MAX_CONCURRENT_REQUESTS,
    on_episode_scraped: Optional[Callable[[EpisodeModel], None]] = None,
) -> list[EpisodeModel]:
    """
    Scrape episodes in parallel with bounded concurrency.
    
    This assumes episode URLs follow the pattern Episode_N.
    
    Args:
        start_episode: Episode number to start from
        end_episode: Episode number to end at (None = scrape until 404)
        concurrency: Maximum concurrent requests
        on_episode_scraped: Callback when an episode is scraped
        
    Returns:
        List of scraped EpisodeModels
    """
    semaphore = asyncio.Semaphore(concurrency)
    episodes: list[EpisodeModel] = []
    episodes_lock = asyncio.Lock()
    
    async def scrape_one(session: aiohttp.ClientSession, episode_num: int) -> bool:
        """Returns True if episode exists, False if 404."""
        url = f"{WIKI_BASE_URL}/Episode_{episode_num}"
        
        try:
            async with semaphore:
                async with session.get(url) as response:
                    if response.status == 404:
                        return False
                    if response.status != 200:
                        print(f"Warning: Episode {episode_num} returned {response.status}")
                        return True  # Assume exists but failed
                    html = await response.text()
                    
            episode, _ = parse_episode_html(html, episode_num)
            
            async with episodes_lock:
                episodes.append(episode)
                
            if on_episode_scraped:
                on_episode_scraped(episode)
            
            return True
                
        except Exception as e:
            print(f"Error scraping episode {episode_num}: {e}")
            return True  # Assume exists but failed
    
    async with aiohttp.ClientSession() as session:
        if end_episode is not None:
            # Known range - create all tasks at once
            tasks = [
                scrape_one(session, ep_num)
                for ep_num in range(start_episode, end_episode + 1)
            ]
            await asyncio.gather(*tasks)
        else:
            # Unknown range - scrape in batches until we hit 404
            batch_size = concurrency * 2
            current_start = start_episode
            
            while True:
                batch_end = current_start + batch_size
                tasks = [
                    scrape_one(session, ep_num)
                    for ep_num in range(current_start, batch_end)
                ]
                results = await asyncio.gather(*tasks)
                
                # Check if any returned False (404)
                if False in results:
                    break
                    
                current_start = batch_end
    
    # Sort by episode_id
    episodes.sort(key=lambda e: e.episode_id)
    
    # Save all episodes at the end
    episodes_data = load_episodes_data()
    for episode in episodes:
        episodes_data[str(episode.episode_id)] = episode.model_dump()
    save_episodes_data(episodes_data)
    
    return episodes


async def scrape_all_episodes(
    concurrency: int = MAX_CONCURRENT_REQUESTS,
    on_episode_scraped: Optional[Callable[[EpisodeModel], None]] = None,
) -> list[EpisodeModel]:
    """
    Scrape all episodes from the One Piece Wiki.
    
    Uses parallel requests with the specified concurrency level.
    Assumes episodes are numbered sequentially from 1.
    
    Args:
        concurrency: Maximum concurrent requests (default: 20)
        on_episode_scraped: Optional callback for progress reporting
        
    Returns:
        List of all scraped EpisodeModels
    """
    return await scrape_episodes_parallel(
        start_episode=1,
        end_episode=None,  # Scrape until 404
        concurrency=concurrency,
        on_episode_scraped=on_episode_scraped,
    )
