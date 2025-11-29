"""
An autonomous agent for AI news curation and social media management.

Author: ranahaani
Version: 2.0.0
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import aiohttp
import google.generativeai as genai
import replicate
import requests
from dotenv import load_dotenv
from gnews import GNews
from instagrapi import Client
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import re
from openai import OpenAI

from .image_generator import ImageGeneratorFactory, ImageGenerator

load_dotenv()


class AgentState(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting_news"
    ANALYZING = "analyzing_content"
    GENERATING = "generating_content"
    POSTING = "posting_content"
    ERROR = "error"


class AgentMetrics(BaseModel):
    news_collected: int = 0
    posts_created: int = 0
    successful_posts: int = 0
    failed_posts: int = 0
    last_run: Optional[datetime] = None
    average_engagement: float = 0.0
    top_performing_topics: List[str] = []


@dataclass
class AgentMemory:
    posted_titles: Set[str] = None
    performance_history: Dict[str, float] = None
    topic_performance: Dict[str, float] = None

    def __post_init__(self):
        self.posted_titles = set()
        self.performance_history = {}
        self.topic_performance = {}

    def save(self, path: Path) -> None:
        data = {
            "posted_titles": list(self.posted_titles),
            "performance_history": self.performance_history,
            "topic_performance": self.topic_performance
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> 'AgentMemory':
        if not path.exists():
            return cls()

        data = json.loads(path.read_text())
        memory = cls()
        memory.posted_titles = set(data["posted_titles"])
        memory.performance_history = data["performance_history"]
        memory.topic_performance = data["topic_performance"]
        return memory


class AgentConfig(BaseModel):
    name: str = "AI News Agent"
    personality: str = "professional"
    news_sources: List[str] = ["gnews"]
    post_frequency: int = 3  # posts per day
    max_news_age: int = 24  # hours
    engagement_threshold: float = 0.5
    memory_path: Path = Path("../agent_memory.json")
    output_dir: Path = Path("../output")
    credentials: Dict[str, str] = {}
    image_generation: Dict[str, Any] = {
        "default_provider": "gemini",
        "providers": {
            "gemini": {
                "model": "gemini-3-pro-image-preview",
                "aspect_ratio": "1:1"
            },
            "openai": {
                "model": "gpt-image-1",
                "size": "1024x1024",
                "quality": "medium",
            },
            "replicate": {
                "model": "ideogram-ai/ideogram-v2"
            }
        }
    }


class BrandTheme(BaseModel):
    logo_url: Optional[str] = None
    primary_color: str = "#000000"
    secondary_color: str = "#FFFFFF"
    background_color: str = "#F0F0F0"
    accent_color: str = "#F0F0F0"
    text_color: str = "#333333"
    font_style: str = "Arial"
    visual_style: str = "futuristic"
    content_tone: str = "informative"


class BrandManager:
    """Manages brand assets and theme"""

    def __init__(self, theme: BrandTheme):
        self.theme = theme
        self.logo: Optional[Image.Image] = self._load_logo()
        self.logger = logging.getLogger(__name__)
        self.style_guide = {
            "tech_motifs": [
                "neural networks", "quantum computing", "robotics",
                "data streams", "circuit boards", "holograms"
            ],
            "composition_rules": [
                "Rule of thirds layout",
                "Negative space for text placement",
                "Dynamic lighting effects"
            ]
        }

    @property
    def theme_prompt(self) -> str:
        """Generate detailed theme prompt for image generation"""
        return (
            f"Brand Style Guide: "
            f"Color Palette: Primary {self.theme.primary_color}, "
            f"Secondary {self.theme.secondary_color}, Accent {self.theme.accent_color}. "
            f"Font: {self.theme.font_style} in {self.theme.text_color}. "
            f"Visual Style: {self.theme.visual_style} with {self.style_guide['composition_rules'][0]}. "
            f"Incorporate subtle elements of {random.choice(self.style_guide['tech_motifs'])}. "
            f"Text must be clearly readable with contrast against background. "
            f"Use modern UI elements and professional tech visualization techniques."
        )

    def _load_logo(self) -> Optional[Image.Image]:
        """Load brand logo from URL"""
        if not self.theme.logo_url:
            return None

        try:
            response = requests.get(self.theme.logo_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGBA")
        except Exception as e:
            self.logger.warning(f"Failed to load logo: {e}")
            return None


class AINewsAgent:

    def __init__(self, config: AgentConfig, theme: BrandTheme):
        self.config = config
        self._state = AgentState.IDLE
        self.memory = AgentMemory.load(config.memory_path)
        self.metrics = AgentMetrics()
        self.logger = self._setup_logging()
        self.brand_manager = BrandManager(theme)
        self._prompt_cache = {}  # Cache for enhanced prompts

        # Ensure the output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self._init_ai_components()

        self.logger.info(f"Agent '{config.name}' initialized")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: AgentState):
        """Log state changes whenever the state updates."""
        if self._state != new_state:
            self.logger.info(f"{new_state.value.capitalize().replace('_', ' ')}...")
        self._state = new_state

    def _setup_logging(self) -> logging.Logger:
        """Set up agent logging"""
        logger = logging.getLogger(self.config.name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        fh = logging.FileHandler(f"{self.config.name.lower()}_agent.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def _init_ai_components(self) -> None:
        try:
            if self.config.credentials.get("GEMINI_API_KEY"):
                genai.configure(api_key=self.config.credentials["GEMINI_API_KEY"])
                self.content_analyzer = genai.GenerativeModel("gemini-1.5-flash")
            else:
                self.content_analyzer = None
                self.logger.warning("Gemini AI not configured")

            # Initialize image generator based on configuration
            self.image_generator = ImageGeneratorFactory.create(
                provider=self.config.image_generation["default_provider"],
                credentials=self.config.credentials,
                config=self.config.image_generation
            )
            
            if not self.image_generator:
                self.logger.warning(f"Image generator ({self.config.image_generation['default_provider']}) not configured")

            if all(k in self.config.credentials for k in ["IG_USERNAME", "IG_PASSWORD"]):
                self.instagram = Client()
                self._login_instagram()
            else:
                self.instagram = None
                self.logger.warning("Instagram client not configured")

        except Exception as e:
            self.logger.error(f"Failed to initialize AI components: {e}")
            self.state = AgentState.ERROR

    def _login_instagram(self) -> None:
        """Login to Instagram with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.instagram.login(
                    self.config.credentials["IG_USERNAME"],
                    self.config.credentials["IG_PASSWORD"]
                )
                self.logger.info("Successfully logged into Instagram")
                return
            except Exception as e:
                self.logger.warning(f"Login attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("Instagram login failed after max retries")
                    raise

    async def run(self) -> None:
        """Main agent loop"""
        self.logger.info("Agent starting...")

        while True:
            try:
                await self._execute_cycle()
                self.memory.save(self.config.memory_path)
                self.metrics.last_run = datetime.now()
                await asyncio.sleep(self._calculate_next_run_delay())

            except Exception as e:
                self.logger.error(f"Error in agent cycle: {e}")
                self.state = AgentState.ERROR
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _execute_cycle(self) -> None:
        """Execute one full agent cycle"""
        self.state = AgentState.COLLECTING
        news_items = await self._collect_news()

        if not news_items:
            self.logger.info("No new news items found")
            return

        self.state = AgentState.ANALYZING
        selected_news = await self._analyze_news(news_items)

        if not selected_news:
            self.logger.info("No suitable news items for posting")
            return

        self.state = AgentState.GENERATING
        content = await self._generate_content(selected_news)

        if not content:
            self.logger.warning("Failed to generate content")
            return

        self.state = AgentState.POSTING
        success = await self._post_content(content)

        if success:
            self._update_memory(selected_news, content)
            self.metrics.successful_posts += 1
        else:
            self.metrics.failed_posts += 1

        self.state = AgentState.IDLE

    async def _collect_news(self) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.config.news_sources:
                tasks.append(self._fetch_from_source(session, source))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_news = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching news: {result}")
                    continue
                all_news.extend(result)

            self.metrics.news_collected += len(all_news)
            return all_news

    async def _fetch_from_source(self, session: aiohttp.ClientSession, source: str) -> List[Dict[str, Any]]:
        if source == "gnews":
            return await self._fetch_gnews()
        # TODO: Add more sources here
        return []

    async def _fetch_gnews(self) -> List[Dict[str, Any]]:
        """Fetch news from GNews"""
        try:
            gnews = GNews(max_results=10, period='1d', country='US')
            news = gnews.get_news('AI News')
            return news
        except Exception as e:
            self.logger.error(f"Error fetching from GNews: {e}")
            return []

    async def _analyze_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """AI-powered news analysis with engagement prediction.
        If the response is not valid JSON (e.g., wrapped in Markdown formatting), re-prompt the AI to provide only JSON."""
        base_prompt = (
            f"Analyze these AI news titles for social media potential:\n"
            f"{chr(10).join([item['title'] for item in news_items])}\n\n"
            f"Score each 1-10 based on:\n"
            f"1. Technical significance\n"
            f"2. Audience interest\n"
            f"3. Visual potential\n"
            f"4. Uniqueness\n"
            f"5. Discussion potential\n"
            f"Return JSON with scores and selection reason. "
            f"ONLY return valid JSON with no extra commentary."
        )

        max_attempts = 2
        attempt = 0
        analysis = None

        while attempt < max_attempts:
            try:
                response = await asyncio.to_thread(
                    self.content_analyzer.generate_content,
                    base_prompt
                )
                # Log the raw response for debugging
                self.logger.debug(f"Content analyzer response (attempt {attempt +1}): {response.text}")

                raw_response = response.text.strip()
                # Check if the response is wrapped in markdown code fences (e.g., ```json ... ```)
                if raw_response.startswith("```"):
                    lines = raw_response.splitlines()
                    # Remove the first line if it starts with triple backticks (and possibly a language marker)
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # Remove the last line if it is the closing triple backticks
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    raw_response = "\n".join(lines).strip()

                analysis = json.loads(raw_response)
                break  # Successfully parsed JSON, exit loop
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}: Failed to parse JSON response: {e}. "
                    f"Re-prompting AI for valid JSON."
                )
                # Append an extra instruction to the prompt to force a JSON-only response
                base_prompt += "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."
                attempt += 1

        if analysis is None:
            self.logger.error("Advanced news analysis failed after multiple attempts.")
            return news_items[:3]  # Fallback: simply use the first 3 news items

        # Assuming the returned JSON has a "titles" key that is a list of objects containing scores and reasons
        scores = analysis.get("titles", [])
        sorted_items = sorted(
            zip(news_items, scores),
            key=lambda x: sum([x[1].get(key, 0) for key in [
                "technical_significance", "audience_interest", 
                "visual_potential", "uniqueness", "discussion_potential"
            ]]),
            reverse=True
        )[:3]

        return [item[0] for item in sorted_items]
    async def _generate_caption(self, news_items: List[Dict[str, Any]]) -> str:
        """Generate engaging social media caption"""
        try:
            caption_template = (
                "🚀 {hook}\n\n"
                "🔍 Key Highlights:\n"
                "{bullets}\n\n"
                "💡 Why This Matters:\n"
                "{analysis}\n\n"
                "{hashtags}"
            )

            news_text = "\n".join([item['title'] for item in news_items])

            analysis_prompt = (
                f"Generate social media caption for these AI news stories:\n{news_text}\n"
                f"Format:\n"
                f"- 1 emoji + attention-grabbing hook (max 12 words)\n"
                f"- 3 bullet points with key technical details\n"
                f"- Short 'Why This Matters' analysis (50-70 words)\n"
                f"Tone: {self.brand_manager.theme.content_tone}"
            )

            response = await asyncio.to_thread(
                self.content_analyzer.generate_content,
                analysis_prompt
            )

            structured_text = response.text.strip().split("\n\n")

            return caption_template.format(
                hook=structured_text[0],
                bullets="\n".join([f"• {line}" for line in structured_text[1].split("\n")]),
                analysis=structured_text[2],
                hashtags=" ".join(await self._generate_hashtags(news_items))
            )
        except Exception as e:
            self.logger.error(f"Caption generation failed: {e}")
            return "\n".join([item['title'] for item in news_items])

    async def _generate_content(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate content for posting with concurrent image generation and caption/hashtag generation"""
        try:
            async with aiohttp.ClientSession() as session:
                image_tasks = [self._generate_image(item['title'], session=session) for item in news_items]
                caption_task = self._generate_caption(news_items)
                hashtags_task = self._generate_hashtags(news_items)
                images, caption, hashtags = await asyncio.gather(
                    asyncio.gather(*image_tasks),
                    caption_task,
                    hashtags_task
                )
            # Filter out any None images
            valid_images = [img for img in images if img]
            content = {
                "images": valid_images,
                "caption": caption + "\n" + " ".join(hashtags),
                "hashtags": hashtags
            }
            return content

        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            return None

    async def _generate_image(self, text: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Path]:
        """Generate an image for the given news title using the configured image generator."""
        if not self.image_generator:
            return None
            
        try:
            prompt = await self._enhance_prompt(text)
            filename = self._sanitize_filename(text)[:50] + ".jpg"
            image_path = self.config.output_dir / filename
            
            result_path = await self.image_generator.generate(prompt, image_path)
            
            if result_path:
                self.logger.info(f"Image generated and saved to {result_path}")
                return result_path
            return None

        except Exception as e:
            self.logger.error(f"Image generation failed for '{text}': {e}")
            return None

    async def _enhance_prompt(self, text: str) -> str:
        """Generate optimized prompt for Gemini 3 Pro image generation with caching.
        
        Leverages Gemini's superior text rendering and instruction-following capabilities.
        """
        if text in self._prompt_cache:
            return self._prompt_cache[text]
        try:
            # Get AI-powered visual concept
            analysis_prompt = (
                f"Create a concise visual concept for this AI news headline:\n"
                f"'{text}'\n\n"
                f"Describe in 2-3 sentences:\n"
                f"- Main visual metaphor or scene\n"
                f"- Key visual elements that represent the technology\n"
                f"- Mood and atmosphere\n\n"
                f"Return ONLY the visual description, no explanations."
            )

            response = await asyncio.to_thread(
                self.content_analyzer.generate_content,
                analysis_prompt
            )

            visual_concept = response.text.strip()

            # Structured prompt optimized for Gemini 3 Pro
            enhanced_prompt = (
                f"Create a professional social media image for AI/tech news.\n\n"
                f"VISUAL CONCEPT:\n{visual_concept}\n\n"
                f"HEADLINE TEXT TO RENDER:\n\"{text}\"\n\n"
                f"DESIGN SPECIFICATIONS:\n"
                f"- Style: {self.brand_manager.theme.visual_style}, modern, clean\n"
                f"- Color scheme: Primary {self.brand_manager.theme.primary_color}, "
                f"Accent {self.brand_manager.theme.accent_color}, "
                f"Background {self.brand_manager.theme.background_color}\n"
                f"- Typography: Bold, sans-serif, high contrast, fully readable\n"
                f"- Text placement: Prominent position with dark gradient backdrop for legibility\n"
                f"- Composition: Professional tech visualization with depth and dimension\n"
                f"- Quality: High-resolution, sharp details, no artifacts\n\n"
                f"IMPORTANT: Render the headline text clearly and accurately. "
                f"Text must be the focal point and completely legible."
            )
            
            self._prompt_cache[text] = enhanced_prompt
            return enhanced_prompt

        except Exception as e:
            self.logger.error(f"Prompt enhancement failed for '{text}': {e}")
            # Improved fallback prompt for Gemini 3 Pro
            fallback = (
                f"Professional tech news social media image.\n"
                f"Headline: \"{text}\"\n"
                f"Style: {self.brand_manager.theme.visual_style}, futuristic\n"
                f"Colors: {self.brand_manager.theme.primary_color} primary, dark background\n"
                f"Render headline text prominently with high contrast and readability."
            )
            self._prompt_cache[text] = fallback
            return fallback

    async def _generate_hashtags(self, news_items: List[Dict[str, Any]]) -> List[str]:
        """Generate context-aware hashtags"""
        try:
            news_context = "\n".join([item['title'] for item in news_items])

            prompt = (
                f"Generate 8-10 relevant hashtags for these AI news stories:\n{news_context}\n"
                f"Mix of:\n- General tech trends\n- Specific technologies mentioned\n- Industry applications\n"
                f"Prioritize hashtags with 10k-1M posts\nReturn only hashtags separated by commas"
            )

            response = await asyncio.to_thread(
                self.content_analyzer.generate_content,
                prompt
            )

            return [tag.strip() for tag in response.text.split(",") if tag.strip()]
        except Exception as e:
            self.logger.error(f"Hashtag generation failed: {e}")
            return ["#AI", "#TechNews", "#Innovation", "#FutureTech"]

    async def _post_content(self, content: Dict[str, Any]) -> bool:
        if not self.instagram:
            self.logger.warning("Instagram client not configured")
            return False

        images = content.get("images", [])
        caption = content.get("caption", "")

        # Validate images
        valid_images: List[str] = []
        for img in images:
            path = Path(img)
            if not path.exists():
                self.logger.error(f"Missing image: {img}")
                continue
            valid_images.append(path)

        if not valid_images:
            self.logger.error("No valid images to post")
            return False

        try:
            if len(valid_images) == 1:
                result = await asyncio.to_thread(
                    self.instagram.photo_upload,
                    valid_images[0],
                    caption
                )
            else:
                result = await asyncio.to_thread(
                    self.instagram.album_upload,
                    valid_images,
                    caption
                )

            self.logger.info(f"Posted successfully! Media ID: {result.id}")
            return True

        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            return False

    def _update_memory(self, news_items: List[Dict[str, Any]], content: Dict[str, Any]) -> None:
        """Update agent's memory with new content"""
        for item in news_items:
            self.memory.posted_titles.add(item['title'])

        # Save memory
        self.memory.save(self.config.memory_path)

    def _calculate_next_run_delay(self) -> int:
        """Calculate delay until next run based on post frequency"""
        seconds_per_day = 86400
        delay = seconds_per_day / self.config.post_frequency
        return int(delay)

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """Sanitize the text to create a filesystem-safe filename."""
        # Remove any character that is not alphanumeric, underscore, or dash
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', '_', text)
        return text
