import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
import logging
from pathlib import Path
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketingCampaignScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.sources = [
            'https://www.marketingweek.com/tag/campaign-failures/',
            'https://www.adweek.com/category/advertising/',
            # Add more sources as needed
        ]
        
    def _get_page_content(self, url: str) -> str:
        """Fetch page content with retry mechanism."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
                continue
        return ""

    def _parse_campaign_data(self, html_content: str) -> List[Dict]:
        """Parse HTML content to extract campaign information."""
        soup = BeautifulSoup(html_content, 'html.parser')
        campaign_data = []
        
        # Example parsing logic - adjust based on actual website structure
        articles = soup.find_all('article')
        for article in articles:
            try:
                title = article.find('h2').text.strip()
                content = article.find('div', class_='content').text.strip()
                
                # Analyze content for failure indicators
                failure_data = {
                    "failure_topic": f"Failure_Topic: {title}",
                    "subtopics": self._analyze_failure_reasons(content)
                }
                campaign_data.append(failure_data)
            except Exception as e:
                logger.error(f"Error parsing article: {str(e)}")
                continue
                
        return campaign_data

    def _analyze_failure_reasons(self, content: str) -> List[str]:
        """Analyze content to identify failure reasons."""
        failure_indicators = [
            ("targeting", "Breakdown due to bad targeting"),
            ("design", "Breakdown due to poor visual design"),
            ("research", "Breakdown due to no customer research"),
            ("platform", "Breakdown due to wrong platform"),
            ("CTA", "Breakdown due to unclear CTA"),
            ("mobile", "Breakdown due to not mobile friendly"),
            ("timing", "Breakdown due to bad timing"),
            ("budget", "Breakdown due to budget misalignment")
        ]
        
        subtopics = []
        for indicator, failure_type in failure_indicators:
            if indicator.lower() in content.lower():
                subtopics.append(failure_type)
        
        return subtopics or ["Breakdown due to unspecified reasons"]

    def scrape_and_save(self, output_file: str = 'scraped_insights.jsonl'):
        """Scrape campaign data from all sources and save to JSONL file."""
        Path('data').mkdir(exist_ok=True)
        output_path = Path('data') / output_file
        
        all_campaign_data = []
        for source in self.sources:
            logger.info(f"Scraping from: {source}")
            content = self._get_page_content(source)
            if content:
                campaign_data = self._parse_campaign_data(content)
                all_campaign_data.extend(campaign_data)
            
            # Respect robots.txt with delay
            time.sleep(random.uniform(2, 4))
        
        # Save to JSONL file
        with open(output_path, 'a', encoding='utf-8') as f:
            for data in all_campaign_data:
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Scraped {len(all_campaign_data)} campaigns to {output_path}")
        return output_path

if __name__ == "__main__":
    scraper = MarketingCampaignScraper()
    scraper.scrape_and_save() 