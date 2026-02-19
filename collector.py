# collector.py
import requests
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_phishtank():
    try:
        url = "https://data.phishtank.com/data/online-valid.json"
        logger.info(f"Fetching data from PhishTank: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTP errors
        data = response.json()
        logger.info(f"Fetched {len(data)} entries from PhishTank")
        return data
    except Exception as e:
        logger.error(f"PhishTank Error: {e}")
        return []

def fetch_openphish():
    try:
        url = "https://openphish.com/feed.txt"
        logger.info(f"Fetching data from OpenPhish: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.split('\n')
        urls = [{"url": line.strip(), "source": "openphish"} for line in lines if line.strip()]
        logger.info(f"Fetched {len(urls)} URLs from OpenPhish")
        return urls
    except Exception as e:
        logger.error(f"OpenPhish Error: {e}")
        return []

def fetch_data():
    """Fetch data from all sources and return as a list"""
    phishtank_data = fetch_phishtank()
    openphish_data = fetch_openphish()
    
    # Combine data from both sources
    all_data = phishtank_data + openphish_data
    logger.info(f"Total phishing entries fetched: {len(all_data)}")
    return all_data

if __name__ == "__main__":
    # For testing purposes
    data = fetch_data()
    print(f"Fetched {len(data)} total entries")