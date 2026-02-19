# data_collector.py
import requests
from kafka import KafkaProducer
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092')

def fetch_phishtank():
    url = "https://data.phishtank.com/data/online-valid.json"
    response = requests.get(url)
    if response.ok:
        for entry in response.json():
            producer.send('phishing-urls', value=str(entry).encode())

def fetch_openphish():
    url = "https://openphish.com/feed.txt"
    response = requests.get(url)
    if response.ok:
        for line in response.text.split('\n'):
            if line.strip():
                producer.send('phishing-urls', value=line.encode())

if __name__ == "__main__":
    while True:
        fetch_phishtank()
        fetch_openphish()
        time.sleep(3600)  # Fetch every hour