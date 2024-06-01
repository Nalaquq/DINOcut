from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
import time
from TikTokApi import TikTokApi
import os

def download_tiktok_video(url, output_path='tiktok_video.mp4'):
    try:
        # Set up Selenium WebDriver with headless option
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Load the TikTok URL
        driver = webdriver.Chrome() 
        driver.get(url)
        
        # Wait for the page to load completely
        time.sleep(10)  # Increased wait time
        
        # Retrieve cookies
        cookies = driver.get_cookies()
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        
        # Find the video tag
        video_tag = driver.find_element(By.TAG_NAME, 'video')
        
        # Extract the video URL
        video_url = video_tag.get_attribute('src')
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Referer': url,
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        }
        
        # Download the video content using session with cookies
        video_response = session.get(video_url, headers=headers)
        video_response.raise_for_status()

        # Write the video content to a file
        with open(output_path, 'wb') as file:
            file.write(video_response.content)
        
        print(f"Video downloaded successfully and saved as {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        driver.quit()

def search_videos_by_hashtag(hashtag, num_videos=5, download_path='downloads'):
    try:
        # Set up Selenium WebDriver with headless option
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Specify the path to your ChromeDriver
        driver = webdriver.Chrome() 
        driver.get(url)
        
        # Open TikTok and search for the hashtag
        search_url = f'https://www.tiktok.com/tag/{hashtag}'
        driver.get(search_url)
        
        # Wait for the page to load completely
        time.sleep(10)
        
        # Get the page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find video links
        video_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/video/' in href and href not in video_links:
                video_links.append(href)
            if len(video_links) >= num_videos:
                break
        
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        
        for idx, video_link in enumerate(video_links):
            video_url = f"https://www.tiktok.com{video_link}"
            output_file = os.path.join(download_path, f"video_{idx + 1}.mp4")
            print(f"Downloading video {idx + 1} from URL: {video_url}")
            driver.get(video_url)
            
            time.sleep(5)  # Ensure the video page loads completely
            
            # Extract video URL from video tag
            video_tag = driver.find_element(By.TAG_NAME, 'video')
            video_src = video_tag.get_attribute('src')
            
            download_tiktok_video(video_src, output_file)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        driver.quit()

'''
# Example usage
hashtag = 'examplehashtag'
search_videos_by_hashtag(hashtag, num_videos=5, download_path='downloads')
'''


def download_tiktok_video(video_url, output_path='tiktok_video.mp4'):
    try:
        # Initialize TikTokApi
        api = TikTokApi.get_instance()

        # Get the video ID from the URL
        video_id = video_url.split('/')[-1].split('?')[0]

        # Get video data
        video_data = api.video(id=video_id).bytes()

        # Write the video content to a file
        with open(output_path, 'wb') as file:
            file.write(video_data)

        print(f"Video downloaded successfully and saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def search_videos_by_hashtag(hashtag, num_videos=5, download_path='downloads'):
    try:
        # Initialize TikTokApi
        api = TikTokApi.get_instance()

        # Search for videos by hashtag
        hashtag_videos = api.by_hashtag(hashtag, count=num_videos)

        # Create download directory if it doesn't exist
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        for idx, video in enumerate(hashtag_videos):
            video_url = f"https://www.tiktok.com/@{video['author']['uniqueId']}/video/{video['id']}"
            output_file = os.path.join(download_path, f"video_{idx + 1}.mp4")
            print(f"Downloading video {idx + 1} from URL: {video_url}")
            download_tiktok_video(video_url, output_file)

    except Exception as e:
        print(f"An error occurred: {e}")

'''
# Example usage
hashtag = 'examplehashtag'
search_videos_by_hashtag(hashtag, num_videos=5, download_path='downloads')
'''

# Example usage
tiktok_url = 'https://www.tiktok.com/@sewyupik/video/7151587944799669550'
download_tiktok_video(tiktok_url, 'downloaded_tiktok.mp4')
