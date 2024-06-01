from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import requests


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
import time

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


# Example usage
tiktok_url = 'https://www.tiktok.com/@sewyupik/video/7151587944799669550'
download_tiktok_video(tiktok_url, 'downloaded_tiktok.mp4')
