from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
import time
from datetime import datetime
import csv

def scrape_nitter_tweets(keyword, count):
    """
    Scrape tweets from Nitter based on a keyword search
    
    Parameters:
    - keyword: The search term to look for
    - count: Number of tweets to scrape
    
    Returns:
    - Saves tweets to a CSV file named keyword_date.csv
    """
    # Setup Chrome options
    chrome_options = Options()
    # Uncomment the line below if you want to run in headless mode
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Go to Nitter
        driver.get("https://nitter.net/")
        print(f"Searching for tweets containing: {keyword}")
        
        # Try multiple methods to find the search input
        search_box = None
        try_selectors = [
            (By.NAME, "q"),
            (By.CSS_SELECTOR, "input[name='q']"),
            (By.CSS_SELECTOR, "input[placeholder='Search...']"),
            (By.XPATH, "//input[@name='q']"),
            (By.XPATH, "//input[@placeholder='Search...']")
        ]
        
        for selector_type, selector in try_selectors:
            try:
                # Wait for the element to be present
                search_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((selector_type, selector))
                )
                print(f"Found search box using {selector_type}: {selector}")
                break
            except TimeoutException:
                continue
                
        if search_box is None:
            print("Could not find search input. Taking screenshot for debugging.")
            driver.save_screenshot("nitter_debug.png")
            raise Exception("Failed to locate search input on the page")
            
        # Clear and input the search term
        search_box.clear()
        search_box.send_keys(keyword)
        
        # Try multiple methods to find the search button
        search_button = None
        try_button_selectors = [
            (By.CSS_SELECTOR, "span.icon-search"),
            (By.XPATH, "//span[contains(@class, 'icon-search')]"),
            (By.XPATH, "//button[@type='submit']"),
            (By.CSS_SELECTOR, "form input[type='submit']"),
            (By.CSS_SELECTOR, "button[type='submit']")
        ]
        
        for selector_type, selector in try_button_selectors:
            try:
                search_button = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                print(f"Found search button using {selector_type}: {selector}")
                break
            except TimeoutException:
                continue
                
        # If we couldn't find a button, just press Enter in the search box
        if search_button:
            search_button.click()
        else:
            print("Could not find search button, pressing Enter instead")
            search_box.send_keys(Keys.RETURN)
        
        # Wait for search results to load - try different selectors
        result_found = False
        result_selectors = [
            (By.CLASS_NAME, "timeline-item"),
            (By.CSS_SELECTOR, ".timeline .timeline-item"),
            (By.XPATH, "//div[contains(@class, 'timeline-item')]")
        ]
        
        for selector_type, selector in result_selectors:
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((selector_type, selector))
                )
                result_found = True
                print(f"Found search results using {selector_type}: {selector}")
                break
            except TimeoutException:
                continue
                
        if not result_found:
            print("No search results found. Taking screenshot for debugging.")
            driver.save_screenshot("nitter_no_results.png")
            raise Exception("No search results found")
        
        # Initialize list to store tweets
        tweets = []
        last_tweet_count = 0
        no_new_tweets_count = 0
        
        while len(tweets) < count:
            # Scroll down to load more tweets dynamically if needed
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for potential dynamic content to load
            
            # Extract tweets from current page
            tweet_selectors = [
                (By.CLASS_NAME, "timeline-item"),
                (By.CSS_SELECTOR, ".timeline .timeline-item"),
                (By.XPATH, "//div[contains(@class, 'timeline-item')]")
            ]
            
            tweet_elements = []
            for selector_type, selector in tweet_selectors:
                try:
                    tweet_elements = driver.find_elements(selector_type, selector)
                    if tweet_elements:
                        break
                except Exception:
                    continue
            
            if not tweet_elements:
                print("No tweet elements found")
                break
                
            print(f"Found {len(tweet_elements)} tweet elements")
            
            for tweet in tweet_elements:
                if len(tweets) >= count:
                    break
                    
                try:
                    # Extract tweet data - try multiple ways to find each element
                    username = ""
                    fullname = ""
                    tweet_text = ""
                    date = ""
                    
                    # Try to get username
                    try:
                        username = tweet.find_element(By.CLASS_NAME, "username").text
                    except NoSuchElementException:
                        try:
                            username = tweet.find_element(By.CSS_SELECTOR, ".tweet-header .username").text
                        except NoSuchElementException:
                            username = "Unknown"
                    
                    # Try to get fullname
                    try:
                        fullname = tweet.find_element(By.CLASS_NAME, "fullname").text
                    except NoSuchElementException:
                        try:
                            fullname = tweet.find_element(By.CSS_SELECTOR, ".tweet-header .fullname").text
                        except NoSuchElementException:
                            fullname = "Unknown"
                    
                    # Try to get tweet text
                    try:
                        tweet_text = tweet.find_element(By.CLASS_NAME, "tweet-content").text
                    except NoSuchElementException:
                        try:
                            tweet_text = tweet.find_element(By.CSS_SELECTOR, ".tweet-body .tweet-content").text
                        except NoSuchElementException:
                            tweet_text = "No text available"
                    
                    # Try to get date
                    try:
                        date_element = tweet.find_element(By.CLASS_NAME, "tweet-date").find_element(By.TAG_NAME, "a")
                        date = date_element.get_attribute("title") or date_element.text
                    except NoSuchElementException:
                        try:
                            date_element = tweet.find_element(By.CSS_SELECTOR, ".tweet-date a")
                            date = date_element.get_attribute("title") or date_element.text
                        except NoSuchElementException:
                            date = "Unknown date"
                    
                    # Check if we already have this tweet
                    tweet_data = {
                        "username": username,
                        "fullname": fullname,
                        "text": tweet_text,
                        "date": date
                    }
                    
                    # Only add if not already in our list (to avoid duplicates)
                    if tweet_data not in tweets:
                        tweets.append(tweet_data)
                        print(f"Scraped {len(tweets)}/{count} tweets")
                        
                except (NoSuchElementException, StaleElementReferenceException) as e:
                    print(f"Error extracting tweet data: {str(e)}")
                    continue
            
            # Check if we got any new tweets
            if len(tweets) == last_tweet_count:
                no_new_tweets_count += 1
                if no_new_tweets_count >= 3:  # If no new tweets after 3 attempts
                    print("No new tweets found after multiple attempts")
                    
                    # Try to find and click "Load more" button
                    load_more_found = False
                    load_more_selectors = [
                        (By.XPATH, "//a[contains(text(), 'Load more')]"),
                        (By.LINK_TEXT, "Load more"),
                        (By.PARTIAL_LINK_TEXT, "Load more"),
                        (By.CSS_SELECTOR, "a:contains('Load more')"),
                        (By.XPATH, "//a[contains(@href, 'cursor=')]")
                    ]
                    
                    for selector_type, selector in load_more_selectors:
                        try:
                            load_more = WebDriverWait(driver, 3).until(
                                EC.element_to_be_clickable((selector_type, selector))
                            )
                            driver.execute_script("arguments[0].scrollIntoView();", load_more)
                            driver.execute_script("arguments[0].click();", load_more)
                            load_more_found = True
                            print("Clicked 'Load more' button")
                            time.sleep(3)  # Wait longer for content to load
                            no_new_tweets_count = 0  # Reset counter
                            break
                        except TimeoutException:
                            continue
                    
                    if not load_more_found:
                        print("No 'Load more' button found. Cannot load additional tweets.")
                        break
            else:
                no_new_tweets_count = 0  # Reset counter if we got new tweets
                
            last_tweet_count = len(tweets)
        
        # Save tweets to CSV
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"{keyword}_{current_date}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['username', 'fullname', 'text', 'date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for tweet in tweets[:count]:  # Ensure we only save up to count tweets
                writer.writerow(tweet)
        
        print(f"Successfully scraped {len(tweets)} tweets and saved to {filename}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        driver.save_screenshot("error_screenshot.png")
        print("Screenshot saved as error_screenshot.png")
    
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    keyword = input()
    count = int(input())
    scrape_nitter_tweets(keyword, count)