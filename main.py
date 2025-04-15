from subprocess import run

def run_scraper():
    print()
    print("Running tweet scrapper...")
    print()
    keyword = input("Enter keyword for scraping: ")
    count = input("Enter number of tweets to scrape: ")
    run(['python', 'scrapper.py'], input=f"{keyword}\n{count}\n", text=True)

def run_analyser():
    print()
    print("Running tweet analyser...")
    print()
    run(['python', 'analyser.py'])

if __name__ == "__main__":
    run_scraper()
    run_analyser()