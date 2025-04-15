from subprocess import run
import os

def run_model_download():
    print()
    print("Checking for BERT model...")
    print()
    
    zip_file = "finetuned_bert_sentiment_model.zip"
    model_dir = "finetuned_bert_sentiment_model"
    
    # check if the model is already there
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print("BERT model already exists. Skipping download and extraction.")
    # check if the zip file exists
    elif os.path.exists(zip_file) and os.path.getsize(zip_file) > 0:
        print(f"Zip file {zip_file} already exists. Extracting only...")
        # only extraction part
        run(['python', 'download_bert.py', '--extract-only'])
    else:
        print("BERT model not found. Downloading and extracting...")
        run(['python', 'download_bert.py'])

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
    run_model_download()
    run_scraper()
    run_analyser()