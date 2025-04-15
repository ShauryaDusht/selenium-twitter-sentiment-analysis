import streamlit as st
from scrapper import scrape_nitter_tweets
import analyser
import os
import glob
from PIL import Image

st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="centered")

st.title("🐦 Twitter Sentiment Analyzer")
st.write("Scrape tweets using a keyword and run sentiment analysis using multiple models.")

# --- Input Section ---
keyword = st.text_input("Enter keyword to search tweets")
count = st.slider("Number of tweets to scrape", min_value=10, max_value=500, step=10, value=100)

if st.button("Scrape Tweets"):
    with st.spinner("Scraping tweets..."):
        scrape_nitter_tweets(keyword, count)
    st.success("Tweet scraping completed!")

if st.button("Run Sentiment Analysis"):
    with st.spinner("Analyzing tweets..."):
        analyser.main()
    st.success("Sentiment analysis completed!")

# --- Visual Output Section ---
def get_latest_output_dir():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        return None
    latest_csv = max(csv_files, key=os.path.getmtime)
    return os.path.splitext(latest_csv)[0]  # folder name is same as file name

output_dir = get_latest_output_dir()
if output_dir and os.path.isdir(output_dir):
    st.header("Visualizations")
    image_files = glob.glob(os.path.join(output_dir, "*.png"))
    for img_path in image_files:
        st.image(Image.open(img_path), caption=os.path.basename(img_path))

    st.header("CSV Downloads")
    for csv_file in ["bert_results.csv", "finetuned_bert_results.csv", "lr_results.csv"]:
        path = os.path.join(output_dir, csv_file)
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    label=f"Download {csv_file}",
                    data=f,
                    file_name=csv_file,
                    mime="text/csv"
                )
