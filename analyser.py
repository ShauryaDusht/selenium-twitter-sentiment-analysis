import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os
import glob
import datetime
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import string
from matplotlib.gridspec import GridSpec
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
    
# Function to find the most recent CSV file in the directory
def get_most_recent_csv(directory='.'):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory")
    
    # Get the most recent file based on modification time
    most_recent = max(csv_files, key=os.path.getmtime)
    return most_recent

# Function to create output directory
def create_output_directory(csv_path):
    # Get the base filename without extension
    base_name = os.path.basename(csv_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    
    # Create directory name
    output_dir = f"{file_name_no_ext}"
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

# Load BERT model
def load_bert_model(model_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None

# Load Logistic Regression model
def load_lr_model(model_path, vectorizer_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open(vectorizer_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        
        return model, vectorizer
    except Exception as e:
        print(f"Error loading LR model: {e}")
        return None, None

# Function to analyze sentiment with BERT
def analyze_sentiment_bert(tweets, model, tokenizer, device='cpu'):
    results = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    for i in range(0, len(tweets), batch_size):
        batch_tweets = tweets[i:i+batch_size]
        
        # Skip empty tweets
        filtered_tweets = [tweet if isinstance(tweet, str) else "" for tweet in batch_tweets]
        
        try:
            # Tokenize the tweets
            inputs = tokenizer(filtered_tweets, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Convert to sentiment scores (0 to 1 scale)
            # Assuming binary classification where index 1 is positive
            scores = probabilities[:, 1].cpu().numpy()
            results.extend(scores)
        except Exception as e:
            print(f"Error analyzing batch: {e}")
            # If error, assign neutral sentiment
            results.extend([0.5] * len(batch_tweets))
    
    return results

# Function to analyze sentiment with Logistic Regression
def analyze_sentiment_lr(tweets, model, vectorizer):
    results = []
    
    # Process in batches
    batch_size = 100
    for i in range(0, len(tweets), batch_size):
        batch_tweets = tweets[i:i+batch_size]
        
        # Skip empty tweets
        filtered_tweets = [tweet if isinstance(tweet, str) else "" for tweet in batch_tweets]
        
        try:
            # Transform tweets
            tweet_vectorized = vectorizer.transform(filtered_tweets)
            
            # Get prediction probabilities
            proba = model.predict_proba(tweet_vectorized)
            
            # Extract positive class probability (assuming binary classification)
            scores = proba[:, 1]
            results.extend(scores)
        except Exception as e:
            print(f"Error analyzing batch with LR: {e}")
            # If error, assign neutral sentiment
            results.extend([0.5] * len(batch_tweets))
    
    return results

# Function to clean tweets for word cloud
def clean_tweet(tweet):
    if not isinstance(tweet, str):
        return ""
    # Remove URLs, user mentions, hashtags, special chars
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)

    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([word for word in tweet.lower().split() if word not in stop_words and len(word) > 2])
    return tweet

# Function to create wordcloud from tweets
def create_wordcloud(tweets):
    all_words = ' '.join([clean_tweet(tweet) for tweet in tweets])
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, contour_width=3, contour_color='steelblue').generate(all_words)
    return wordcloud

# Function to plot visualizations for model results
def plot_model_analysis(data, model_name, output_dir, text_column):
    # Add sentiment category with neutral zone
    data['sentiment_category'] = pd.cut(
        data['sentiment'], 
        bins=[0, 0.45, 0.55, 1], 
        labels=['Negative', 'Neutral', 'Positive'],
        include_lowest=True
    )
    
    # Adjust sentiment for visualization to -1 to 1 scale
    data['sentiment_adjusted'] = 2 * data['sentiment'] - 1
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Sentiment Value Histogram (-1 to +1)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data['sentiment_adjusted'], bins=20, ax=ax1, color='darkblue', alpha=0.7)
    ax1.axvspan(2*0.45-1, 2*0.55-1, alpha=0.2, color='gray')  # Highlight neutral zone
    ax1.set_xlabel('Sentiment Score (-1 to +1)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Sentiment Distribution - {model_name}')
    
    # 2. Sentiment Pie Chart (Positive, Negative, Neutral)
    ax2 = fig.add_subplot(gs[0, 1])
    sentiment_counts = data['sentiment_category'].value_counts()
    # Ensure we have all three categories for consistent pie chart
    for category in ['Positive', 'Neutral', 'Negative']:
        if category not in sentiment_counts:
            sentiment_counts[category] = 0
    
    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=['red', 'gray', 'green'], explode=[0.05, 0.05, 0.05])
    ax2.set_title(f'Sentiment Distribution - {model_name}')
    
    # 3. Word Cloud
    ax3 = fig.add_subplot(gs[1, 0])
    wordcloud = create_wordcloud(data[text_column])
    ax3.imshow(wordcloud)
    ax3.set_title(f'Word Cloud - {model_name}')
    ax3.axis('off')
    
    # 4. Tweet Length Distribution by Sentiment
    ax4 = fig.add_subplot(gs[1, 1])
    data['text_length'] = data[text_column].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
    
    for category, color in zip(['Positive', 'Neutral', 'Negative'], ['green', 'gray', 'red']):
        subset = data[data['sentiment_category'] == category]
        if not subset.empty:
            sns.histplot(subset['text_length'], bins=15, alpha=0.5, color=color, label=category, ax=ax4)
    
    ax4.set_xlabel('Text Length (words)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Text Length Distribution by Sentiment - {model_name}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_visualization.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return sentiment_counts

# Function to create model comparison visualization
def create_model_comparison(result_dfs, output_dir, text_column):
    try:
        print("Creating model comparison visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Extract sentiment distributions for each model
        model_names = list(result_dfs.keys())
        pos_percentages = []
        neu_percentages = []
        neg_percentages = []
        
        for model_name, df in result_dfs.items():
            # Get sentiment categories
            df['sentiment_category'] = pd.cut(
                df['sentiment'], 
                bins=[0, 0.45, 0.55, 1], 
                labels=['Negative', 'Neutral', 'Positive'],
                include_lowest=True
            )
            
            # Calculate percentages
            sentiment_counts = df['sentiment_category'].value_counts(normalize=True) * 100
            pos_percentages.append(sentiment_counts.get('Positive', 0))
            neu_percentages.append(sentiment_counts.get('Neutral', 0))
            neg_percentages.append(sentiment_counts.get('Negative', 0))
        
        # Plot sentiment distribution comparison
        width = 0.25
        x = np.arange(len(model_names))
        
        axes[0].bar(x, pos_percentages, width, label='Positive', color='green')
        axes[0].bar(x, neu_percentages, width, bottom=pos_percentages, label='Neutral', color='gray')
        axes[0].bar(x, neg_percentages, width, bottom=[p+n for p, n in zip(pos_percentages, neu_percentages)], label='Negative', color='red')
        
        axes[0].set_ylabel('Percentage')
        axes[0].set_title('Sentiment Distribution Across Models')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].legend()
        
        # Text length comparison
        for i, (model_name, df) in enumerate(result_dfs.items()):
            df['text_length'] = df[text_column].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
            
            axes[1].boxplot([
                df[df['sentiment_category'] == 'Positive']['text_length'],
                df[df['sentiment_category'] == 'Neutral']['text_length'],
                df[df['sentiment_category'] == 'Negative']['text_length']
            ], positions=[i*4, i*4+1, i*4+2], widths=0.6)
        
        axes[1].set_xlabel('Models and Sentiment Categories')
        axes[1].set_ylabel('Text Length')
        axes[1].set_title('Text Length Distribution by Model and Sentiment')
        
        # Set custom x-ticks
        tick_positions = []
        tick_labels = []
        for i, model in enumerate(model_names):
            tick_positions.extend([i*4, i*4+1, i*4+2])
            tick_labels.extend([f"{model}\nPos", f"{model}\nNeu", f"{model}\nNeg"])
        
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Sentiment score histograms
        for model_name, df in result_dfs.items():
            # Convert sentiment to -1 to 1 scale for visualization
            sentiment_adj = 2 * df['sentiment'] - 1
            sns.kdeplot(sentiment_adj, ax=axes[2], label=model_name)
        
        axes[2].axvspan(2*0.45-1, 2*0.55-1, alpha=0.2, color='gray')  # Highlight neutral zone
        axes[2].set_xlabel('Sentiment Score (-1 to +1)')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Sentiment Score Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'model_comparison.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating model comparison: {e}")

# Main function
def main():
    try:
        # Find most recent CSV file
        csv_path = get_most_recent_csv()
        print(f"Using most recent CSV file: {csv_path}")
        
        # Load the CSV data
        df = pd.read_csv(csv_path)
        
        # Map column names if needed
        text_column = 'text' if 'text' in df.columns else 'tweet'
        username_column = 'username' if 'username' in df.columns else None
        
        # Check required columns
        required_columns = [username_column, text_column]
        required_columns = [col for col in required_columns if col is not None]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create output directory
        output_dir = create_output_directory(csv_path)
        print(f"Created output directory: {output_dir}")
        
        # Device for PyTorch (use GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load models
        print("Loading models...")
        
        # Define model paths
        bert_model_dir = "bert_sentiment_model"
        ft_bert_model_dir = "finetuned_bert_sentiment_model"
        lr_model_path = "lr_sentiment_model/sentiment_lr_model.pkl"
        lr_vectorizer_path = "lr_sentiment_model/sentiment_tfidf_vectorizer.pkl"
        
        # Load BERT model
        bert_model, bert_tokenizer = load_bert_model("bert-base-uncased")
        
        # Load Fine-tuned BERT model
        ft_bert_model, ft_bert_tokenizer = load_bert_model(ft_bert_model_dir)
        
        # Load LR model
        lr_model, lr_vectorizer = load_lr_model(lr_model_path, lr_vectorizer_path)
        
        # Process tweets with each model
        tweets = df[text_column].tolist()
        
        # Create separate dataframes for each model's predictions
        result_dfs = {}
        
        # Process with BERT model
        if bert_model and bert_tokenizer:
            print(f"Processing {text_column}s with BERT model...")
            bert_scores = analyze_sentiment_bert(tweets, bert_model, bert_tokenizer, device)
            
            # Create dataframe with results
            bert_df = df.copy()
            bert_df['sentiment'] = bert_scores
            
            # Save to CSV
            bert_output_path = os.path.join(output_dir, 'bert_results.csv')
            bert_df.to_csv(bert_output_path, index=False)
            print(f"BERT results saved to: {bert_output_path}")
            
            # Plot and save visualizations
            plot_model_analysis(bert_df, 'Pretrained BERT', output_dir, text_column)
            result_dfs['Pretrained BERT'] = bert_df
        
        # Process with Fine-tuned BERT model
        if ft_bert_model and ft_bert_tokenizer:
            print(f"Processing {text_column}s with Fine-tuned BERT model...")
            ft_bert_scores = analyze_sentiment_bert(tweets, ft_bert_model, ft_bert_tokenizer, device)
            
            # Create dataframe with results
            ft_bert_df = df.copy()
            ft_bert_df['sentiment'] = ft_bert_scores
            
            # Save to CSV
            ft_bert_output_path = os.path.join(output_dir, 'finetuned_bert_results.csv')
            ft_bert_df.to_csv(ft_bert_output_path, index=False)
            print(f"Fine-tuned BERT results saved to: {ft_bert_output_path}")
            
            # Plot and save visualizations
            plot_model_analysis(ft_bert_df, 'Fine-tuned BERT', output_dir, text_column)
            
            result_dfs['Fine-tuned BERT'] = ft_bert_df
        
        # Process with LR model
        if lr_model and lr_vectorizer:
            print(f"Processing {text_column}s with Logistic Regression model...")
            lr_scores = analyze_sentiment_lr(tweets, lr_model, lr_vectorizer)
            
            # Create dataframe with results
            lr_df = df.copy()
            lr_df['sentiment'] = lr_scores
            
            # Save to CSV
            lr_output_path = os.path.join(output_dir, 'lr_results.csv')
            lr_df.to_csv(lr_output_path, index=False)
            print(f"Logistic Regression results saved to: {lr_output_path}")
            
            # Plot and save visualizations
            plot_model_analysis(lr_df, 'Logistic Regression', output_dir, text_column)
            
            result_dfs['Logistic Regression'] = lr_df
        
        # Create model comparison visualization if multiple models were used
        if len(result_dfs) > 1:
            create_model_comparison(result_dfs, output_dir, text_column)
        
        print(f"Analysis complete. Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()