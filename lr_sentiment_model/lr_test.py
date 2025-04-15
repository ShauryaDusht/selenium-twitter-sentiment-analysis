import pickle
import numpy as np

def load_model(model_path, vectorizer_path):
    """Load the sentiment analysis model and vectorizer from disk."""
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return model, vectorizer

def analyze_sentiment(tweet, model, vectorizer):
    """Analyze the sentiment of a tweet and return the prediction with confidence."""
    # Transform the tweet using the vectorizer
    tweet_vectorized = vectorizer.transform([tweet])
    
    # Get prediction
    prediction = model.predict(tweet_vectorized)[0]
    
    # Get confidence (probability)
    confidence = np.max(model.predict_proba(tweet_vectorized)[0]) * 100
    
    # Map prediction to sentiment label
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence

def main():
    # Paths to your model and vectorizer
    model_path = r"C:\Users\shaur\OneDrive\Desktop\projects\selenium-twitter-sentiment-analysis\lr_sentiment_model\sentiment_lr_model.pkl"
    vectorizer_path = r"C:\Users\shaur\OneDrive\Desktop\projects\selenium-twitter-sentiment-analysis\lr_sentiment_model\sentiment_tfidf_vectorizer.pkl"
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer = load_model(model_path, vectorizer_path)
    print("Model loaded successfully!")
    
    while True:
        # Get tweet from user
        tweet = input("\nEnter a tweet to analyze (or 'q' to quit): ")
        
        if tweet.lower() == 'q':
            break
        
        if not tweet.strip():
            print("Please enter a valid tweet.")
            continue
            
        # Analyze sentiment
        sentiment, confidence = analyze_sentiment(tweet, model, vectorizer)
        
        # Display result
        print(f"\nSentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()