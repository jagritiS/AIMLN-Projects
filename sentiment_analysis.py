# sentiment_analysis.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    # Initialize VADER Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Get the sentiment score
    sentiment_score = analyzer.polarity_scores(text)
    
    # Determine sentiment based on the compound score
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, sentiment_score

def main():
    print("Sentiment Analysis Tool (VADER)\n")
    
    # User input for text
    text = input("Enter a sentence or text to analyze sentiment: ")
    
    # Analyze the sentiment of the input text
    sentiment, sentiment_score = analyze_sentiment(text)
    
    # Output the result
    print(f"\nSentiment: {sentiment}")
    print(f"Sentiment Scores: {sentiment_score}")

if __name__ == "__main__":
    main()
