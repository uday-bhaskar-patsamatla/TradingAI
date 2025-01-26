import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

class StockSentimentAnalyzer:
    def __init__(self, ticker):
        """
        Initialize the Stock Sentiment Analyzer with a specific stock ticker
        
        :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker
        self.stock_data = None
        self.news_sentiment = None
        
    def fetch_stock_data(self, period='1mo'):
        """
        Fetch historical stock data
        
        :param period: Time period for data retrieval (default: 1 month)
        :return: Pandas DataFrame with stock information
        """
        try:
            self.stock_data = yf.Ticker(self.ticker).history(period=period)
            return self.stock_data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def fetch_news_sentiment(self):
        """
        Fetch recent news and analyze sentiment
        
        :return: Average sentiment score
        """
        try:
            # Using a simple news API (replace with actual API key)
            url = f'https://newsapi.org/v2/everything?q={self.ticker}&apiKey=0e17c5ffb27d4e51a5e9416b09378f3a'
            response = requests.get(url)
            news_data = response.json()
            
            sentiments = []
            for article in news_data.get('articles', []):
                # Use TextBlob for sentiment analysis
                if article.get('description', '') != None:
                    blob = TextBlob(article['title'] + ' ' + article.get('description', ''))
                    sentiments.append(blob.sentiment.polarity)
            
            # Calculate average sentiment
            self.news_sentiment = np.mean(sentiments) if sentiments else 0
            return self.news_sentiment
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return 0
    
    def generate_recommendation(self):
        """
        Generate stock recommendation based on price trends and news sentiment
        
        :return: Recommendation dictionary
        """
        if self.stock_data is None or len(self.stock_data) == 0:
            return {"error": "No stock data available"}
        
        # Calculate basic price trends
        closing_prices = self.stock_data['Close']
        
        price_change = ((closing_prices.iloc[-1] - closing_prices.iloc[0]) / closing_prices.iloc[0]) * 100
        
        # Sentiment analysis
        news_sentiment = self.fetch_news_sentiment()
        
        # Recommendation logic
        recommendation = {
            'ticker': self.ticker,
            'current_price': closing_prices.iloc[-1],
            'price_change_percentage': price_change,
            'news_sentiment': news_sentiment
        }
        
        # Simple recommendation logic
        if price_change > 0 and news_sentiment > 0:
            recommendation['action'] = 'Strong Buy'
            recommendation['confidence'] = 'High'
        elif price_change > 0 and news_sentiment > -0.5:
            recommendation['action'] = 'Buy'
            recommendation['confidence'] = 'Medium'
        elif price_change < 0 and news_sentiment < 0:
            recommendation['action'] = 'Sell'
            recommendation['confidence'] = 'High'
        else:
            recommendation['action'] = 'Hold'
            recommendation['confidence'] = 'Low'
        
        return recommendation

def main():
    # Example usage
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    for ticker in tickers:
        # Create analyzer for each stock
        analyzer = StockSentimentAnalyzer(ticker)
        
        # Fetch stock data
        price = analyzer.fetch_stock_data()
        # print(price)
        
        # Generate recommendation
        recommendation = analyzer.generate_recommendation()
        
        # Print results
        print("\n--- Stock Recommendation ---")
        for key, value in recommendation.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()