from flask import Flask, render_template, request, jsonify
from enhanced_main import AdvancedStockTradingAI
import yfinance as yf
import logging
import plotly.graph_objects as go

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self):
        self.trading_ai = None
    
    def initialize_ai(self, ticker, capital):
        """Initialize the trading AI with a single ticker and capital"""
        self.trading_ai = AdvancedStockTradingAI([ticker], trading_capital=capital)
        
    def get_stock_info(self, ticker):
        """Get basic stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching stock info: {e}")
            return None

    def create_price_chart(self, ticker):
        """Create an interactive price chart"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
            
            fig.update_layout(
                title=f'{ticker} Stock Price',
                yaxis_title='Price',
                xaxis_title='Date'
            )
            
            return fig.to_html(full_html=False)
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None

analyzer = StockAnalyzer()

@app.route('/')
def home():
    """Home page with stock ticker input form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the stock and return results including trade execution"""
    try:
        ticker = request.form['ticker'].upper()
        capital = float(request.form['capital'])
        
        # Initialize AI with the ticker and capital
        analyzer.initialize_ai(ticker, capital)
        
        # Get stock information
        stock_info = analyzer.get_stock_info(ticker)
        if not stock_info:
            return jsonify({'error': 'Failed to fetch stock information'})
        
        # Execute trades and get formatted string result
        trade_results = analyzer.trading_ai.execute_trades()
        
        # Create price chart
        price_chart = analyzer.create_price_chart(ticker)
        
        # Get social sentiment
        sentiment = analyzer.trading_ai._analyze_social_sentiment(ticker)
        
        return render_template(
            'result.html',
            ticker=ticker,
            stock_info=stock_info,
            trade_results=trade_results,  # Now passing the formatted string
            sentiment=sentiment,
            price_chart=price_chart,
            initial_capital=capital
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True)