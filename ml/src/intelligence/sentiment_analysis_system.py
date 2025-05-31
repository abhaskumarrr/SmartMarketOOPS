#!/usr/bin/env python3
"""
Real-time Sentiment Analysis System for Enhanced SmartMarketOOPS
Integrates news, social media, and market sentiment for trading decisions
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import json
from collections import deque
import feedparser
import tweepy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """Analyzes financial news for sentiment and market impact"""
    
    def __init__(self):
        """Initialize news analyzer"""
        self.news_sources = {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'bitcoin_magazine': 'https://bitcoinmagazine.com/.rss/full/',
            'decrypt': 'https://decrypt.co/feed',
            'the_block': 'https://www.theblockcrypto.com/rss.xml'
        }
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            logger.info("FinBERT sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT, using TextBlob: {e}")
            self.sentiment_analyzer = None
        
        self.news_cache = deque(maxlen=1000)
        self.sentiment_history = deque(maxlen=500)
        
        # Keywords for different cryptocurrencies
        self.crypto_keywords = {
            'BTCUSDT': ['bitcoin', 'btc', 'bitcoin price', 'bitcoin market'],
            'ETHUSDT': ['ethereum', 'eth', 'ethereum price', 'smart contract'],
            'SOLUSDT': ['solana', 'sol', 'solana price', 'solana network'],
            'ADAUSDT': ['cardano', 'ada', 'cardano price', 'cardano network']
        }
        
        logger.info("News Analyzer initialized")
    
    async def fetch_news_feeds(self) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds"""
        news_articles = []
        
        for source, url in self.news_sources.items():
            try:
                # Parse RSS feed
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:10]:  # Latest 10 articles per source
                    article = {
                        'source': source,
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'timestamp': datetime.now()
                    }
                    news_articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {e}")
        
        logger.info(f"Fetched {len(news_articles)} news articles")
        return news_articles
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if not text:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        try:
            if self.sentiment_analyzer:
                # Use FinBERT for financial sentiment
                result = self.sentiment_analyzer(text[:512])  # Truncate to model limit
                
                sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
                sentiment_score = sentiment_map.get(result[0]['label'].lower(), 0.0)
                confidence = result[0]['score']
                
                return {'sentiment': sentiment_score, 'confidence': confidence}
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity  # -1 to 1
                confidence = abs(sentiment_score)
                
                return {'sentiment': sentiment_score, 'confidence': confidence}
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    def extract_crypto_relevance(self, text: str, symbol: str) -> float:
        """Extract relevance score for specific cryptocurrency"""
        text_lower = text.lower()
        keywords = self.crypto_keywords.get(symbol, [])
        
        relevance_score = 0.0
        for keyword in keywords:
            if keyword in text_lower:
                relevance_score += 1.0
        
        # Normalize by number of keywords
        if keywords:
            relevance_score /= len(keywords)
        
        return min(relevance_score, 1.0)
    
    async def analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze news sentiment for a specific symbol"""
        # Fetch latest news
        news_articles = await self.fetch_news_feeds()
        
        relevant_articles = []
        sentiment_scores = []
        
        for article in news_articles:
            # Check relevance to symbol
            title_relevance = self.extract_crypto_relevance(article['title'], symbol)
            summary_relevance = self.extract_crypto_relevance(article['summary'], symbol)
            
            overall_relevance = max(title_relevance, summary_relevance)
            
            if overall_relevance > 0.1:  # Minimum relevance threshold
                # Analyze sentiment
                title_sentiment = self.analyze_text_sentiment(article['title'])
                summary_sentiment = self.analyze_text_sentiment(article['summary'])
                
                # Combine sentiments
                combined_sentiment = (title_sentiment['sentiment'] + summary_sentiment['sentiment']) / 2
                combined_confidence = (title_sentiment['confidence'] + summary_sentiment['confidence']) / 2
                
                article_analysis = {
                    'article': article,
                    'relevance': overall_relevance,
                    'sentiment': combined_sentiment,
                    'confidence': combined_confidence,
                    'weighted_sentiment': combined_sentiment * overall_relevance * combined_confidence
                }
                
                relevant_articles.append(article_analysis)
                sentiment_scores.append(article_analysis['weighted_sentiment'])
        
        # Calculate overall sentiment
        if sentiment_scores:
            overall_sentiment = np.mean(sentiment_scores)
            sentiment_strength = np.std(sentiment_scores)
            article_count = len(relevant_articles)
        else:
            overall_sentiment = 0.0
            sentiment_strength = 0.0
            article_count = 0
        
        # Store in cache
        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'overall_sentiment': overall_sentiment,
            'sentiment_strength': sentiment_strength,
            'article_count': article_count,
            'relevant_articles': relevant_articles[:5]  # Keep top 5
        }
        
        self.sentiment_history.append(sentiment_data)
        
        return sentiment_data


class SocialMediaAnalyzer:
    """Analyzes social media sentiment (Twitter, Reddit, etc.)"""
    
    def __init__(self, twitter_api_key: str = None, twitter_api_secret: str = None,
                 twitter_access_token: str = None, twitter_access_token_secret: str = None):
        """Initialize social media analyzer"""
        
        # Twitter API setup (optional)
        self.twitter_api = None
        if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret]):
            try:
                auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
                auth.set_access_token(twitter_access_token, twitter_access_token_secret)
                self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("Twitter API initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter API: {e}")
        
        # Sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except:
            self.sentiment_analyzer = None
        
        self.social_sentiment_history = deque(maxlen=500)
        
        # Social media keywords
        self.social_keywords = {
            'BTCUSDT': ['#bitcoin', '#btc', '$btc', 'bitcoin'],
            'ETHUSDT': ['#ethereum', '#eth', '$eth', 'ethereum'],
            'SOLUSDT': ['#solana', '#sol', '$sol', 'solana'],
            'ADAUSDT': ['#cardano', '#ada', '$ada', 'cardano']
        }
        
        logger.info("Social Media Analyzer initialized")
    
    async def fetch_twitter_sentiment(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """Fetch Twitter sentiment for a symbol"""
        if not self.twitter_api:
            return []
        
        keywords = self.social_keywords.get(symbol, [])
        if not keywords:
            return []
        
        tweets = []
        
        try:
            # Search for tweets
            query = ' OR '.join(keywords)
            search_results = tweepy.Cursor(
                self.twitter_api.search_tweets,
                q=query,
                lang='en',
                result_type='recent',
                tweet_mode='extended'
            ).items(count)
            
            for tweet in search_results:
                # Analyze sentiment
                sentiment_result = self.analyze_social_sentiment(tweet.full_text)
                
                tweet_data = {
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user_followers': tweet.user.followers_count,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence']
                }
                
                tweets.append(tweet_data)
                
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
        
        return tweets
    
    def analyze_social_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of social media text"""
        if not text:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text[:512])
                
                sentiment_map = {'POSITIVE': 1.0, 'NEGATIVE': -1.0, 'NEUTRAL': 0.0}
                sentiment_score = sentiment_map.get(result[0]['label'], 0.0)
                confidence = result[0]['score']
                
                return {'sentiment': sentiment_score, 'confidence': confidence}
            else:
                # Fallback to simple keyword analysis
                positive_words = ['bullish', 'moon', 'pump', 'buy', 'hodl', 'green', 'up']
                negative_words = ['bearish', 'dump', 'sell', 'crash', 'red', 'down', 'fear']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count + negative_count == 0:
                    return {'sentiment': 0.0, 'confidence': 0.0}
                
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                confidence = min((positive_count + negative_count) / 10, 1.0)
                
                return {'sentiment': sentiment, 'confidence': confidence}
                
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    async def analyze_social_media_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze overall social media sentiment for a symbol"""
        
        # Fetch Twitter data
        tweets = await self.fetch_twitter_sentiment(symbol)
        
        if not tweets:
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'tweet_count': 0,
                'engagement_score': 0.0
            }
        
        # Calculate weighted sentiment based on engagement
        weighted_sentiments = []
        total_engagement = 0
        
        for tweet in tweets:
            # Calculate engagement weight
            engagement = (
                tweet['retweet_count'] * 2 +
                tweet['favorite_count'] +
                np.log10(tweet['user_followers'] + 1)
            )
            
            weighted_sentiment = tweet['sentiment'] * tweet['confidence'] * (1 + engagement / 100)
            weighted_sentiments.append(weighted_sentiment)
            total_engagement += engagement
        
        # Calculate overall metrics
        overall_sentiment = np.mean(weighted_sentiments) if weighted_sentiments else 0.0
        sentiment_strength = np.std(weighted_sentiments) if len(weighted_sentiments) > 1 else 0.0
        engagement_score = total_engagement / len(tweets) if tweets else 0.0
        
        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'overall_sentiment': overall_sentiment,
            'sentiment_strength': sentiment_strength,
            'tweet_count': len(tweets),
            'engagement_score': engagement_score,
            'sample_tweets': tweets[:5]  # Keep top 5 tweets
        }
        
        self.social_sentiment_history.append(sentiment_data)
        
        return sentiment_data


class MarketSentimentAggregator:
    """Aggregates sentiment from multiple sources"""
    
    def __init__(self):
        """Initialize market sentiment aggregator"""
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialMediaAnalyzer()
        
        self.sentiment_weights = {
            'news': 0.6,
            'social': 0.4
        }
        
        self.aggregated_sentiment_history = deque(maxlen=500)
        
        logger.info("Market Sentiment Aggregator initialized")
    
    async def get_comprehensive_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis for a symbol"""
        
        # Fetch sentiment from all sources
        news_sentiment = await self.news_analyzer.analyze_news_sentiment(symbol)
        social_sentiment = await self.social_analyzer.analyze_social_media_sentiment(symbol)
        
        # Calculate weighted aggregate sentiment
        news_weight = self.sentiment_weights['news']
        social_weight = self.sentiment_weights['social']
        
        # Adjust weights based on data availability
        if news_sentiment['article_count'] == 0:
            news_weight = 0.0
            social_weight = 1.0
        elif social_sentiment['tweet_count'] == 0:
            news_weight = 1.0
            social_weight = 0.0
        
        # Normalize weights
        total_weight = news_weight + social_weight
        if total_weight > 0:
            news_weight /= total_weight
            social_weight /= total_weight
        
        # Calculate aggregate sentiment
        aggregate_sentiment = (
            news_sentiment['overall_sentiment'] * news_weight +
            social_sentiment['overall_sentiment'] * social_weight
        )
        
        # Calculate sentiment confidence
        sentiment_confidence = (
            abs(news_sentiment['overall_sentiment']) * news_weight +
            abs(social_sentiment['overall_sentiment']) * social_weight
        )
        
        # Calculate sentiment momentum (change over time)
        sentiment_momentum = self.calculate_sentiment_momentum(symbol, aggregate_sentiment)
        
        # Create comprehensive sentiment data
        comprehensive_sentiment = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'aggregate_sentiment': aggregate_sentiment,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_momentum': sentiment_momentum,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'sentiment_signal': self.generate_sentiment_signal(
                aggregate_sentiment, sentiment_confidence, sentiment_momentum
            )
        }
        
        self.aggregated_sentiment_history.append(comprehensive_sentiment)
        
        return comprehensive_sentiment
    
    def calculate_sentiment_momentum(self, symbol: str, current_sentiment: float) -> float:
        """Calculate sentiment momentum"""
        # Get recent sentiment history for this symbol
        recent_sentiments = [
            data['aggregate_sentiment'] for data in self.aggregated_sentiment_history
            if data['symbol'] == symbol and 
            (datetime.now() - data['timestamp']).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_sentiments) < 2:
            return 0.0
        
        # Calculate momentum as rate of change
        momentum = current_sentiment - np.mean(recent_sentiments[:-1])
        return momentum
    
    def generate_sentiment_signal(self, sentiment: float, confidence: float, momentum: float) -> Dict[str, Any]:
        """Generate trading signal based on sentiment analysis"""
        
        # Sentiment thresholds
        strong_positive_threshold = 0.3
        weak_positive_threshold = 0.1
        weak_negative_threshold = -0.1
        strong_negative_threshold = -0.3
        
        # Confidence threshold
        min_confidence = 0.3
        
        # Generate signal
        if confidence < min_confidence:
            signal = 'NEUTRAL'
            strength = 0.0
        elif sentiment > strong_positive_threshold:
            signal = 'STRONG_BUY'
            strength = min(sentiment * confidence, 1.0)
        elif sentiment > weak_positive_threshold:
            signal = 'BUY'
            strength = sentiment * confidence * 0.7
        elif sentiment < strong_negative_threshold:
            signal = 'STRONG_SELL'
            strength = min(abs(sentiment) * confidence, 1.0)
        elif sentiment < weak_negative_threshold:
            signal = 'SELL'
            strength = abs(sentiment) * confidence * 0.7
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        # Adjust for momentum
        if momentum > 0.1 and signal in ['BUY', 'STRONG_BUY']:
            strength *= 1.2  # Boost bullish signals with positive momentum
        elif momentum < -0.1 and signal in ['SELL', 'STRONG_SELL']:
            strength *= 1.2  # Boost bearish signals with negative momentum
        
        return {
            'signal': signal,
            'strength': min(strength, 1.0),
            'sentiment_score': sentiment,
            'confidence_score': confidence,
            'momentum_score': momentum
        }
    
    def get_sentiment_summary(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_data = [
            data for data in self.aggregated_sentiment_history
            if data['symbol'] == symbol and data['timestamp'] > cutoff_time
        ]
        
        if not recent_data:
            return {
                'symbol': symbol,
                'period_hours': hours,
                'average_sentiment': 0.0,
                'sentiment_trend': 'NEUTRAL',
                'data_points': 0
            }
        
        sentiments = [data['aggregate_sentiment'] for data in recent_data]
        
        return {
            'symbol': symbol,
            'period_hours': hours,
            'average_sentiment': np.mean(sentiments),
            'sentiment_volatility': np.std(sentiments),
            'sentiment_trend': 'BULLISH' if sentiments[-1] > sentiments[0] else 'BEARISH',
            'data_points': len(recent_data),
            'latest_signal': recent_data[-1]['sentiment_signal'] if recent_data else None
        }


async def main():
    """Test sentiment analysis system"""
    aggregator = MarketSentimentAggregator()
    
    # Test sentiment analysis for Bitcoin
    sentiment_data = await aggregator.get_comprehensive_sentiment('BTCUSDT')
    
    print("Comprehensive Sentiment Analysis:")
    print(f"Symbol: {sentiment_data['symbol']}")
    print(f"Aggregate Sentiment: {sentiment_data['aggregate_sentiment']:.3f}")
    print(f"Confidence: {sentiment_data['sentiment_confidence']:.3f}")
    print(f"Momentum: {sentiment_data['sentiment_momentum']:.3f}")
    print(f"Signal: {sentiment_data['sentiment_signal']['signal']}")
    print(f"Signal Strength: {sentiment_data['sentiment_signal']['strength']:.3f}")
    
    # Get sentiment summary
    summary = aggregator.get_sentiment_summary('BTCUSDT', hours=24)
    print(f"\n24-hour Sentiment Summary:")
    print(f"Average Sentiment: {summary['average_sentiment']:.3f}")
    print(f"Trend: {summary['sentiment_trend']}")
    print(f"Data Points: {summary['data_points']}")


if __name__ == "__main__":
    asyncio.run(main())
