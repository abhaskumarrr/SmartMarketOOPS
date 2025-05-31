#!/usr/bin/env python3
"""
Automated Feature Engineering Pipeline for Enhanced SmartMarketOOPS
Discovers and creates new features automatically using genetic algorithms and ML
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Callable
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Generates new features using various mathematical operations"""
    
    def __init__(self):
        """Initialize feature generator"""
        self.generated_features = []
        self.feature_importance_scores = {}
        
    def generate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive technical analysis features"""
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5) / df[f'sma_{period}'].shift(5)
        
        # Volatility features
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['price_change'].rolling(window=period).std()
            df[f'volatility_{period}_norm'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(100).mean()
        
        # Momentum indicators
        for period in [14, 21, 50]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # RSI variations
        for period in [14, 21, 50]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            df[f'macd_histogram_{fast}_{slow}_{signal}'] = macd - macd_signal
        
        # Bollinger Bands variations
        for period, std_dev in [(20, 2), (20, 1), (50, 2)]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}_{std_dev}'] = sma + (std * std_dev)
            df[f'bb_lower_{period}_{std_dev}'] = sma - (std * std_dev)
            df[f'bb_width_{period}_{std_dev}'] = df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}']
            df[f'bb_position_{period}_{std_dev}'] = (df['close'] - df[f'bb_lower_{period}_{std_dev}']) / df[f'bb_width_{period}_{std_dev}']
        
        return df
    
    def generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features"""
        df = data.copy()
        
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume price trend
        df['vpt'] = (df['volume'] * df['price_change']).cumsum()
        
        # On-balance volume
        df['obv'] = (df['volume'] * np.sign(df['price_change'])).cumsum()
        
        # Volume weighted average price
        for period in [10, 20]:
            df[f'vwap_{period}'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
            df[f'price_vwap_{period}_ratio'] = df['close'] / df[f'vwap_{period}']
        
        # Accumulation/Distribution line
        df['ad_line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['ad_line'] = df['ad_line'].cumsum()
        
        return df
    
    def generate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        df = data.copy()
        
        # Rolling statistics
        for period in [10, 20, 50]:
            df[f'skewness_{period}'] = df['price_change'].rolling(window=period).skew()
            df[f'kurtosis_{period}'] = df['price_change'].rolling(window=period).kurt()
            df[f'quantile_25_{period}'] = df['close'].rolling(window=period).quantile(0.25)
            df[f'quantile_75_{period}'] = df['close'].rolling(window=period).quantile(0.75)
            df[f'iqr_{period}'] = df[f'quantile_75_{period}'] - df[f'quantile_25_{period}']
        
        # Z-scores
        for period in [20, 50]:
            rolling_mean = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            df[f'zscore_{period}'] = (df['close'] - rolling_mean) / rolling_std
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = df['price_change'].rolling(window=50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        return df
    
    def generate_fractal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fractal and pattern-based features"""
        df = data.copy()
        
        # Fractal dimension (simplified Hurst exponent)
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent"""
            if len(ts) < max_lag * 2:
                return 0.5
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        df['hurst_exponent'] = df['close'].rolling(window=100).apply(hurst_exponent)
        
        # Support and resistance levels
        def find_support_resistance(prices, window=20):
            """Find support and resistance levels"""
            if len(prices) < window:
                return 0, 0
            
            highs = prices.rolling(window=window).max()
            lows = prices.rolling(window=window).min()
            
            resistance = highs.iloc[-1]
            support = lows.iloc[-1]
            
            return support, resistance
        
        support_resistance = df['close'].rolling(window=50).apply(
            lambda x: find_support_resistance(x)[0] if len(x) >= 50 else x.iloc[-1]
        )
        df['support_level'] = support_resistance
        df['resistance_level'] = df['close'].rolling(window=50).apply(
            lambda x: find_support_resistance(x)[1] if len(x) >= 50 else x.iloc[-1]
        )
        
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between existing features"""
        df = data.copy()
        
        # Price and volume interactions
        if 'volume' in df.columns:
            df['price_volume_trend'] = df['price_change'] * df['volume']
            df['price_volume_correlation'] = df['price_change'].rolling(window=20).corr(df['volume'])
        
        # RSI and price interactions
        if 'rsi_14' in df.columns:
            df['rsi_price_divergence'] = df['rsi_14'].diff() - (df['price_change'] * 100)
            df['rsi_momentum'] = df['rsi_14'].diff()
        
        # Moving average interactions
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_cross_signal'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['ma_distance'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Volatility and momentum interactions
        if 'volatility_20' in df.columns and 'momentum_14' in df.columns:
            df['vol_momentum_ratio'] = df['volatility_20'] / (abs(df['momentum_14']) + 1e-8)
        
        return df


class GeneticFeatureSelector:
    """Genetic algorithm for feature selection and creation"""
    
    def __init__(self, population_size: int = 50, generations: int = 20, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """Initialize genetic feature selector"""
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Mathematical operations for feature creation
        self.operations = [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / (y + 1e-8),
            lambda x: np.log(abs(x) + 1e-8),
            lambda x: np.sqrt(abs(x)),
            lambda x: x ** 2,
            lambda x: np.tanh(x),
            lambda x: np.sin(x),
            lambda x: np.cos(x)
        ]
        
        self.best_features = []
        self.best_fitness = 0.0
    
    def create_random_feature(self, data: pd.DataFrame) -> pd.Series:
        """Create a random feature using genetic operations"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            return data[numeric_columns[0]] if numeric_columns else pd.Series([0] * len(data))
        
        # Choose operation type
        operation_type = np.random.choice(['unary', 'binary'], p=[0.3, 0.7])
        
        if operation_type == 'unary':
            # Single feature operation
            col = np.random.choice(numeric_columns)
            op_idx = np.random.choice([4, 5, 6, 7, 8, 9])  # Unary operations
            operation = self.operations[op_idx]
            
            try:
                result = operation(data[col])
                return pd.Series(result, index=data.index).fillna(0)
            except:
                return pd.Series([0] * len(data), index=data.index)
        
        else:
            # Binary feature operation
            col1, col2 = np.random.choice(numeric_columns, 2, replace=False)
            op_idx = np.random.choice([0, 1, 2, 3])  # Binary operations
            operation = self.operations[op_idx]
            
            try:
                result = operation(data[col1], data[col2])
                return pd.Series(result, index=data.index).fillna(0)
            except:
                return pd.Series([0] * len(data), index=data.index)
    
    def evaluate_feature_fitness(self, feature: pd.Series, target: pd.Series) -> float:
        """Evaluate feature fitness using correlation and mutual information"""
        try:
            # Remove NaN values
            valid_mask = ~(feature.isna() | target.isna() | np.isinf(feature))
            if valid_mask.sum() < 10:
                return 0.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]
            
            # Correlation-based fitness
            correlation = abs(np.corrcoef(feature_clean, target_clean)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
            
            # Mutual information fitness
            try:
                mi_score = mutual_info_regression(
                    feature_clean.values.reshape(-1, 1), 
                    target_clean.values
                )[0]
            except:
                mi_score = 0.0
            
            # Combined fitness
            fitness = (correlation * 0.7) + (mi_score * 0.3)
            
            return fitness
            
        except Exception as e:
            return 0.0
    
    def evolve_features(self, data: pd.DataFrame, target: pd.Series, 
                       n_features: int = 10) -> List[Tuple[pd.Series, float]]:
        """Evolve features using genetic algorithm"""
        logger.info(f"Starting genetic feature evolution for {n_features} features")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            feature = self.create_random_feature(data)
            fitness = self.evaluate_feature_fitness(feature, target)
            population.append((feature, fitness))
        
        # Evolution loop
        for generation in range(self.generations):
            # Sort by fitness
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Track best fitness
            current_best = population[0][1]
            if current_best > self.best_fitness:
                self.best_fitness = current_best
            
            # Selection (top 50%)
            selected = population[:self.population_size // 2]
            
            # Create new generation
            new_population = selected.copy()
            
            while len(new_population) < self.population_size:
                # Crossover
                if np.random.random() < self.crossover_rate and len(selected) >= 2:
                    parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
                    
                    # Simple crossover: average of parents
                    try:
                        child_feature = (selected[parent1][0] + selected[parent2][0]) / 2
                        child_fitness = self.evaluate_feature_fitness(child_feature, target)
                        new_population.append((child_feature, child_fitness))
                    except:
                        # If crossover fails, create random feature
                        feature = self.create_random_feature(data)
                        fitness = self.evaluate_feature_fitness(feature, target)
                        new_population.append((feature, fitness))
                else:
                    # Mutation: create new random feature
                    feature = self.create_random_feature(data)
                    fitness = self.evaluate_feature_fitness(feature, target)
                    new_population.append((feature, fitness))
            
            population = new_population
            
            if generation % 5 == 0:
                logger.info(f"Generation {generation}: Best fitness = {current_best:.4f}")
        
        # Return top features
        population.sort(key=lambda x: x[1], reverse=True)
        return population[:n_features]


class AutomatedFeatureEngineer:
    """Complete automated feature engineering system"""
    
    def __init__(self, max_features: int = 100):
        """Initialize automated feature engineer"""
        self.max_features = max_features
        self.feature_generator = FeatureGenerator()
        self.genetic_selector = GeneticFeatureSelector()
        self.feature_selector = SelectKBest(score_func=f_regression, k=max_features)
        self.scaler = StandardScaler()
        
        self.selected_features = []
        self.feature_importance = {}
        self.is_fitted = False
        
        logger.info(f"Automated Feature Engineer initialized (max_features={max_features})")
    
    def engineer_features(self, data: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Engineer comprehensive feature set"""
        logger.info("Starting automated feature engineering...")
        
        # Start with original data
        engineered_data = data.copy()
        
        # Generate technical features
        logger.info("Generating technical features...")
        engineered_data = self.feature_generator.generate_technical_features(engineered_data)
        
        # Generate volume features
        logger.info("Generating volume features...")
        engineered_data = self.feature_generator.generate_volume_features(engineered_data)
        
        # Generate statistical features
        logger.info("Generating statistical features...")
        engineered_data = self.feature_generator.generate_statistical_features(engineered_data)
        
        # Generate fractal features
        logger.info("Generating fractal features...")
        engineered_data = self.feature_generator.generate_fractal_features(engineered_data)
        
        # Generate interaction features
        logger.info("Generating interaction features...")
        engineered_data = self.feature_generator.generate_interaction_features(engineered_data)
        
        # Genetic feature creation (if target is provided)
        if target is not None:
            logger.info("Creating genetic features...")
            genetic_features = self.genetic_selector.evolve_features(
                engineered_data, target, n_features=20
            )
            
            for i, (feature, fitness) in enumerate(genetic_features):
                feature_name = f'genetic_feature_{i}'
                engineered_data[feature_name] = feature
                self.feature_importance[feature_name] = fitness
        
        # Remove infinite and NaN values
        engineered_data = engineered_data.replace([np.inf, -np.inf], np.nan)
        engineered_data = engineered_data.fillna(method='ffill').fillna(0)
        
        logger.info(f"Feature engineering completed. Generated {len(engineered_data.columns)} features")
        
        return engineered_data
    
    def select_best_features(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select best features using multiple selection methods"""
        logger.info("Selecting best features...")
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            logger.warning("No numeric features found")
            return data
        
        # Remove features with zero variance
        variance_threshold = 1e-8
        feature_variances = numeric_data.var()
        valid_features = feature_variances[feature_variances > variance_threshold].index.tolist()
        numeric_data = numeric_data[valid_features]
        
        if len(numeric_data.columns) == 0:
            logger.warning("No features with sufficient variance")
            return data
        
        # Align data and target
        common_index = numeric_data.index.intersection(target.index)
        numeric_data = numeric_data.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove rows with NaN in target
        valid_mask = ~target_aligned.isna()
        numeric_data = numeric_data[valid_mask]
        target_aligned = target_aligned[valid_mask]
        
        if len(numeric_data) < 10:
            logger.warning("Insufficient data for feature selection")
            return data
        
        # Statistical feature selection
        k_best = min(self.max_features, len(numeric_data.columns))
        self.feature_selector.k = k_best
        
        try:
            selected_features_array = self.feature_selector.fit_transform(numeric_data, target_aligned)
            selected_feature_names = numeric_data.columns[self.feature_selector.get_support()].tolist()
            
            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(selected_features_array, target_aligned)
            
            # Store feature importance
            for i, feature_name in enumerate(selected_feature_names):
                self.feature_importance[feature_name] = rf.feature_importances_[i]
            
            self.selected_features = selected_feature_names
            self.is_fitted = True
            
            logger.info(f"Selected {len(selected_feature_names)} best features")
            
            # Return data with selected features
            result_data = data.copy()
            for col in data.columns:
                if col not in selected_feature_names and col in numeric_data.columns:
                    result_data = result_data.drop(columns=[col])
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return data
    
    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Fit feature engineer and transform data"""
        # Engineer features
        engineered_data = self.engineer_features(data, target)
        
        # Select best features
        selected_data = self.select_best_features(engineered_data, target)
        
        return selected_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature engineer"""
        if not self.is_fitted:
            logger.warning("Feature engineer not fitted. Using original data.")
            return data
        
        # Engineer features (without target)
        engineered_data = self.engineer_features(data)
        
        # Keep only selected features
        result_data = data.copy()
        for col in self.selected_features:
            if col in engineered_data.columns:
                result_data[col] = engineered_data[col]
        
        # Remove non-selected features
        for col in result_data.columns:
            if col not in self.selected_features and col not in data.columns:
                result_data = result_data.drop(columns=[col])
        
        return result_data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by importance"""
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_features[:n]


def main():
    """Test automated feature engineering"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    prices = 45000 + np.cumsum(np.random.randn(1000) * 100)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices + np.random.randn(1000) * 50,
        'volume': np.random.lognormal(15, 1, 1000)
    })
    
    # Create target (future price movement)
    target = (data['close'].shift(-1) > data['close']).astype(int)
    target = target.dropna()
    
    # Initialize feature engineer
    feature_engineer = AutomatedFeatureEngineer(max_features=50)
    
    # Fit and transform
    engineered_data = feature_engineer.fit_transform(data[:-1], target)
    
    print(f"Original features: {len(data.columns)}")
    print(f"Engineered features: {len(engineered_data.columns)}")
    
    # Show top features
    top_features = feature_engineer.get_top_features(10)
    print("\nTop 10 features:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
