import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "macro_indicators.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MacroIndicators:
    """Class for generating macroeconomic indicators."""
    
    def __init__(self, data_dir=None):
        """
        Initialize MacroIndicators class.
        
        Parameters
        ----------
        data_dir : str or Path, optional
            Directory containing the data files.
        """
        if data_dir is None:
            # Default to the standard project structure
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        logger.info(f"MacroIndicators initialized with data directory: {self.data_dir}")
    
    def load_processed_data(self):
        """
        Load processed price and macro data.
        
        Returns
        -------
        tuple
            Processed price data and macro data.
        """
        try:
            #load price data
            price_path = self.processed_dir / "price_data.csv"
            logger.info(f"loading processed price data from {price_path}")
            price_df = pd.read_csv(price_path)
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            
            #load macro data
            macro_path = self.processed_dir/ "macro_data.csv"
            logger.info(f"loading processed macro data from {macro_path}")
            macro_df = pd.read_csv(macro_path)
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            
            logger.info("Processed data loaded successfully")
            return price_df,macro_df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise

    def calculate_market_indicators(self,price_df):
        """
        Calculate market-wide indicators based on price data.
        
        Parameters
        ----------
        price_df : pandas.DataFrame
            Processed price data.
        
        Returns
        -------
        pandas.DataFrame
            Market indicators by date.
        """
        try:
            logger.info("Calculating market indicators")
            #group by date to get market-wide metrics
            market_df = price_df.groupby('Date').agg({
                'Returns': 'mean',
                'Volume': 'sum',
                'Adjusted': 'mean'
            }).reset_index()

            #calculate market indicators
            market_df = market_df.sort_values('Date')
            
            # Market momentum (20-day rolling return)
            market_df['Market_Momentum_20d'] = market_df['Returns'].rolling(window=20).mean()
            
            # Market trend (50-day moving average)
            market_df['Market_Trend_50d'] = market_df['Returns'].rolling(window=50).mean()
            
            # Market volatility (20-day standard deviation of returns)
            market_df['Market_Volatility_20d'] = market_df['Returns'].rolling(window=20).std()
            
            # Volume trend (20-day moving average of volume)
            market_df['Volume_Trend_20d'] = market_df['Volume'].rolling(window=20).mean() / \
                                           market_df['Volume'].rolling(window=60).mean()
            
            logger.info("Market indicators calculated successfully")
            return market_df
        
        except Exception as e:
            logger.error(f"Error calculating market indicators: {e}")
            raise
        
    def calculate_trend_indicators(self, macro_df):
        """
        Calculate trend indicators from macro data.
        
        Parameters
        ----------
        macro_df : pandas.DataFrame
            Macro economic data.
        
        Returns
        -------
        pandas.DataFrame
            Enhanced macro data with trend indicators.
        """
        try:
            logger.info("calculating trend indicators")
            #sort by date
            macro_df = macro_df.sort_values('Date')
            
            #calculate trend indicators for VIX
            macro_df['VIX_MA_10d'] = macro_df['VIX'].rolling(window=10).mean()
            macro_df['VIX_TREND'] = (macro_df['VIX']/macro_df['VIX_MA_10d']) -1
            
            # VIX regime (high volatility vs low volatility)
            vix_median = macro_df['VIX'].median()
            macro_df['VIX_Regime'] = np.where(macro_df['VIX'] > vix_median, 1, 0)
            
            # Calculate trend for Market Return
            macro_df['Market_Return_MA_20d'] = macro_df['Market_Return'].rolling(window=20).mean()
            macro_df['Market_Return_MA_50d'] = macro_df['Market_Return'].rolling(window=50).mean()
            
            # Market regime based on 20d vs 50d moving average
            macro_df['Market_Trend_Regime'] = np.where(
                macro_df['Market_Return_MA_20d'] > macro_df['Market_Return_MA_50d'], 1, 0
            )
            
            # Risk-free rate trend
            macro_df['Risk_Free_Rate_Trend'] = macro_df['Risk_Free_Rate'].pct_change(20)
            
            # Rate regime (rising vs falling)
            macro_df['Rate_Regime'] = np.where(macro_df['Risk_Free_Rate_Trend'] > 0, 1, 0)
            
            logger.info("Trend indicators calculated successfully")
            return macro_df
        
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            raise
    
    def calculate_economic_cycle_indicators(self, macro_df):
        """
        Calculate indicators that help identify the economic cycle.
        
        Parameters
        ----------
        macro_df : pandas.DataFrame
            Macro economic data.
        
        Returns
        -------
        pandas.DataFrame
            Enhanced macro data with economic cycle indicators.
        """
        try:
            logger.info("Calculating economic cycle indicators")
            
            # Sort by date
            macro_df = macro_df.sort_values('Date')
            
            # Calculate market return momentum
            macro_df['Market_Return_Momentum'] = macro_df['Market_Return'].rolling(window=60).mean()
            
            # VIX momentum
            macro_df['VIX_Momentum'] = macro_df['VIX'].pct_change(20)
            
            # Simplified economic cycle indicator based on market return and volatility
            # This is a simplified approach; in practice, use real economic data
            # 0: Contraction, 1: Early Recovery, 2: Expansion, 3: Late Cycle
            conditions = [
                (macro_df['Market_Return_MA_50d'] < 0) & (macro_df['VIX'] > macro_df['VIX'].quantile(0.75)),  # Contraction
                (macro_df['Market_Return_MA_50d'] > 0) & (macro_df['VIX'] > macro_df['VIX'].median()),  # Early Recovery
                (macro_df['Market_Return_MA_50d'] > 0) & (macro_df['VIX'] < macro_df['VIX'].median()),  # Expansion
                (macro_df['Market_Return_MA_50d'] < 0) & (macro_df['VIX'] < macro_df['VIX'].quantile(0.75))   # Late Cycle
            ]
            choices = [0, 1, 2, 3]
            macro_df['Economic_Cycle'] = np.select(conditions, choices, default=1)
            
            logger.info("Economic cycle indicators calculated successfully")
            return macro_df
        
        except Exception as e:
            logger.error(f"Error calculating economic cycle indicators: {e}")
            raise

    def merge_indicators(self, market_df, macro_df):
        """
        Merge market and macro indicators.
        
        Parameters
        ----------
        market_df : pandas.DataFrame
            Market indicators.
        macro_df : pandas.DataFrame
            Macro indicators.
        
        Returns
        -------
        pandas.DataFrame
            Merged macro indicators.
        """
        try:
            logger.info("Merging market and macro indicators")
            
            # Merge market and macro dataframes on Date
            merged_df = pd.merge(market_df, macro_df, on='Date', how='inner')
            
            logger.info(f"Merged indicators with shape {merged_df.shape}")
            return merged_df
        
        except Exception as e:
            logger.error(f"Error merging indicators: {e}")
            raise

    def generate_macro_features(self):
        """
        Generate all macro indicators.
        
        Returns
        -------
        pandas.DataFrame
            Complete set of macro indicators.
        """
        try:
            logger.info("Generating all macro indicators")
            
            # Load processed data
            price_df, macro_df = self.load_processed_data()
            
            # Calculate market indicators
            market_df = self.calculate_market_indicators(price_df)
            
            # Calculate trend indicators
            enhanced_macro_df = self.calculate_trend_indicators(macro_df)
            
            # Calculate economic cycle indicators
            enhanced_macro_df = self.calculate_economic_cycle_indicators(enhanced_macro_df)
            
            # Merge all indicators
            all_macro_indicators = self.merge_indicators(market_df, enhanced_macro_df)
            
            # Save to processed directory
            output_path = self.processed_dir / "macro_indicators.csv"
            all_macro_indicators.to_csv(output_path, index=False)
            logger.info(f"Saved all macro indicators to {output_path}")
            
            return all_macro_indicators
        
        except Exception as e:
            logger.error(f"Error generating macro features: {e}")
            raise

if __name__ == "__main__":
    #example usage of MacroIndicators class
    macro_indicators = MacroIndicators()
    
    # Generate all macro indicators
    all_indicators = macro_indicators.generate_macro_features()
    
    # Print sample data
    print("\nMacro Indicators Sample:")
    print(all_indicators.head())
    
    # Print statistics
    print("\nMacro Indicators Statistics:")
    print(all_indicators.describe())
