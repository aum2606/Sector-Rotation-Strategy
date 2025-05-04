# src/data/data_preprocessing.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from .data_ingestion import DataIngestion
import os


logs_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "data_preprocessing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataPreprocessing:
    """Class for preprocessing data for sector rotation strategy."""
    
    def __init__(self, data_ingestion=None):
        """
        Initialize DataPreprocessing class.
        
        Parameters
        ----------
        data_ingestion : DataIngestion
            DataIngestion instance.
        """
        if data_ingestion is None:
            self.data_ingestion = DataIngestion()
        else:
            self.data_ingestion = data_ingestion
        
        logger.info("DataPreprocessing initialized")

    
    def process_portfolio_data(self):
        """
        Process portfolio data.
        
        Returns
        -------
        pandas.DataFrame
            Processed portfolio data.
        """
        try:
            logger.info("Processing portfolio data")
            
            # Load portfolio data
            portfolio_df = self.data_ingestion.load_portfolio()
            
            # Clean data
            # Check for missing values
            missing_values = portfolio_df.isnull().sum()
            if missing_values.sum() > 0:
                logger.warning(f"Missing values found in portfolio data: {missing_values}")
                portfolio_df = portfolio_df.dropna()
                logger.info("Dropped rows with missing values")
            
            # Validate data types
            portfolio_df['Ticker'] = portfolio_df['Ticker'].astype(str)
            portfolio_df['Quantity'] = portfolio_df['Quantity'].astype(int)
            portfolio_df['Sector'] = portfolio_df['Sector'].astype(str)
            portfolio_df['Close'] = portfolio_df['Close'].astype(float)
            portfolio_df['Weight'] = portfolio_df['Weight'].astype(float)
            
            # Calculate market value
            portfolio_df['Market_Value'] = portfolio_df['Quantity'] * portfolio_df['Close']
            
            # Validate weights sum to 1 (or close to 1)
            weight_sum = portfolio_df['Weight'].sum()
            if not 0.99 <= weight_sum <= 1.01:
                logger.warning(f"Portfolio weights do not sum to 1. Sum: {weight_sum}")
                # Normalize weights
                portfolio_df['Weight'] = portfolio_df['Weight'] / weight_sum
                logger.info("Normalized portfolio weights to sum to 1")
            
            logger.info("Portfolio data processed successfully")
            return portfolio_df
        
        except Exception as e:
            logger.error(f"Error processing portfolio data: {e}")
            raise
    
    def process_price_data(self):
        """
        Process portfolio price data.
        
        Returns
        -------
        pandas.DataFrame
            Processed price data.
        """
        try:
            logger.info("Processing portfolio price data")
            
            # Load price data
            prices_df = self.data_ingestion.load_portfolio_prices()
            
            # Convert Date to datetime
            prices_df['Date'] = pd.to_datetime(prices_df['Date'])
            
            # Check for missing values
            missing_values = prices_df.isnull().sum()
            if missing_values.sum() > 0:
                logger.warning(f"Missing values found in price data: {missing_values}")
                
                # Forward fill prices for minor gaps
                prices_df = prices_df.sort_values(['Ticker', 'Date'])
                prices_df = prices_df.groupby('Ticker').ffill()
                
                # Drop remaining rows with missing values
                missing_after_fill = prices_df.isnull().sum().sum()
                if missing_after_fill > 0:
                    logger.warning(f"Still have {missing_after_fill} missing values after forward fill")
                    prices_df = prices_df.dropna()
                    logger.info("Dropped remaining rows with missing values")
            
            # Calculate additional metrics if Returns column has issues
            if prices_df['Returns'].isnull().sum() > 0:
                logger.info("Recalculating returns")
                # Calculate daily returns using Adjusted prices
                prices_df = prices_df.sort_values(['Ticker', 'Date'])
                prices_df['Returns'] = prices_df.groupby('Ticker')['Adjusted'].pct_change()
            
            # Check for outliers in Returns
            returns_mean = prices_df['Returns'].mean()
            returns_std = prices_df['Returns'].std()
            outlier_threshold = 5  # 5 standard deviations
            outlier_mask = abs(prices_df['Returns'] - returns_mean) > outlier_threshold * returns_std
            
            if outlier_mask.sum() > 0:
                logger.warning(f"Found {outlier_mask.sum()} outliers in Returns")
                # Cap outliers
                cap_value = returns_mean + outlier_threshold * returns_std
                floor_value = returns_mean - outlier_threshold * returns_std
                
                prices_df.loc[prices_df['Returns'] > cap_value, 'Returns'] = cap_value
                prices_df.loc[prices_df['Returns'] < floor_value, 'Returns'] = floor_value
                logger.info(f"Capped outliers in Returns at {cap_value:.4f} and {floor_value:.4f}")
            
            logger.info("Portfolio price data processed successfully")
            return prices_df
        
        except Exception as e:
            logger.error(f"Error processing portfolio price data: {e}")
            raise
    
    def create_sector_data(self, portfolio_df, prices_df):
        """
        Create sector-level data from portfolio and price data.
        
        Parameters
        ----------
        portfolio_df : pandas.DataFrame
            Processed portfolio data.
        prices_df : pandas.DataFrame
            Processed price data.
        
        Returns
        -------
        pandas.DataFrame
            Sector-level data.
        """
        try:
            logger.info("Creating sector-level data")
            
            # Merge portfolio data with price data to get sector information
            merged_df = pd.merge(prices_df, portfolio_df[['Ticker', 'Sector']], on='Ticker', how='left')
            
            # Group by date and sector to calculate sector-level metrics
            sector_df = merged_df.groupby(['Date', 'Sector']).agg({
                'Returns': 'mean',  # Average returns across sector stocks
                'Volume': 'sum',    # Total volume for the sector
                'Adjusted': 'mean'  # Average adjusted price
            }).reset_index()
            
            # Calculate sector cumulative returns
            sector_df['Cumulative_Returns'] = sector_df.groupby('Sector')['Returns'].transform(
                lambda x: (1 + x).cumprod() - 1
            )
            
            # Calculate rolling metrics
            sector_df = sector_df.sort_values(['Sector', 'Date'])
            
            # 20-day rolling average of returns (momentum)
            sector_df['Return_20d'] = sector_df.groupby('Sector')['Returns'].transform(
                lambda x: x.rolling(window=20).mean()
            )
            
            # 50-day rolling average
            sector_df['Return_50d'] = sector_df.groupby('Sector')['Returns'].transform(
                lambda x: x.rolling(window=50).mean()
            )
            
            # 20-day rolling volatility
            sector_df['Volatility_20d'] = sector_df.groupby('Sector')['Returns'].transform(
                lambda x: x.rolling(window=20).std()
            )
            
            logger.info("Sector-level data created successfully")
            return sector_df
        
        except Exception as e:
            logger.error(f"Error creating sector-level data: {e}")
            raise
    
    def run_preprocessing_pipeline(self):
        """
        Run the complete preprocessing pipeline.
        
        Returns
        -------
        tuple
            Processed portfolio data, price data, and sector data.
        """
        try:
            logger.info("Running preprocessing pipeline")
            
            # Process portfolio data
            portfolio_df = self.process_portfolio_data()
            self.data_ingestion.save_processed_data(portfolio_df, "portfolio_data.csv")
            
            # Process price data
            prices_df = self.process_price_data()
            
            # Create sector data
            sector_df = self.create_sector_data(portfolio_df, prices_df)
            
            # Save processed data
            self.data_ingestion.save_processed_data(prices_df, "price_data.csv")
            self.data_ingestion.save_processed_data(sector_df, "sector_data.csv")
            
            # Create macro data (placeholder for now)
            # In practice, this would include economic indicators, interest rates, etc.
            dates = prices_df['Date'].unique()
            macro_df = pd.DataFrame({
                'Date': dates,
                'Market_Return': np.random.normal(0.0005, 0.01, len(dates)),  # Placeholder
                'Risk_Free_Rate': np.random.uniform(0.02, 0.05, len(dates)) / 252,  # Daily rate
                'VIX': np.random.uniform(15, 30, len(dates))  # Placeholder
            })
            macro_df = macro_df.sort_values('Date')
            self.data_ingestion.save_processed_data(macro_df, "macro_data.csv")
            
            logger.info("Preprocessing pipeline completed successfully")
            return portfolio_df, prices_df, sector_df, macro_df
        
        except Exception as e:
            logger.error(f"Error running preprocessing pipeline: {e}")
            raise


if __name__ == "__main__":
    # Example usage of the DataPreprocessing class
    data_preprocessing = DataPreprocessing()
    
    # Run preprocessing pipeline
    portfolio_df, prices_df, sector_df, macro_df = data_preprocessing.run_preprocessing_pipeline()
    
    # Print sample data
    print("\nProcessed Portfolio Data Sample:")
    print(portfolio_df.head())
    
    print("\nProcessed Price Data Sample:")
    print(prices_df.head())
    
    print("\nSector Data Sample:")
    print(sector_df.head())
    
    print("\nMacro Data Sample:")
    print(macro_df.head())