import os
import pandas as pd
import logging
from pathlib import Path

# Create logs directory if it doesn't exist
# Use an absolute path to ensure logs directory is created in the proper location
logs_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "data_ingestion.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataIngestion:
    """Class for ingesting data from CSV files."""
    
    def __init__(self, data_dir=None):
        """
        Initialize DataIngestion class.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the data files.
        """
        if data_dir is None:
            # Default to the standard project structure
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        logger.info(f"Data directories initialized: {self.data_dir}")
    
    def load_portfolio(self):
        """
        Load portfolio data from CSV.
        
        Returns
        -------
        pandas.DataFrame
            Portfolio data.
        """
        try:
            file_path = self.raw_dir / "Portfolio.csv"
            logger.info(f"Loading portfolio data from {file_path}")
            
            portfolio_df = pd.read_csv(file_path)
            
            # Validate data
            expected_columns = ["Ticker", "Quantity", "Sector", "Close", "Weight"]
            for col in expected_columns:
                if col not in portfolio_df.columns:
                    raise ValueError(f"Column {col} not found in portfolio data")
            
            logger.info(f"Portfolio data loaded successfully with shape {portfolio_df.shape}")
            return portfolio_df
        
        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            raise
    
    def load_portfolio_prices(self):
        """
        Load portfolio price history data from CSV.
        
        Returns
        -------
        pandas.DataFrame
            Portfolio price history data.
        """
        try:
            file_path = self.raw_dir / "Portfolio_prices.csv"
            logger.info(f"Loading portfolio price data from {file_path}")
            
            prices_df = pd.read_csv(file_path)
            
            # Validate data
            expected_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adjusted", "Returns", "Volume"]
            for col in expected_columns:
                if col not in prices_df.columns:
                    raise ValueError(f"Column {col} not found in price data")
            
            logger.info(f"Portfolio price data loaded successfully with shape {prices_df.shape}")
            return prices_df
        
        except Exception as e:
            logger.error(f"Error loading portfolio price data: {e}")
            raise
    
    def save_processed_data(self, data, file_name):
        """
        Save processed data to CSV.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Processed data.
        file_name : str
            Name of the output file.
        """
        try:
            output_path = self.processed_dir / file_name
            logger.info(f"Saving processed data to {output_path}")
            
            data.to_csv(output_path, index=False)
            logger.info(f"Data saved successfully to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise


if __name__ == "__main__":
    # Example usage of the DataIngestion class
    data_ingestion = DataIngestion()
    
    # Load the data
    portfolio_df = data_ingestion.load_portfolio()
    prices_df = data_ingestion.load_portfolio_prices()
    
    # Print sample data
    print("\nPortfolio Data Sample:")
    print(portfolio_df.head())
    
    print("\nPortfolio Prices Data Sample:")
    print(prices_df.head())
    
    # Print data info
    print("\nPortfolio Data Info:")
    print(portfolio_df.info())
    
    print("\nPortfolio Prices Data Info:")
    print(prices_df.info())