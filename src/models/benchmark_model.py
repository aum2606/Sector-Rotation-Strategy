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
        logging.FileHandler(logs_dir / "benchmark_model.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BenchmarkModel:
    """
    Class for implementing benchmark models to compare against sector rotation strategy.
    Implements buy-and-hold, equal weight, and market cap weighted strategies.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize BenchmarkModel class.
        
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
        
        logger.info(f"BenchmarkModel initialized with data directory: {self.data_dir}")
    
    def load_processed_data(self):
        """
        Load processed portfolio, price, and sector data.
        
        Returns
        -------
        tuple
            Portfolio data, price data, and sector data.
        """
        try:
            # Load portfolio data
            portfolio_path = self.processed_dir / "portfolio_data.csv"
            logger.info(f"Loading processed portfolio data from {portfolio_path}")
            portfolio_df = pd.read_csv(portfolio_path)
            
            # Load price data
            price_path = self.processed_dir / "price_data.csv"
            logger.info(f"Loading processed price data from {price_path}")
            price_df = pd.read_csv(price_path)
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            
            # Load sector data
            sector_path = self.processed_dir / "sector_data.csv"
            logger.info(f"Loading processed sector data from {sector_path}")
            sector_df = pd.read_csv(sector_path)
            sector_df['Date'] = pd.to_datetime(sector_df['Date'])
            
            logger.info("Processed data loaded successfully")
            return portfolio_df, price_df, sector_df
        
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def buy_and_hold_strategy(self, portfolio_df, price_df, start_date=None, end_date=None):
        """
        Implement a buy and hold strategy based on initial portfolio weights.
        
        Parameters
        ----------
        portfolio_df : pandas.DataFrame
            Portfolio data with initial weights.
        price_df : pandas.DataFrame
            Price data for portfolio stocks.
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of buy and hold strategy.
        """
        try:
            logger.info("Implementing buy and hold strategy")
            
            # Filter price data by date if specified
            if start_date is not None:
                price_df = price_df[price_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                price_df = price_df[price_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = price_df['Date'].unique()
            dates.sort()
            
            # Get initial portfolio composition
            initial_portfolio = portfolio_df.copy()
            
            # Calculate daily returns for each asset weighted by initial weights
            daily_returns = []
            
            for date in dates:
                # Get returns for this date
                date_returns = price_df[price_df['Date'] == date]
                
                # Merge with portfolio to get weights
                merged = pd.merge(date_returns, initial_portfolio[['Ticker', 'Weight']], 
                                 on='Ticker', how='inner')
                
                # Calculate weighted returns
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Buy and Hold',
                        'Daily_Return': portfolio_return,
                        'Tickers_Count': len(merged)
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Buy and hold strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing buy and hold strategy: {e}")
            raise
    
    def equal_weight_strategy(self, portfolio_df, price_df, rebalance_frequency='Q', 
                             start_date=None, end_date=None):
        """
        Implement an equal weight strategy with periodic rebalancing.
        
        Parameters
        ----------
        portfolio_df : pandas.DataFrame
            Portfolio data.
        price_df : pandas.DataFrame
            Price data for portfolio stocks.
        rebalance_frequency : str, default='Q'
            Frequency for rebalancing: 'D' (daily), 'W' (weekly), 'M' (monthly), 
            'Q' (quarterly), or 'Y' (yearly).
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of equal weight strategy.
        """
        try:
            logger.info(f"Implementing equal weight strategy with {rebalance_frequency} rebalancing")
            
            # Filter price data by date if specified
            if start_date is not None:
                price_df = price_df[price_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                price_df = price_df[price_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = price_df['Date'].unique()
            dates.sort()
            
            # Get unique tickers in the portfolio
            tickers = portfolio_df['Ticker'].unique()
            
            # Calculate daily returns
            daily_returns = []
            current_weights = None  # Will be initialized at first rebalance
            
            # Define rebalance dates based on frequency
            if rebalance_frequency == 'D':
                rebalance_dates = dates
            else:
                # Convert dates to pandas DatetimeIndex for resampling
                date_index = pd.DatetimeIndex(dates)
                temp_rebalance_dates = pd.date_range(
                    start=date_index.min(),
                    end=date_index.max(),
                    freq=rebalance_frequency
                )
                # Convert to same format as dates for comparison
                temp_rebalance_dates = [pd.Timestamp(d) for d in temp_rebalance_dates]
                # Ensure rebalance dates exist in our data
                # Find the closest available dates to our desired rebalance dates
                rebalance_dates = []
                for rebal_date in temp_rebalance_dates:
                    # Find the closest date in our actual dates that's >= rebal_date
                    closest_dates = [d for d in dates if d >= rebal_date]
                    if closest_dates:
                        rebalance_dates.append(min(closest_dates))
                
                # Ensure we have unique dates
                rebalance_dates = sorted(list(set(rebalance_dates)))
            
            # Add the first date as a rebalance date if it's not already included
            if len(rebalance_dates) == 0 or dates[0] != rebalance_dates[0]:
                rebalance_dates = [dates[0]] + rebalance_dates
            
            logger.info(f"Rebalancing will occur on {len(rebalance_dates)} dates")
            
            next_rebalance_idx = 0
            
            for date in dates:
                # Check if we need to rebalance
                if next_rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[next_rebalance_idx]:
                    # Get available tickers for this date
                    available_tickers = price_df[price_df['Date'] == date]['Ticker'].unique()
                    
                    # Calculate equal weights
                    equal_weight = 1.0 / len(available_tickers)
                    
                    # Create new weights dataframe
                    current_weights = pd.DataFrame({
                        'Ticker': available_tickers,
                        'Weight': equal_weight
                    })
                    
                    logger.debug(f"Rebalanced on {date} with {len(available_tickers)} tickers")
                    
                    # Move to next rebalance date
                    next_rebalance_idx += 1
                
                # Skip if we haven't initialized weights yet
                if current_weights is None:
                    continue
                
                # Get returns for this date
                date_returns = price_df[price_df['Date'] == date]
                
                # Merge with current weights
                merged = pd.merge(date_returns, current_weights, on='Ticker', how='inner')
                
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Equal Weight',
                        'Daily_Return': portfolio_return,
                        'Tickers_Count': len(merged)
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Equal weight strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing equal weight strategy: {e}")
            raise
    
    def market_cap_weighted_strategy(self, portfolio_df, price_df, rebalance_frequency='Q',
                                   start_date=None, end_date=None):
        """
        Implement a market cap weighted strategy with periodic rebalancing.
        
        Parameters
        ----------
        portfolio_df : pandas.DataFrame
            Portfolio data with market cap information.
        price_df : pandas.DataFrame
            Price data for portfolio stocks.
        rebalance_frequency : str, default='Q'
            Frequency for rebalancing: 'D' (daily), 'W' (weekly), 'M' (monthly), 
            'Q' (quarterly), or 'Y' (yearly).
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of market cap weighted strategy.
        """
        try:
            logger.info(f"Implementing market cap weighted strategy with {rebalance_frequency} rebalancing")
            
            # Filter price data by date if specified
            if start_date is not None:
                price_df = price_df[price_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                price_df = price_df[price_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = price_df['Date'].unique()
            dates.sort()
            
            # Calculate daily returns
            daily_returns = []
            current_weights = None  # Will be initialized at first rebalance
            
            # Define rebalance dates based on frequency
            if rebalance_frequency == 'D':
                rebalance_dates = dates
            else:
                # Convert dates to pandas DatetimeIndex for resampling
                date_index = pd.DatetimeIndex(dates)
                temp_rebalance_dates = pd.date_range(
                    start=date_index.min(),
                    end=date_index.max(),
                    freq=rebalance_frequency
                )
                # Convert to same format as dates for comparison
                temp_rebalance_dates = [pd.Timestamp(d) for d in temp_rebalance_dates]
                # Ensure rebalance dates exist in our data
                # Find the closest available dates to our desired rebalance dates
                rebalance_dates = []
                for rebal_date in temp_rebalance_dates:
                    # Find the closest date in our actual dates that's >= rebal_date
                    closest_dates = [d for d in dates if d >= rebal_date]
                    if closest_dates:
                        rebalance_dates.append(min(closest_dates))
                
                # Ensure we have unique dates
                rebalance_dates = sorted(list(set(rebalance_dates)))
            
            # Add the first date as a rebalance date if it's not already included
            if len(rebalance_dates) == 0 or dates[0] != rebalance_dates[0]:
                rebalance_dates = [dates[0]] + rebalance_dates
            
            logger.info(f"Rebalancing will occur on {len(rebalance_dates)} dates")
            
            next_rebalance_idx = 0
            
            for date in dates:
                # Check if we need to rebalance
                if next_rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[next_rebalance_idx]:
                    # Get prices for this date
                    date_prices = price_df[price_df['Date'] == date]
                    
                    # Calculate market cap for each ticker (using Adjusted price and Quantity)
                    market_cap = pd.merge(
                        date_prices[['Ticker', 'Adjusted']],
                        portfolio_df[['Ticker', 'Quantity']],
                        on='Ticker',
                        how='inner'
                    )
                    
                    market_cap['Market_Cap'] = market_cap['Adjusted'] * market_cap['Quantity']
                    
                    # Calculate weights based on market cap
                    total_market_cap = market_cap['Market_Cap'].sum()
                    market_cap['Weight'] = market_cap['Market_Cap'] / total_market_cap
                    
                    # Create new weights dataframe
                    current_weights = market_cap[['Ticker', 'Weight']]
                    
                    logger.debug(f"Rebalanced on {date} with {len(current_weights)} tickers")
                    
                    # Move to next rebalance date
                    next_rebalance_idx += 1
                
                # Skip if we haven't initialized weights yet
                if current_weights is None:
                    continue
                
                # Get returns for this date
                date_returns = price_df[price_df['Date'] == date]
                
                # Merge with current weights
                merged = pd.merge(date_returns, current_weights, on='Ticker', how='inner')
                
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Market Cap Weighted',
                        'Daily_Return': portfolio_return,
                        'Tickers_Count': len(merged)
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Market cap weighted strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing market cap weighted strategy: {e}")
            raise
    
    def run_all_benchmarks(self, start_date=None, end_date=None):
        """
        Run all benchmark strategies and combine results.
        
        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Combined performance of all benchmark strategies.
        """
        try:
            logger.info("Running all benchmark strategies")
            
            # Load data
            portfolio_df, price_df, sector_df = self.load_processed_data()
            
            # Run buy and hold strategy
            buy_hold_results = self.buy_and_hold_strategy(
                portfolio_df, price_df, start_date, end_date
            )
            
            # Run equal weight strategy with quarterly rebalancing
            equal_weight_results = self.equal_weight_strategy(
                portfolio_df, price_df, 'Q', start_date, end_date
            )
            
            # Run market cap weighted strategy with quarterly rebalancing
            market_cap_results = self.market_cap_weighted_strategy(
                portfolio_df, price_df, 'Q', start_date, end_date
            )
            
            # Combine results
            all_results = pd.concat([
                buy_hold_results,
                equal_weight_results,
                market_cap_results
            ])
            
            # Save results
            output_path = self.processed_dir / "benchmark_results.csv"
            all_results.to_csv(output_path, index=False)
            logger.info(f"Benchmark results saved to {output_path}")
            
            return all_results
        
        except Exception as e:
            logger.error(f"Error running all benchmark strategies: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the BenchmarkModel class
    benchmark_model = BenchmarkModel()
    
    # Run all benchmark strategies
    results = benchmark_model.run_all_benchmarks()
    
    # Print sample results
    print("\nBenchmark Results Sample:")
    print(results.head())
    
    # Get final performance for each strategy
    final_performance = results.groupby('Strategy')['Cumulative_Return'].last()
    print("\nFinal Performance:")
    print(final_performance)