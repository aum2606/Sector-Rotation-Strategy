import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "backtest_engine.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Class for backtesting investment strategies.
    Implements a framework for running backtests on different strategies
    and comparing their performance.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize BacktestEngine class.
        
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
        self.reports_dir = Path(__file__).parent.parent.parent / "reports"
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Store strategies and their results
        self.strategies = {}
        self.results = {}
        
        logger.info(f"BacktestEngine initialized with data directory: {self.data_dir}")

    def load_strategy_results(self, strategy_name, results_file=None):
        """
        Load results for a specific strategy.
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy.
        results_file : str, optional
            Path to the results CSV file. If None, use default naming convention.
        
        Returns
        -------
        pandas.DataFrame
            Strategy results.
        """
        try:
            if results_file is None:
                if strategy_name.lower() == "benchmark":
                    results_file = self.processed_dir / "benchmark_results.csv"
                elif strategy_name.lower() == "sector_rotation":
                    results_file = self.processed_dir / "sector_rotation_results.csv"
                else:
                    results_file = self.processed_dir / f"{strategy_name.lower()}_results.csv"
            
            logger.info(f"Loading results for {strategy_name} from {results_file}")
            
            results_df = pd.read_csv(results_file)
            
            # Convert Date to datetime
            if 'Date' in results_df.columns:
                results_df['Date'] = pd.to_datetime(results_df['Date'])
            
            self.strategies[strategy_name] = results_file
            self.results[strategy_name] = results_df
            
            logger.info(f"Loaded {len(results_df)} rows of results for {strategy_name}")
            return results_df
        
        except Exception as e:
            logger.error(f"Error loading results for {strategy_name}: {e}")
            raise
    
    def combine_strategy_results(self, strategy_dataframes):
        """
        Combine results from multiple strategy DataFrames for comparison.
        
        Parameters
        ----------
        strategy_dataframes : list
            List of DataFrames containing strategy results.
        
        Returns
        -------
        pandas.DataFrame
            Combined strategy results.
        """
        try:
            if not strategy_dataframes:
                logger.warning("No strategy DataFrames provided")
                return pd.DataFrame()
                
            logger.info(f"Combining results from {len(strategy_dataframes)} strategy DataFrames")
            combined_results = []
            
            for df in strategy_dataframes:
                if isinstance(df, pd.DataFrame):
                    # Check if 'Strategy' column exists in the DataFrame
                    if 'Strategy' in df.columns:
                        combined_results.append(df)
                    else:
                        # Skip DataFrames without a Strategy column
                        logger.warning("Skipping DataFrame without 'Strategy' column")
                else:
                    logger.warning(f"Skipping non-DataFrame object: {type(df)}")
                        
            if combined_results:
                # Combine all results
                all_results = pd.concat(combined_results, ignore_index=True)
                
                # Save combined results
                output_path = self.reports_dir / "combined_results.csv"
                all_results.to_csv(output_path, index=False)
                logger.info(f"Combined results saved to {output_path}")
                
                return all_results
            else:
                logger.warning("No valid strategy results to combine")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error combining strategy results: {e}")
            raise
        
    def run_backtest(self, strategy_class, strategy_params=None, start_date=None, end_date=None):
        """
        Run a backtest for a specific strategy.
        
        Parameters
        ----------
        strategy_class : class
            Class implementing the strategy to backtest.
        strategy_params : dict, optional
            Parameters to pass to the strategy.
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Backtest results.
        """
        try:
            strategy_name = strategy_class.__name__
            logger.info(f"Running backtest for {strategy_name}")
            
            #initialize strategy 
            if strategy_params is None:
                strategy_params = {}
                
            strategy = strategy_class
            
            #get available backtest methods
            if hasattr(strategy, 'run_all_strategies'):
                # Strategy class has a method to run all strategies
                results = strategy.run_all_strategies(start_date=start_date, end_date=end_date)
            elif hasattr(strategy, 'run_all_benchmarks'):
                # Benchmark class has a method to run all benchmarks
                results = strategy.run_all_benchmarks(start_date=start_date, end_date=end_date)
            else:
                logger.error(f"Strategy {strategy_name} does not have a recognized backtest method")
                raise ValueError(f"Strategy {strategy_name} does not have a recognized backtest method")
            
            #store results
            self.strategies[strategy_name] = strategy
            self.results[strategy_name] = results
            
            logger.info(f"Backtest completed for {strategy_name}")
            return results
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    def save_backtest_results(self, strategy_name, output_file=None):
        """
        Save backtest results to a CSV file.
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy.
        output_file : str, optional
            Path to the output CSV file. If None, use default naming convention.
        """
        try:
            if strategy_name not in self.results:
                logger.error(f"startegy {strategy_name} not found in results")
                raise ValueError(f"Strategy {strategy_name} not found in results")
            
            #get results for this strategy
            results = self.results[strategy_name]
            
            #detemine output files
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.reports_dir / f"{strategy_name}_backtest_{timestamp}.csv"
            
            logger.info(f"Saving backtest results for {strategy_name} to {output_file}")
            
            #save results
            results.to_csv(output_file,index=False)
            logger.info(f"Backtest results saved to {output_file}")
        
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the BacktestEngine class
    backtest_engine = BacktestEngine()
    
    # Load existing results
    try:
        benchmark_results = backtest_engine.load_strategy_results("benchmark")
        sector_rotation_results = backtest_engine.load_strategy_results("sector_rotation")
        
        # Combine results
        combined_results = backtest_engine.combine_strategy_results([benchmark_results, sector_rotation_results])
        
        # Print sample results
        print("\nCombined Results Sample:")
        print(combined_results.head())
        
        # Print final performance for each strategy
        print("\nFinal Performance by Strategy:")
        final_performance = combined_results.groupby('Strategy')['Cumulative_Return'].last()
        print(final_performance)
    except FileNotFoundError:
        print("Results files not found. Run strategy backtests first.")