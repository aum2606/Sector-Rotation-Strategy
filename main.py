import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Import custom modules
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.features.macro_indicators import MacroIndicators
from src.features.sector_indicators import SectorIndicators
from src.models.sector_rotation import SectorRotationModel
from src.models.benchmark_model import BenchmarkModel
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.visualization import Visualization

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "main.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sector Rotation Strategy Pipeline'
    )
    
    parser.add_argument(
        '--start_date', type=str, default=None,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date', type=str, default=None,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--strategies', type=str, nargs='+', 
        default=['momentum', 'relative_strength', 'economic_cycle', 'combined'],
        help='Strategies to run (momentum, relative_strength, economic_cycle, combined)'
    )
    
    parser.add_argument(
        '--benchmarks', type=str, nargs='+',
        default=['buy_and_hold', 'equal_weight', 'market_cap'],
        help='Benchmarks to run (buy_and_hold, equal_weight, market_cap)'
    )
    
    parser.add_argument(
        '--rebalance_frequency', type=str, default='M',
        help='Rebalancing frequency (D, W, M, Q, Y)'
    )
    
    parser.add_argument(
        '--n_sectors', type=int, default=3,
        help='Number of top sectors to invest in'
    )
    
    parser.add_argument(
        '--generate_reports', action='store_true',
        help='Generate performance reports and visualizations'
    )
    
    return parser.parse_args()

def data_processing_pipeline():
    """Run the data processing pipeline."""
    logger.info("Starting data processing pipeline")
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Run data preprocessing
    data_preprocessing = DataPreprocessing(data_ingestion)
    portfolio_df, prices_df, sector_df, macro_df = data_preprocessing.run_preprocessing_pipeline()
    
    # Generate macro indicators
    macro_indicators = MacroIndicators()
    macro_indicators_df = macro_indicators.generate_macro_features()
    logger.info(f"Generated macro indicators with shape {macro_indicators_df.shape}")
    
    # Generate sector indicators
    sector_indicators = SectorIndicators()
    sector_indicators_df = sector_indicators.generate_sector_features()
    logger.info(f"Generated sector indicators with shape {sector_indicators_df.shape}")
    
    logger.info("Data processing pipeline completed successfully")
    return portfolio_df, prices_df, sector_df, macro_df, macro_indicators_df, sector_indicators_df

def strategy_backtest_pipeline(start_date=None, end_date=None, strategies=None, 
                             benchmarks=None, rebalance_frequency='M', n_sectors=3):
    """Run the strategy backtesting pipeline."""
    logger.info("Starting strategy backtesting pipeline")
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine()
    
    # Run strategy backtests
    results_list = []
    
    # Initialize models
    sector_rotation = SectorRotationModel()
    benchmark_model = BenchmarkModel()
    
    # Run benchmark strategies if requested
    if benchmarks:
        logger.info(f"Running benchmark strategies: {benchmarks}")
        if 'all' in benchmarks or any(b in ['buy_and_hold', 'equal_weight', 'market_cap'] for b in benchmarks):
            benchmark_results = benchmark_model.run_all_benchmarks(start_date, end_date)
            results_list.append(benchmark_results)
            logger.info(f"Benchmark strategies completed with {len(benchmark_results)} rows")
    
    # Run sector rotation strategies if requested
    if strategies:
        logger.info(f"Running sector rotation strategies: {strategies}")
        if 'all' in strategies or any(s in ['momentum', 'relative_strength', 'economic_cycle', 'combined'] for s in strategies):
            rotation_results = sector_rotation.run_all_strategies(start_date, end_date)
            results_list.append(rotation_results)
            logger.info(f"Sector rotation strategies completed with {len(rotation_results)} rows")
    
    # Combine all results
    if results_list:
        all_results = backtest_engine.combine_strategy_results(results_list)
        logger.info(f"Combined results with {len(all_results)} rows")
    else:
        all_results = pd.DataFrame()
        logger.warning("No strategy results to combine")
    
    logger.info("Strategy backtesting pipeline completed successfully")
    return all_results


def analysis_pipeline(results_df):
    """Run the analysis pipeline."""
    logger.info("Starting analysis pipeline")
    
    # Initialize performance metrics calculator
    performance_metrics = PerformanceMetrics()
    
    # Calculate performance metrics for all strategies
    metrics_df = performance_metrics.evaluate_strategy(results_df)
    logger.info(f"Performance metrics calculated for {len(metrics_df)} strategies")
    
    # Save metrics to reports directory
    reports_dir = Path(__file__).parent / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    metrics_path = reports_dir / "performance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Performance metrics saved to {metrics_path}")
    
    # Initialize visualization module
    visualizer = Visualization()
    
    # Create various plots
    logger.info("Generating visualizations")
    
    # Cumulative returns
    visualizer.plot_cumulative_returns(results_df)
    
    # Drawdowns
    visualizer.plot_drawdowns(results_df)
    
    # Rolling returns
    visualizer.plot_rolling_returns(results_df, window=60)
    
    # Returns distribution
    visualizer.plot_returns_distribution(results_df)
    
    # Performance metrics visualization
    visualizer.plot_performance_metrics(metrics_df)
    
    # Correlation heatmap
    visualizer.plot_correlation_heatmap(results_df)
    
    # Underwater chart
    visualizer.plot_underwater(results_df)
    
    # Risk-return scatter plot
    visualizer.plot_risk_return_scatter(metrics_df)
    
    # If there's sector allocation data, plot it for sector rotation strategies
    if 'Selected_Sectors' in results_df.columns:
        # Filter for sector rotation strategies only
        rotation_strategies = [s for s in results_df['Strategy'].unique() 
                            if 'Rotation' in s or 'rotation' in s]
        if rotation_strategies:
            sector_data = results_df[results_df['Strategy'].isin(rotation_strategies)]
            visualizer.plot_sector_allocation(sector_data)
    
    # Load macro data for regime analysis if available
    try:
        macro_indicators = MacroIndicators()
        _, macro_df = macro_indicators.load_processed_data()
        
        # Plot regime analysis if economic cycle data is available
        if 'Economic_Cycle' in macro_df.columns:
            visualizer.plot_regime_analysis(results_df, macro_df)
    except Exception as e:
        logger.warning(f"Could not perform regime analysis: {e}")
    
    # Benchmark comparison
    benchmark_strategy = 'Buy and Hold'
    if benchmark_strategy in results_df['Strategy'].unique():
        visualizer.plot_benchmark_comparison(results_df, benchmark_strategy)
    
    logger.info("Analysis pipeline completed successfully")
    return metrics_df

def main():
    """Main function to run the entire pipeline."""
    logger.info("Starting Sector Rotation Strategy Pipeline")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Run data processing pipeline
    portfolio_df, prices_df, sector_df, macro_df, macro_indicators_df, sector_indicators_df = data_processing_pipeline()
    
    # Run strategy backtesting pipeline
    results_df = strategy_backtest_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        strategies=args.strategies,
        benchmarks=args.benchmarks,
        rebalance_frequency=args.rebalance_frequency,
        n_sectors=args.n_sectors
    )
    
    # Run analysis pipeline if requested
    if args.generate_reports:
        metrics_df = analysis_pipeline(results_df)
    
    logger.info("Sector Rotation Strategy Pipeline completed successfully")

if __name__ == "__main__":
    main()