import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from scipy import stats

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "performance_metrics.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Class for calculating performance metrics for investment strategies.
    Implements common financial performance metrics like Sharpe ratio,
    Sortino ratio, maximum drawdown, etc.
    """
    
    def __init__(self):
        """Initialize PerformanceMetrics class."""
        logger.info("PerformanceMetrics initialized")
    
    def calculate_returns(self, returns, annualize=True, trading_days=252):
        """
        Calculate total and annualized returns from a series of returns.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        annualize : bool, default=True
            Whether to annualize the return.
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        dict
            Total and annualized returns.
        """
        try:
            logger.info("Calculating returns")
            
            # Calculate total return
            total_return = (1 + returns).prod() - 1
            
            # Calculate annualized return if requested
            if annualize:
                n_periods = len(returns)
                if n_periods > 0:
                    # Annualize based on trading days
                    annualized_return = (1 + total_return) ** (trading_days / n_periods) - 1
                else:
                    annualized_return = 0
            else:
                annualized_return = None
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return
            }
        
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise
    
    def calculate_volatility(self, returns, annualize=True, trading_days=252):
        """
        Calculate return volatility.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        annualize : bool, default=True
            Whether to annualize the volatility.
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        float
            Volatility of returns.
        """
        try:
            logger.info("Calculating volatility")
            
            # Calculate standard deviation of returns
            volatility = np.std(returns, ddof=1)
            
            # Annualize if requested
            if annualize:
                volatility = volatility * np.sqrt(trading_days)
            
            return volatility
        
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            raise
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0, trading_days=252):
        """
        Calculate Sharpe ratio.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        risk_free_rate : float, default=0
            Risk-free rate (annualized).
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        float
            Sharpe ratio.
        """
        try:
            logger.info("Calculating Sharpe ratio")
            
            # Calculate excess returns
            excess_returns = returns - risk_free_rate / trading_days
            
            # Calculate annualized mean and standard deviation
            mean_excess_return = np.mean(excess_returns) * trading_days
            std_excess_return = np.std(excess_returns, ddof=1) * np.sqrt(trading_days)
            
            # Calculate Sharpe ratio
            if std_excess_return > 0:
                sharpe_ratio = mean_excess_return / std_excess_return
            else:
                sharpe_ratio = np.nan
            
            return sharpe_ratio
        
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            raise
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0, trading_days=252):
        """
        Calculate Sortino ratio.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        risk_free_rate : float, default=0
            Risk-free rate (annualized).
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        float
            Sortino ratio.
        """
        try:
            logger.info("Calculating Sortino ratio")
            
            # Calculate excess returns
            excess_returns = returns - risk_free_rate / trading_days
            
            # Calculate annualized mean
            mean_excess_return = np.mean(excess_returns) * trading_days
            
            # Calculate downside deviation
            downside_returns = np.minimum(excess_returns, 0)
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(trading_days)
            
            # Calculate Sortino ratio
            if downside_deviation > 0:
                sortino_ratio = mean_excess_return / downside_deviation
            else:
                sortino_ratio = np.nan
            
            return sortino_ratio
        
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            raise
    
    def calculate_maximum_drawdown(self, returns):
        """
        Calculate maximum drawdown.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        
        Returns
        -------
        float
            Maximum drawdown.
        """
        try:
            logger.info("Calculating maximum drawdown")
            
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)
            
            # Calculate drawdown
            drawdown = (cum_returns / running_max) - 1
            
            # Calculate maximum drawdown
            max_drawdown = np.min(drawdown)
            
            return max_drawdown
        
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            raise
    
    def calculate_calmar_ratio(self, returns, trading_days=252):
        """
        Calculate Calmar ratio.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        float
            Calmar ratio.
        """
        try:
            logger.info("Calculating Calmar ratio")
            
            # Calculate annualized return
            returns_result = self.calculate_returns(returns, annualize=True, trading_days=trading_days)
            annualized_return = returns_result['annualized_return']
            
            # Calculate maximum drawdown
            max_drawdown = self.calculate_maximum_drawdown(returns)
            
            # Calculate Calmar ratio
            if max_drawdown < 0:
                calmar_ratio = annualized_return / abs(max_drawdown)
            else:
                calmar_ratio = np.nan
            
            return calmar_ratio
        
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            raise
    
    def calculate_ulcer_index(self, returns):
        """
        Calculate Ulcer index.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of period returns.
        
        Returns
        -------
        float
            Ulcer index.
        """
        try:
            logger.info("Calculating Ulcer index")
            
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)
            
            # Calculate percentage drawdown
            pct_drawdown = (cum_returns / running_max) - 1
            
            # Calculate squared drawdowns
            squared_drawdowns = pct_drawdown ** 2
            
            # Calculate Ulcer index
            ulcer_index = np.sqrt(np.mean(squared_drawdowns))
            
            return ulcer_index
        
        except Exception as e:
            logger.error(f"Error calculating Ulcer index: {e}")
            raise
    
    def calculate_beta(self, returns, market_returns):
        """
        Calculate beta of returns against market returns.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of strategy returns.
        market_returns : pandas.Series or numpy.ndarray
            Series of market returns.
        
        Returns
        -------
        float
            Beta.
        """
        try:
            logger.info("Calculating beta")
            
            if market_returns is None:
                logger.warning("Market returns not provided, cannot calculate beta")
                return np.nan
                
            # Check if arrays have the same length
            if len(returns) != len(market_returns):
                logger.warning(f"Returns and market returns have different lengths: {len(returns)} vs {len(market_returns)}")
                logger.warning("Cannot calculate beta with misaligned data")
                return np.nan
            
            # Calculate covariance of returns with market returns
            covariance = np.cov(returns, market_returns)[0, 1]
            
            # Calculate variance of market returns
            market_variance = np.var(market_returns, ddof=1)
            
            # Calculate beta
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = np.nan
            
            return beta
        
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return np.nan
    
    def calculate_alpha(self, returns, market_returns, risk_free_rate=0, trading_days=252):
        """
        Calculate alpha of returns against market returns.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of strategy returns.
        market_returns : pandas.Series or numpy.ndarray
            Series of market returns.
        risk_free_rate : float, default=0
            Risk-free rate (annualized).
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        float
            Alpha.
        """
        try:
            logger.info("Calculating alpha")
            
            if market_returns is None:
                logger.warning("Market returns not provided, cannot calculate alpha")
                return np.nan
                
            # Check if arrays have the same length
            if len(returns) != len(market_returns):
                logger.warning(f"Returns and market returns have different lengths: {len(returns)} vs {len(market_returns)}")
                logger.warning("Cannot calculate alpha with misaligned data")
                return np.nan
            
            # Calculate beta
            beta = self.calculate_beta(returns, market_returns)
            
            if np.isnan(beta):
                return np.nan
            
            # Calculate annualized returns
            returns_result = self.calculate_returns(returns, annualize=True, trading_days=trading_days)
            annualized_return = returns_result['annualized_return']
            
            market_returns_result = self.calculate_returns(market_returns, annualize=True, trading_days=trading_days)
            annualized_market_return = market_returns_result['annualized_return']
            
            # Calculate alpha
            alpha = annualized_return - (risk_free_rate + beta * (annualized_market_return - risk_free_rate))
            
            return alpha
        
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return np.nan
    
    def calculate_information_ratio(self, returns, benchmark_returns, trading_days=252):
        """
        Calculate information ratio.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of strategy returns.
        benchmark_returns : pandas.Series or numpy.ndarray
            Series of benchmark returns.
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        float
            Information ratio.
        """
        try:
            logger.info("Calculating information ratio")
            
            # Calculate tracking error
            tracking_error = np.std(returns - benchmark_returns, ddof=1) * np.sqrt(trading_days)
            
            # Calculate annualized active return
            active_return = np.mean(returns - benchmark_returns) * trading_days
            
            # Calculate information ratio
            if tracking_error > 0:
                information_ratio = active_return / tracking_error
            else:
                information_ratio = np.nan
            
            return information_ratio
        
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            raise
    
    def calculate_all_metrics(self, returns, market_returns=None, benchmark_returns=None, 
                            risk_free_rate=0, trading_days=252):
        """
        Calculate all performance metrics.
        
        Parameters
        ----------
        returns : pandas.Series or numpy.ndarray
            Series of strategy returns.
        market_returns : pandas.Series or numpy.ndarray, optional
            Series of market returns.
        benchmark_returns : pandas.Series or numpy.ndarray, optional
            Series of benchmark returns.
        risk_free_rate : float, default=0
            Risk-free rate (annualized).
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        dict
            All performance metrics.
        """
        try:
            logger.info("Calculating all performance metrics")
            
            # Calculate returns
            returns_result = self.calculate_returns(returns, annualize=True, trading_days=trading_days)
            
            # Calculate volatility
            volatility = self.calculate_volatility(returns, annualize=True, trading_days=trading_days)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate, trading_days)
            
            # Calculate Sortino ratio
            sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate, trading_days)
            
            # Calculate maximum drawdown
            max_drawdown = self.calculate_maximum_drawdown(returns)
            
            # Calculate Calmar ratio
            calmar_ratio = self.calculate_calmar_ratio(returns, trading_days)
            
            # Calculate Ulcer index
            ulcer_index = self.calculate_ulcer_index(returns)
            
            # Create results dictionary
            results = {
                'total_return': returns_result['total_return'],
                'annualized_return': returns_result['annualized_return'],
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'ulcer_index': ulcer_index
            }
            
            # Calculate market-related metrics if market returns are provided
            if market_returns is not None:
                # Calculate beta
                beta = self.calculate_beta(returns, market_returns)
                results['beta'] = beta
                
                # Calculate alpha
                alpha = self.calculate_alpha(returns, market_returns, risk_free_rate, trading_days)
                results['alpha'] = alpha
            
            # Calculate benchmark-related metrics if benchmark returns are provided
            if benchmark_returns is not None:
                # Calculate information ratio
                information_ratio = self.calculate_information_ratio(returns, benchmark_returns, trading_days)
                results['information_ratio'] = information_ratio
            
            return results
        
        except Exception as e:
            logger.error(f"Error calculating all metrics: {e}")
            raise
    
    def evaluate_strategy(self, strategy_returns, market_returns=None, benchmark_returns=None,
                    risk_free_rate=0, trading_days=252):
        """
        Evaluate a strategy using all performance metrics.
        
        Parameters
        ----------
        strategy_returns : pandas.DataFrame
            DataFrame with strategy returns. Should have 'Date' and 'Daily_Return' columns.
        market_returns : pandas.DataFrame, optional
            DataFrame with market returns. Should have 'Date' and 'Daily_Return' columns.
        benchmark_returns : pandas.DataFrame, optional
            DataFrame with benchmark returns. Should have 'Date' and 'Daily_Return' columns.
        risk_free_rate : float, default=0
            Risk-free rate (annualized).
        trading_days : int, default=252
            Number of trading days in a year.
        
        Returns
        -------
        pandas.DataFrame
            Performance metrics for each strategy.
        """
        try:
            logger.info("Evaluating strategy")
            
            # Check if strategy_returns has multiple strategies
            if 'Strategy' in strategy_returns.columns:
                # Evaluate each strategy separately
                strategies = strategy_returns['Strategy'].unique()
                results = []
                
                for strategy in strategies:
                    # Get returns for this strategy
                    strategy_data = strategy_returns[strategy_returns['Strategy'] == strategy]
                    
                    # Align dates with market/benchmark returns if provided
                    if market_returns is not None:
                        # Get common dates between strategy and market
                        common_dates = set(strategy_data['Date']).intersection(set(market_returns['Date']))
                        
                        if not common_dates:
                            logger.warning(f"No common dates between strategy {strategy} and market returns")
                            continue
                        
                        # Filter data to common dates
                        strategy_data = strategy_data[strategy_data['Date'].isin(common_dates)]
                        market_data = market_returns[market_returns['Date'].isin(common_dates)]
                        
                        # Sort by date to ensure alignment
                        strategy_data = strategy_data.sort_values('Date')
                        market_data = market_data.sort_values('Date')
                        
                        # Calculate metrics with aligned data
                        metrics = self.calculate_all_metrics(
                            strategy_data['Daily_Return'].values,
                            market_data['Daily_Return'].values,
                            benchmark_returns['Daily_Return'].values if benchmark_returns is not None else None,
                            risk_free_rate,
                            trading_days
                        )
                    else:
                        # Calculate metrics without market returns
                        metrics = self.calculate_all_metrics(
                            strategy_data['Daily_Return'].values,
                            None,
                            benchmark_returns['Daily_Return'].values if benchmark_returns is not None else None,
                            risk_free_rate,
                            trading_days
                        )
                    
                    # Add strategy name
                    metrics['Strategy'] = strategy
                    
                    results.append(metrics)
                
                # Combine results
                if results:
                    return pd.DataFrame(results)
                else:
                    logger.warning("No valid strategies to evaluate")
                    return pd.DataFrame()
                    
            else:
                # Calculate all metrics for a single strategy
                # Similar alignment logic would apply here too
                if market_returns is not None:
                    # Get common dates
                    common_dates = set(strategy_returns['Date']).intersection(set(market_returns['Date']))
                    
                    if not common_dates:
                        logger.warning("No common dates between strategy and market returns")
                        # Calculate without market returns
                        metrics = self.calculate_all_metrics(
                            strategy_returns['Daily_Return'].values,
                            None,
                            benchmark_returns['Daily_Return'].values if benchmark_returns is not None else None,
                            risk_free_rate,
                            trading_days
                        )
                    else:
                        # Filter to common dates
                        strat_data = strategy_returns[strategy_returns['Date'].isin(common_dates)].sort_values('Date')
                        market_data = market_returns[market_returns['Date'].isin(common_dates)].sort_values('Date')
                        
                        metrics = self.calculate_all_metrics(
                            strat_data['Daily_Return'].values,
                            market_data['Daily_Return'].values,
                            benchmark_returns['Daily_Return'].values if benchmark_returns is not None else None,
                            risk_free_rate,
                            trading_days
                        )
                else:
                    metrics = self.calculate_all_metrics(
                        strategy_returns['Daily_Return'].values,
                        None,
                        benchmark_returns['Daily_Return'].values if benchmark_returns is not None else None,
                        risk_free_rate,
                        trading_days
                    )
                    
                # Convert to DataFrame
                return pd.DataFrame([metrics])
        
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the PerformanceMetrics class
    performance_metrics = PerformanceMetrics()
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    
    # Strategy returns (slightly better than market)
    strategy_returns = pd.DataFrame({
        'Date': dates,
        'Strategy': 'Sample Strategy',
        'Daily_Return': np.random.normal(0.0008, 0.01, len(dates))
    })
    
    # Market returns
    market_returns = pd.DataFrame({
        'Date': dates,
        'Daily_Return': np.random.normal(0.0005, 0.012, len(dates))
    })
    
    # Evaluate strategy
    evaluation = performance_metrics.evaluate_strategy(
        strategy_returns, 
        market_returns=market_returns,
        risk_free_rate=0.02  # 2% annualized risk-free rate
    )
    
    print("Strategy Evaluation:")
    print(evaluation)