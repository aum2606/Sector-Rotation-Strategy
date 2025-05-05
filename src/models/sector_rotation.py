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
        logging.FileHandler(logs_dir / "sector_rotation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SectorRotationModel:
    """
    Class for implementing sector rotation strategies based on market indicators
    and sector performance metrics.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize SectorRotationModel class.
        
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
        
        logger.info(f"SectorRotationModel initialized with data directory: {self.data_dir}")
    
    def load_processed_data(self):
        """
        Load processed portfolio, price, sector, and macro indicator data.
        
        Returns
        -------
        tuple
            Portfolio data, price data, sector indicators, and macro indicators.
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
            
            # Load sector indicators
            sector_path = self.processed_dir / "sector_indicators.csv"
            logger.info(f"Loading sector indicators from {sector_path}")
            sector_df = pd.read_csv(sector_path)
            sector_df['Date'] = pd.to_datetime(sector_df['Date'])
            
            # Load macro indicators
            macro_path = self.processed_dir / "macro_indicators.csv"
            logger.info(f"Loading macro indicators from {macro_path}")
            macro_df = pd.read_csv(macro_path)
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            
            logger.info("Processed data loaded successfully")
            return portfolio_df, price_df, sector_df, macro_df
        
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def momentum_strategy(self, sector_df, n_sectors=3, rebalance_frequency='M',
                         start_date=None, end_date=None):
        """
        Implement a momentum-based sector rotation strategy.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector indicators data.
        n_sectors : int, default=3
            Number of top sectors to invest in.
        rebalance_frequency : str, default='M'
            Frequency for rebalancing: 'D' (daily), 'W' (weekly), 'M' (monthly), 
            'Q' (quarterly), or 'Y' (yearly).
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of momentum strategy.
        """
        try:
            logger.info(f"Implementing momentum strategy with top {n_sectors} sectors and "
                      f"{rebalance_frequency} rebalancing")
            
            # Filter data by date if specified
            if start_date is not None:
                sector_df = sector_df[sector_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                sector_df = sector_df[sector_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = sector_df['Date'].unique()
            dates.sort()
            
            # Calculate daily returns
            daily_returns = []
            current_sectors = None  # Will be initialized at first rebalance
            
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
                    # Get data for this date
                    date_data = sector_df[sector_df['Date'] == date]
                    
                    # Rank sectors by momentum score
                    if 'Momentum_Score' in date_data.columns:
                        momentum_ranking = date_data.sort_values('Momentum_Score', ascending=False)
                    else:
                        # Fallback to return-based momentum
                        momentum_ranking = date_data.sort_values('Momentum_6m', ascending=False)
                    
                    # Select top N sectors
                    top_sectors = momentum_ranking['Sector'].unique()[:n_sectors]
                    
                    # Equal weight allocation to top sectors
                    if len(top_sectors) > 0:
                        sector_weight = 1.0 / len(top_sectors)
                        current_sectors = pd.DataFrame({
                            'Sector': top_sectors,
                            'Weight': sector_weight
                        })
                        
                        selected_sectors = ', '.join(top_sectors)
                        logger.debug(f"Rebalanced on {date}: Selected sectors {selected_sectors}")
                    
                    # Move to next rebalance date
                    next_rebalance_idx += 1
                
                # Skip if we haven't initialized sectors yet
                if current_sectors is None:
                    continue
                
                # Get returns for this date
                date_returns = sector_df[sector_df['Date'] == date]
                
                # Merge with current sectors
                merged = pd.merge(date_returns, current_sectors, on='Sector', how='inner')
                
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Momentum Sector Rotation',
                        'Daily_Return': portfolio_return,
                        'Sectors_Count': len(merged),
                        'Selected_Sectors': ', '.join(merged['Sector'].tolist())
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Momentum strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing momentum strategy: {e}")
            raise
    
    def relative_strength_strategy(self, sector_df, n_sectors=3, rebalance_frequency='M',
                                 lookback_period=60, start_date=None, end_date=None):
        """
        Implement a relative strength sector rotation strategy.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector indicators data.
        n_sectors : int, default=3
            Number of top sectors to invest in.
        rebalance_frequency : str, default='M'
            Frequency for rebalancing: 'D' (daily), 'W' (weekly), 'M' (monthly), 
            'Q' (quarterly), or 'Y' (yearly).
        lookback_period : int, default=60
            Number of days to look back for calculating relative strength.
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of relative strength strategy.
        """
        try:
            logger.info(f"Implementing relative strength strategy with top {n_sectors} sectors "
                      f"and {rebalance_frequency} rebalancing")
            
            # Filter data by date if specified
            if start_date is not None:
                sector_df = sector_df[sector_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                sector_df = sector_df[sector_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = sector_df['Date'].unique()
            dates.sort()
            
            # Calculate daily returns
            daily_returns = []
            current_sectors = None  # Will be initialized at first rebalance
            
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
                    # For relative strength, we need historical data
                    # Get index of current date in dates array
                    date_idx = np.where(dates == date)[0][0]
                    
                    # Check if we have enough historical data
                    if date_idx >= lookback_period:
                        # Get start date for lookback period
                        start_idx = date_idx - lookback_period
                        lookback_start = dates[start_idx]
                        
                        # Get data for lookback period
                        lookback_data = sector_df[(sector_df['Date'] >= lookback_start) & 
                                                 (sector_df['Date'] <= date)]
                        
                        # Calculate relative strength for each sector
                        relative_strength = {}
                        
                        for sector in lookback_data['Sector'].unique():
                            sector_data = lookback_data[lookback_data['Sector'] == sector]
                            
                            # Calculate cumulative return for this sector
                            if len(sector_data) > 0:
                                sector_returns = sector_data['Returns'].values
                                cumulative_return = (1 + sector_returns).cumprod()[-1] - 1
                                relative_strength[sector] = cumulative_return
                        
                        # Rank sectors by relative strength
                        ranked_sectors = sorted(relative_strength.items(), 
                                               key=lambda x: x[1], reverse=True)
                        
                        # Select top N sectors
                        top_sectors = [sector for sector, _ in ranked_sectors[:n_sectors]]
                        
                        # Equal weight allocation to top sectors
                        if len(top_sectors) > 0:
                            sector_weight = 1.0 / len(top_sectors)
                            current_sectors = pd.DataFrame({
                                'Sector': top_sectors,
                                'Weight': sector_weight
                            })
                            
                            selected_sectors = ', '.join(top_sectors)
                            logger.debug(f"Rebalanced on {date}: Selected sectors {selected_sectors}")
                    
                    # Move to next rebalance date
                    next_rebalance_idx += 1
                
                # Skip if we haven't initialized sectors yet
                if current_sectors is None:
                    continue
                
                # Get returns for this date
                date_returns = sector_df[sector_df['Date'] == date]
                
                # Merge with current sectors
                merged = pd.merge(date_returns, current_sectors, on='Sector', how='inner')
                
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Relative Strength Sector Rotation',
                        'Daily_Return': portfolio_return,
                        'Sectors_Count': len(merged),
                        'Selected_Sectors': ', '.join(merged['Sector'].tolist())
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Relative strength strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing relative strength strategy: {e}")
            raise
    
    def economic_cycle_strategy(self, sector_df, macro_df, n_sectors=3, 
                              rebalance_frequency='M', start_date=None, end_date=None):
        """
        Implement a sector rotation strategy based on economic cycle indicators.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector indicators data.
        macro_df : pandas.DataFrame
            Macro economic data with cycle indicators.
        n_sectors : int, default=3
            Number of top sectors to invest in.
        rebalance_frequency : str, default='M'
            Frequency for rebalancing: 'D' (daily), 'W' (weekly), 'M' (monthly), 
            'Q' (quarterly), or 'Y' (yearly).
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of economic cycle strategy.
        """
        try:
            logger.info(f"Implementing economic cycle strategy with top {n_sectors} sectors "
                      f"and {rebalance_frequency} rebalancing")
            
            # Check if Economic_Cycle is in macro_df
            if 'Economic_Cycle' not in macro_df.columns:
                logger.error("Economic_Cycle not found in macro data")
                raise ValueError("Economic_Cycle column required for economic cycle strategy")
            
            # Filter data by date if specified
            if start_date is not None:
                sector_df = sector_df[sector_df['Date'] >= pd.to_datetime(start_date)]
                macro_df = macro_df[macro_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                sector_df = sector_df[sector_df['Date'] <= pd.to_datetime(end_date)]
                macro_df = macro_df[macro_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = sector_df['Date'].unique()
            dates.sort()
            
            # Define sector preferences for each economic cycle phase
            # 0: Contraction, 1: Early Recovery, 2: Expansion, 3: Late Cycle
            cycle_preferences = {
                0: ['Health Care', 'Utilities', 'Consumer Staples'],  # Contraction
                1: ['Consumer Discretionary', 'Financials', 'Materials'],  # Early Recovery
                2: ['Information Technology', 'Industrials', 'Communication Services'],  # Expansion
                3: ['Energy', 'Materials', 'Real Estate']  # Late Cycle
            }
            
            # Calculate daily returns
            daily_returns = []
            current_sectors = None  # Will be initialized at first rebalance
            
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
                    # Get economic cycle phase for this date
                    date_macro = macro_df[macro_df['Date'] == date]
                    
                    if not date_macro.empty:
                        economic_cycle = date_macro['Economic_Cycle'].values[0]
                        
                        # Get preferred sectors for this cycle
                        preferred_sectors = cycle_preferences.get(economic_cycle, [])
                        
                        # Adjust if we don't have enough preferred sectors
                        available_sectors = sector_df[sector_df['Date'] == date]['Sector'].unique()
                        preferred_sectors = [s for s in preferred_sectors if s in available_sectors]
                        
                        # If we need more sectors, add based on momentum
                        if len(preferred_sectors) < n_sectors:
                            # Get momentum scores for remaining sectors
                            remaining_sectors = [s for s in available_sectors if s not in preferred_sectors]
                            date_data = sector_df[(sector_df['Date'] == date) & 
                                                (sector_df['Sector'].isin(remaining_sectors))]
                            
                            if 'Momentum_Score' in date_data.columns:
                                momentum_ranking = date_data.sort_values('Momentum_Score', ascending=False)
                            else:
                                # Fallback to return-based momentum
                                momentum_ranking = date_data.sort_values('Momentum_6m', ascending=False)
                            
                            # Add top momentum sectors to reach n_sectors
                            additional_sectors = momentum_ranking['Sector'].unique()[:n_sectors - len(preferred_sectors)]
                            preferred_sectors = list(preferred_sectors) + list(additional_sectors)
                        
                        # If more sectors than needed, take top n_sectors
                        if len(preferred_sectors) > n_sectors:
                            preferred_sectors = preferred_sectors[:n_sectors]
                        
                        # Equal weight allocation to preferred sectors
                        if len(preferred_sectors) > 0:
                            sector_weight = 1.0 / len(preferred_sectors)
                            current_sectors = pd.DataFrame({
                                'Sector': preferred_sectors,
                                'Weight': sector_weight
                            })
                            
                            selected_sectors = ', '.join(preferred_sectors)
                            logger.debug(f"Rebalanced on {date}: Economic cycle {economic_cycle}, "
                                       f"Selected sectors {selected_sectors}")
                    
                    # Move to next rebalance date
                    next_rebalance_idx += 1
                
                # Skip if we haven't initialized sectors yet
                if current_sectors is None:
                    continue
                
                # Get returns for this date
                date_returns = sector_df[sector_df['Date'] == date]
                
                # Merge with current sectors
                merged = pd.merge(date_returns, current_sectors, on='Sector', how='inner')
                
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Economic Cycle Sector Rotation',
                        'Daily_Return': portfolio_return,
                        'Sectors_Count': len(merged),
                        'Selected_Sectors': ', '.join(merged['Sector'].tolist())
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Economic cycle strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing economic cycle strategy: {e}")
            raise
    
    def combined_strategy(self, sector_df, macro_df, n_sectors=3, rebalance_frequency='M',
                        weights={'momentum': 0.4, 'relative_strength': 0.4, 'economic_cycle': 0.2},
                        start_date=None, end_date=None):
        """
        Implement a combined sector rotation strategy using multiple factor scores.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector indicators data.
        macro_df : pandas.DataFrame
            Macro economic data with cycle indicators.
        n_sectors : int, default=3
            Number of top sectors to invest in.
        rebalance_frequency : str, default='M'
            Frequency for rebalancing: 'D' (daily), 'W' (weekly), 'M' (monthly), 
            'Q' (quarterly), or 'Y' (yearly).
        weights : dict, default={'momentum': 0.4, 'relative_strength': 0.4, 'economic_cycle': 0.2}
            Weights for each strategy component.
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Performance of combined strategy.
        """
        try:
            logger.info(f"Implementing combined strategy with top {n_sectors} sectors "
                      f"and {rebalance_frequency} rebalancing")
            
            # Filter data by date if specified
            if start_date is not None:
                sector_df = sector_df[sector_df['Date'] >= pd.to_datetime(start_date)]
                macro_df = macro_df[macro_df['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                sector_df = sector_df[sector_df['Date'] <= pd.to_datetime(end_date)]
                macro_df = macro_df[macro_df['Date'] <= pd.to_datetime(end_date)]
            
            # Get unique dates
            dates = sector_df['Date'].unique()
            dates.sort()
            
            # Define sector preferences for each economic cycle phase
            # 0: Contraction, 1: Early Recovery, 2: Expansion, 3: Late Cycle
            cycle_preferences = {
                0: ['Health Care', 'Utilities', 'Consumer Staples'],  # Contraction
                1: ['Consumer Discretionary', 'Financials', 'Materials'],  # Early Recovery
                2: ['Information Technology', 'Industrials', 'Communication Services'],  # Expansion
                3: ['Energy', 'Materials', 'Real Estate']  # Late Cycle
            }
            
            # Calculate daily returns
            daily_returns = []
            current_sectors = None  # Will be initialized at first rebalance
            
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
            lookback_period = 60  # For relative strength
            
            for date in dates:
                # Check if we need to rebalance
                if next_rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[next_rebalance_idx]:
                    # Get data for this date
                    date_data = sector_df[sector_df['Date'] == date]
                    available_sectors = date_data['Sector'].unique()
                    
                    # Skip if no sectors available
                    if len(available_sectors) == 0:
                        next_rebalance_idx += 1
                        continue
                    
                    # Calculate scores for each strategy
                    sector_scores = {sector: 0 for sector in available_sectors}
                    
                    # 1. Momentum score
                    if 'Momentum_Score' in date_data.columns:
                        for _, row in date_data.iterrows():
                            sector = row['Sector']
                            momentum_score = row['Momentum_Score']
                            sector_scores[sector] += weights.get('momentum', 0.4) * momentum_score
                    
                    # 2. Relative strength score
                    # For relative strength, we need historical data
                    date_idx = np.where(dates == date)[0][0]
                    if date_idx >= lookback_period:
                        # Get start date for lookback period
                        start_idx = date_idx - lookback_period
                        lookback_start = dates[start_idx]
                        
                        # Get data for lookback period
                        lookback_data = sector_df[(sector_df['Date'] >= lookback_start) & 
                                                 (sector_df['Date'] <= date)]
                        
                        # Calculate relative strength for each sector
                        relative_strength = {}
                        
                        for sector in lookback_data['Sector'].unique():
                            sector_data = lookback_data[lookback_data['Sector'] == sector]
                            
                            if len(sector_data) > 0:
                                sector_returns = sector_data['Returns'].values
                                cumulative_return = (1 + sector_returns).cumprod()[-1] - 1
                                relative_strength[sector] = cumulative_return
                        
                        # Normalize relative strength scores
                        if relative_strength:
                            min_rs = min(relative_strength.values())
                            max_rs = max(relative_strength.values())
                            
                            if max_rs > min_rs:  # Avoid division by zero
                                for sector, rs in relative_strength.items():
                                    normalized_rs = (rs - min_rs) / (max_rs - min_rs)
                                    sector_scores[sector] += weights.get('relative_strength', 0.4) * normalized_rs
                    
                    # 3. Economic cycle score
                    date_macro = macro_df[macro_df['Date'] == date]
                    
                    if not date_macro.empty and 'Economic_Cycle' in date_macro.columns:
                        economic_cycle = date_macro['Economic_Cycle'].values[0]
                        
                        # Get preferred sectors for this cycle
                        preferred_sectors = cycle_preferences.get(economic_cycle, [])
                        
                        # Add economic cycle score
                        for sector in available_sectors:
                            if sector in preferred_sectors:
                                sector_scores[sector] += weights.get('economic_cycle', 0.2)
                    
                    # Rank sectors by combined score
                    ranked_sectors = sorted(sector_scores.items(), 
                                           key=lambda x: x[1], reverse=True)
                    
                    # Select top N sectors
                    top_sectors = [sector for sector, _ in ranked_sectors[:n_sectors]]
                    
                    # Equal weight allocation to top sectors
                    if len(top_sectors) > 0:
                        sector_weight = 1.0 / len(top_sectors)
                        current_sectors = pd.DataFrame({
                            'Sector': top_sectors,
                            'Weight': sector_weight
                        })
                        
                        selected_sectors = ', '.join(top_sectors)
                        logger.debug(f"Rebalanced on {date}: Selected sectors {selected_sectors}")
                    
                    # Move to next rebalance date
                    next_rebalance_idx += 1
                
                # Skip if we haven't initialized sectors yet
                if current_sectors is None:
                    continue
                
                # Get returns for this date
                date_returns = sector_df[sector_df['Date'] == date]
                
                # Merge with current sectors
                merged = pd.merge(date_returns, current_sectors, on='Sector', how='inner')
                
                if not merged.empty:
                    # Calculate portfolio return for this date
                    portfolio_return = (merged['Returns'] * merged['Weight']).sum()
                    
                    daily_returns.append({
                        'Date': date,
                        'Strategy': 'Combined Sector Rotation',
                        'Daily_Return': portfolio_return,
                        'Sectors_Count': len(merged),
                        'Selected_Sectors': ', '.join(merged['Sector'].tolist())
                    })
            
            # Create dataframe with results
            results_df = pd.DataFrame(daily_returns)
            
            # Calculate cumulative returns
            results_df['Cumulative_Return'] = (1 + results_df['Daily_Return']).cumprod() - 1
            
            logger.info("Combined strategy implemented successfully")
            return results_df
        
        except Exception as e:
            logger.error(f"Error implementing combined strategy: {e}")
            raise
    
    def run_all_strategies(self, start_date=None, end_date=None):
        """
        Run all sector rotation strategies and combine results.
        
        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for the backtest.
        end_date : str or datetime, optional
            End date for the backtest.
        
        Returns
        -------
        pandas.DataFrame
            Combined performance of all strategies.
        """
        try:
            logger.info("Running all sector rotation strategies")
            
            # Load data
            portfolio_df, price_df, sector_df, macro_df = self.load_processed_data()
            
            # Run momentum strategy
            momentum_results = self.momentum_strategy(
                sector_df, n_sectors=3, rebalance_frequency='M', 
                start_date=start_date, end_date=end_date
            )
            
            # Run relative strength strategy
            relative_strength_results = self.relative_strength_strategy(
                sector_df, n_sectors=3, rebalance_frequency='M',
                start_date=start_date, end_date=end_date
            )
            
            # Run economic cycle strategy
            economic_cycle_results = self.economic_cycle_strategy(
                sector_df, macro_df, n_sectors=3, rebalance_frequency='M',
                start_date=start_date, end_date=end_date
            )
            
            # Run combined strategy
            combined_results = self.combined_strategy(
                sector_df, macro_df, n_sectors=3, rebalance_frequency='M',
                start_date=start_date, end_date=end_date
            )
            
            # Combine results
            all_results = pd.concat([
                momentum_results,
                relative_strength_results,
                economic_cycle_results,
                combined_results
            ])
            
            # Save results
            output_path = self.processed_dir / "sector_rotation_results.csv"
            all_results.to_csv(output_path, index=False)
            logger.info(f"Sector rotation results saved to {output_path}")
            
            return all_results
        
        except Exception as e:
            logger.error(f"Error running all sector rotation strategies: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the SectorRotationModel class
    sector_rotation = SectorRotationModel()
    
    # Run all strategies
    results = sector_rotation.run_all_strategies()
    
    # Print sample results
    print("\nSector Rotation Results Sample:")
    print(results.head())
    
    # Get final performance for each strategy
    final_performance = results.groupby('Strategy')['Cumulative_Return'].last()
    print("\nFinal Performance:")
    print(final_performance)