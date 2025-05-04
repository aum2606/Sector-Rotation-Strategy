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
        logging.FileHandler(logs_dir / "sector_indicators.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SectorIndicators:
    """Class for generating sector-specific indicators."""
    
    def __init__(self, data_dir=None):
        """
        Initialize SectorIndicators class.
        
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
        
        logger.info(f"SectorIndicators initialized with data directory: {self.data_dir}")
    
    def load_processed_data(self):
        """
        Load processed sector, price, and macro data.
        
        Returns
        -------
        tuple
            Sector data, price data, and macro indicators.
        """
        try:
            #load sector data
            sector_path = self.processed_dir / "sector_data.csv"
            logger.info(f"Loading processed sector data from {sector_path}")
            sector_df = pd.read_csv(sector_path)
            sector_df['Date'] = pd.to_datetime(sector_df['Date'])
            
            #load price data
            price_path = self.processed_dir / "price_data.csv"
            logger.info(f"Loading processed price data from {price_path}")
            price_df = pd.read_csv(price_path)
            price_df['Date'] = pd.to_datetime(price_df['Date'])

            #load macro indicators if available
            try:
                macro_path = self.processed_dir / "macro_indicators.csv"
                logger.info(f"Loading macro indicators from {macro_path}")
                macro_df = pd.read_csv(macro_path)
                macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            except FileNotFoundError:
                logger.warning("Macro indicators not found, loading basic macro data instead")
                macro_path = self.processed_dir / "macro_data.csv"
                macro_df = pd.read_csv(macro_path)
                macro_df['Date'] = pd.to_datetime(macro_df['Date'])

            logger.info("Preprocessed data loaded successfully")
            return sector_df,price_df,macro_df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise

    def calculate_relative_strength(self,sector_df,market_df):
        """
        Calculate relative strength of sectors compared to the market.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector-level data.
        market_df : pandas.DataFrame
            Market-level data.
        
        Returns
        -------
        pandas.DataFrame
            Sector data with relative strength indicators.
        """
        try:
            logger.info("calculating sector relative strength")
            
            # Merge sector data with market data
            # First, extract just the relevant market columns
            market_subset = market_df[['Date', 'Returns']].rename(columns={'Returns': 'Market_Returns'})
            
            #merge with sector data
            merged_df = pd.merge(sector_df,market_subset,on='Date',how='left')
            
            #calculate relative strength
            merged_df['Relative_Strength'] = merged_df['Returns'] - merged_df['Market_Returns']
            
            # Calculate 20-day rolling relative strength
            merged_df = merged_df.sort_values(['Sector', 'Date'])
            merged_df['Relative_Strength_20d'] = merged_df.groupby('Sector')['Relative_Strength'].transform(
                lambda x: x.rolling(window=20).mean()
            )
            
            # Calculate 50-day rolling relative strength
            merged_df['Relative_Strength_50d'] = merged_df.groupby('Sector')['Relative_Strength'].transform(
                lambda x: x.rolling(window=50).mean()
            )
            
            # Relative strength trend (short-term vs long-term)
            merged_df['RS_Trend'] = merged_df['Relative_Strength_20d'] - merged_df['Relative_Strength_50d']
            
            logger.info("Sector relative strength calculated successfully")
            return merged_df
        
        except Exception as e:
            logger.error(f"Error calculating sector relative strength: {e}")
            raise
    
    
    def calculate_momentum_indicators(self, sector_df):
        """
        Calculate momentum indicators for each sector.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector-level data.
        
        Returns
        -------
        pandas.DataFrame
            Sector data with momentum indicators.
        """
        try:
            logger.info("Calculating sector momentum indicators")
            
            #sort data
            sector_df = sector_df.sort_values(['Sector','Date'])
            
            # Calculate momentum over different time periods
            # 1-month momentum (approx. 20 trading days)
            sector_df['Momentum_1m'] = sector_df.groupby('Sector')['Adjusted'].transform(
                lambda x: x.pct_change(20)
            )
            
            #3-month momentum
            sector_df['Momentum_3m'] = sector_df.groupby('Sector')['Adjusted'].transform(
                lambda x: x.pct_change(60)
            )
            
            # 6-month momentum (approx. 125 trading days)
            sector_df['Momentum_6m'] = sector_df.groupby('Sector')['Adjusted'].transform(
                lambda x: x.pct_change(125)
            )
            
            # 12-month momentum (approx. 252 trading days)
            sector_df['Momentum_12m'] = sector_df.groupby('Sector')['Adjusted'].transform(
                lambda x: x.pct_change(252)
            )
            
            # Calculate weighted momentum score (higher weight to more recent momentum)
            sector_df['Momentum_Score'] = (
                sector_df['Momentum_1m'] * 0.4 +
                sector_df['Momentum_3m'] * 0.3 +
                sector_df['Momentum_6m'] * 0.2 +
                sector_df['Momentum_12m'] * 0.1
            )
            
            logger.info("Sector momentum indicators calculated successfully")
            return sector_df
        except Exception as e:
            logger.error(f"Error calculating sector momentum indicators: {e}")
            raise

    
    def calculate_volatility_indicators(self, sector_df):
        """
        Calculate volatility indicators for each sector.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector-level data.
        
        Returns
        -------
        pandas.DataFrame
            Sector data with volatility indicators.
        """
        try:
            logger.info("Calculating sector volatility indicators")
            
            # Sort data
            sector_df = sector_df.sort_values(['Sector', 'Date'])
            
            # Calculate volatility over different time periods
            # 1-month volatility (approx. 20 trading days)
            sector_df['Volatility_1m'] = sector_df.groupby('Sector')['Returns'].transform(
                lambda x: x.rolling(window=20).std() * np.sqrt(252)  # Annualized
            )
            
            # 3-month volatility (approx. 60 trading days)
            sector_df['Volatility_3m'] = sector_df.groupby('Sector')['Returns'].transform(
                lambda x: x.rolling(window=60).std() * np.sqrt(252)  # Annualized
            )
            
            # Volatility trend (change in volatility)
            sector_df['Volatility_Trend'] = (
                sector_df['Volatility_1m'] / sector_df['Volatility_3m']
            ) - 1
            
            # Volatility relative to market (using the previously calculated volatility)
            market_vol = sector_df.groupby('Date')['Volatility_1m'].mean().reset_index()
            market_vol = market_vol.rename(columns={'Volatility_1m': 'Market_Volatility_1m'})
            
            # Merge with sector data
            sector_df = pd.merge(sector_df, market_vol, on='Date', how='left')
            
            # Calculate relative volatility
            sector_df['Relative_Volatility'] = (
                sector_df['Volatility_1m'] / sector_df['Market_Volatility_1m']
            )
            
            logger.info("Sector volatility indicators calculated successfully")
            return sector_df
        
        except Exception as e:
            logger.error(f"Error calculating sector volatility indicators: {e}")
            raise

    def calculate_economic_cycle_performance(self, sector_df, macro_df):
        """
        Calculate sector performance in different economic cycles.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector-level data.
        macro_df : pandas.DataFrame
            Macro economic data with cycle indicators.
        
        Returns
        -------
        pandas.DataFrame
            Sector data with economic cycle performance indicators.
        """
        try:
            logger.info("Calculating sector performance in economic cycles")
            
            #check if economic cycle exist in macro_df
            if 'Economic_Cycle' not in macro_df.columns:
                logger.warning("Economic_Cycle not found in macro data, skipping economic cycle performance calculation")
                return sector_df
            
            #extract relevant columns from macro_df
            macro_subset = macro_df[['Date','Economic_Cycle']]
            
            #merge with sector data
            merged_df = pd.merge(sector_df,macro_subset,on='Date',how='left')
            
            # Calculate average returns by sector and economic cycle
            cycle_performance = merged_df.groupby(['Sector', 'Economic_Cycle'])['Returns'].mean().reset_index()
            cycle_performance = cycle_performance.rename(columns={'Returns': 'Avg_Cycle_Returns'})
            
            # Create a dictionary to map each cycle to a new column
            cycle_names = {
                0: 'Contraction_Returns',
                1: 'Early_Recovery_Returns',
                2: 'Expansion_Returns',
                3: 'Late_Cycle_Returns'
            }
            
            # Create separate columns for each cycle
            cycle_wide = pd.pivot_table(
                cycle_performance, 
                values='Avg_Cycle_Returns',
                index='Sector',
                columns='Economic_Cycle'
            ).reset_index()
         # Rename columns
            cycle_wide.columns = ['Sector'] + [cycle_names.get(col, f'Cycle_{col}_Returns') for col in cycle_wide.columns[1:]]
            
            # Fill NaN values with 0
            cycle_wide = cycle_wide.fillna(0)
            
            # Merge back to merged_df
            final_df = pd.merge(merged_df, cycle_wide, on='Sector', how='left')
            
            # Calculate cycle-specific expected returns based on current cycle
            for cycle in range(4):
                col_name = cycle_names.get(cycle)
                if col_name in final_df.columns:
                    final_df[f'Expected_{col_name}'] = np.where(
                        final_df['Economic_Cycle'] == cycle,
                        final_df[col_name],
                        0
                    )
            
            # Calculate a cycle expectation score
            cycle_cols = [f'Expected_{col}' for col in cycle_names.values() if f'Expected_{col}' in final_df.columns]
            if cycle_cols:
                final_df['Cycle_Expectation_Score'] = final_df[cycle_cols].sum(axis=1)
            
            logger.info("Sector economic cycle performance calculated successfully")
            return final_df
        
        except Exception as e:
            logger.error(f"Error calculating sector economic cycle performance: {e}")
            raise    
        
    def calculate_correlation_indicators(self, sector_df, macro_df):
        """
        Calculate correlations between sectors and macro indicators.
        
        Parameters
        ----------
        sector_df : pandas.DataFrame
            Sector-level data.
        macro_df : pandas.DataFrame
            Macro indicators.
        
        Returns
        -------
        pandas.DataFrame
            Sector data with correlation indicators.
        """
        try:
            logger.info("Calculating sector correlation indicators")
            
            #sort data
            sector_Df = sector_df.sort_values(['Sector','Date'])
            
            #extract relevant columns from macro_df
            if 'Market_Return' in macro_df.columns:
                macro_cols = ['Date','Market_Return']
                correlation_col = 'Market_Return'
            else:
                # Fallback to using Returns from market data if merged
                macro_cols = ['Date', 'Returns']
                correlation_col = 'Returns'
                
            macro_subset = macro_df[macro_cols]
            
            # Merge with sector data
            merged_df = pd.merge(sector_df, macro_subset, on='Date', how='left')
            
            # Calculate 60-day rolling correlations with market
            sectors = merged_df['Sector'].unique()
            
            # Create a dictionary to store correlation results
            corr_results = {}
            
            for sector in sectors:
                # Filter for this sector
                sector_data = merged_df[merged_df['Sector'] == sector].sort_values('Date')
                
                # Calculate 60-day rolling correlation
                rolling_corr = sector_data['Returns'].rolling(window=60).corr(sector_data[correlation_col])
                
                # Store results
                corr_results[sector] = rolling_corr
            
            # Create a dataframe with the correlations
            corr_df = pd.DataFrame({sector: corr_results[sector] for sector in sectors})
            corr_df = corr_df.reset_index()
            
            # Generate a sector trend based on rolling correlation
            for sector in sectors:
                # Convert column name to a string to avoid any potential issues
                sector_str = str(sector)
                corr_df[f'{sector_str}_Trend'] = corr_df[sector].diff(20)
            
            # Melt the dataframe to long format for easier merging
            dates = merged_df.sort_values('Date')['Date'].unique()
            corr_df['Date'] = dates[:len(corr_df)]  # Ensure lengths match
            
            # Melt the correlation data
            id_vars = ['Date']
            value_vars = [col for col in corr_df.columns if col not in id_vars and '_Trend' not in col]
            trend_vars = [col for col in corr_df.columns if '_Trend' in col]
            
            corr_melt = pd.melt(corr_df, id_vars=id_vars, value_vars=value_vars, 
                               var_name='Sector', value_name='Market_Correlation')
            
            trend_melt = pd.melt(corr_df, id_vars=id_vars, value_vars=trend_vars,
                              var_name='Sector_Trend', value_name='Correlation_Trend')
            
            # Fix the Sector_Trend column to match Sector
            trend_melt['Sector'] = trend_melt['Sector_Trend'].str.replace('_Trend', '')
            trend_melt = trend_melt.drop(columns=['Sector_Trend'])
            
            # Merge the correlation and trend data
            corr_data = pd.merge(corr_melt, trend_melt, on=['Date', 'Sector'], how='left')
            
            # Merge with the original sector data
            final_df = pd.merge(merged_df, corr_data, on=['Date', 'Sector'], how='left')
            
            logger.info("Sector correlation indicators calculated successfully")
            return final_df
        
        except Exception as e:
            logger.error(f"Error calculating sector correlation indicators: {e}")
            # Return the original dataframe if there's an error
            return sector_df
        
    def generate_sector_features(self, create_ranking=True):
        """
        Generate all sector indicators.
        
        Parameters
        ----------
        create_ranking : bool, default=True
            Whether to create ranking features.
        
        Returns
        -------
        pandas.DataFrame
            Complete set of sector indicators.
        """
        try:
            logger.info("Generating all sector indicators")
            
            # Load processed data
            sector_df, price_df, macro_df = self.load_processed_data()
            
            # Create a market dataframe from the price data
            market_df = price_df.groupby('Date').agg({
                'Returns': 'mean',
                'Volume': 'sum',
                'Adjusted': 'mean'
            }).reset_index()
            
            # Calculate relative strength
            sector_df = self.calculate_relative_strength(sector_df, market_df)
            
            # Calculate momentum indicators
            sector_df = self.calculate_momentum_indicators(sector_df)
            
            # Calculate volatility indicators
            sector_df = self.calculate_volatility_indicators(sector_df)
            
            # Calculate economic cycle performance
            sector_df = self.calculate_economic_cycle_performance(sector_df, macro_df)
            
            # Calculate correlation indicators
            sector_df = self.calculate_correlation_indicators(sector_df, macro_df)
            
            # Create sector rankings if requested
            if create_ranking:
                logger.info("Creating sector rankings")
                
                # Sort data
                sector_df = sector_df.sort_values(['Date', 'Sector'])
                
                # List of features to rank
                rank_features = [
                    'Relative_Strength_20d',
                    'Momentum_Score',
                    'Volatility_1m',  # Lower is better, will invert
                    'Market_Correlation'  # Depending on market, may want low or high
                ]
                
                # Create rankings for each feature by date
                for feature in rank_features:
                    if feature in sector_df.columns:
                        # For volatility, lower is better
                        ascending = (feature == 'Volatility_1m')
                        
                        # Create the ranking
                        sector_df[f'{feature}_Rank'] = sector_df.groupby('Date')[feature].rank(ascending=ascending)
                
                # Create a composite ranking
                rank_cols = [col for col in sector_df.columns if col.endswith('_Rank')]
                if rank_cols:
                    sector_df['Composite_Rank'] = sector_df[rank_cols].mean(axis=1)
                    # Rank the composite score (lowest is best)
                    sector_df['Final_Rank'] = sector_df.groupby('Date')['Composite_Rank'].rank()
                
                logger.info("Sector rankings created successfully")
            
            # Save to processed directory
            output_path = self.processed_dir / "sector_indicators.csv"
            sector_df.to_csv(output_path, index=False)
            logger.info(f"Saved all sector indicators to {output_path}")
            
            return sector_df
        
        except Exception as e:
            logger.error(f"Error generating sector features: {e}")
            raise


if __name__ == "__main__":
    # Example usage of the SectorIndicators class
    sector_indicators = SectorIndicators()
    
    # Generate all sector indicators
    all_indicators = sector_indicators.generate_sector_features()
    
    # Print sample data
    print("\nSector Indicators Sample:")
    print(all_indicators.head())
    
    # Print unique sectors
    print("\nUnique Sectors:")
    print(all_indicators['Sector'].unique())
    
    # Print statistics for a few key indicators
    key_indicators = ['Relative_Strength', 'Momentum_Score', 'Volatility_1m', 'Market_Correlation']
    print("\nKey Indicators Statistics:")
    for indicator in key_indicators:
        if indicator in all_indicators.columns:
            print(f"\n{indicator}:")
            print(all_indicators.groupby('Sector')[indicator].describe())

    