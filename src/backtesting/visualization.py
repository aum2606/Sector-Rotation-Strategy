import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
        logging.FileHandler(logs_dir / "visualization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Visualization:
    """
    Class for visualizing backtest results and performance metrics.
    Implements various plots and charts for analyzing strategy performance.
    """
    
    def __init__(self, figures_dir=None):
        """
        Initialize Visualization class.
        
        Parameters
        ----------
        figures_dir : str or Path, optional
            Directory to save figures.
        """
        if figures_dir is None:
            # Default to the standard project structure
            self.figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        else:
            self.figures_dir = Path(figures_dir)
        
        # Create figures directory if it doesn't exist
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set default style
        sns.set_style("whitegrid")
        
        logger.info(f"Visualization initialized with figures directory: {self.figures_dir}")
    
    def plot_cumulative_returns(self, results_df, strategies=None, title="Cumulative Returns",
                              figsize=(12, 8), save_path=None):
        """
        Plot cumulative returns for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Cumulative_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        title : str, default="Cumulative Returns"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting cumulative returns")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                ax.plot(strategy_data['Date'], strategy_data['Cumulative_Return'], 
                       label=strategy, linewidth=2)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Return", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "cumulative_returns.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cumulative returns plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting cumulative returns: {e}")
            raise
    
    def plot_drawdowns(self, results_df, strategies=None, title="Drawdowns",
                     figsize=(12, 8), save_path=None):
        """
        Plot drawdowns for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        title : str, default="Drawdowns"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting drawdowns")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                
                # Calculate cumulative returns
                cum_returns = (1 + strategy_data['Daily_Return']).cumprod()
                
                # Calculate running maximum
                running_max = np.maximum.accumulate(cum_returns)
                
                # Calculate drawdowns
                drawdowns = (cum_returns / running_max) - 1
                
                ax.plot(strategy_data['Date'], drawdowns, 
                       label=strategy, linewidth=2)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Drawdown", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "drawdowns.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdowns plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting drawdowns: {e}")
            raise
    
    def plot_rolling_returns(self, results_df, strategies=None, window=60, 
                          title="Rolling Returns", figsize=(12, 8), save_path=None):
        """
        Plot rolling returns for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        window : int, default=60
            Rolling window size.
        title : str, default="Rolling Returns"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info(f"Plotting {window}-day rolling returns")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                
                # Calculate rolling returns
                rolling_returns = strategy_data['Daily_Return'].rolling(window=window).mean() * 252  # Annualized
                
                ax.plot(strategy_data['Date'], rolling_returns, 
                       label=strategy, linewidth=2)
            
            # Set title and labels
            ax.set_title(f"{title} ({window}-day Rolling Window)", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(f"{window}-day Rolling Return (Annualized)", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / f"rolling_returns_{window}d.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rolling returns plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting rolling returns: {e}")
            raise
    
    def plot_rolling_volatility(self, results_df, strategies=None, window=60, 
                             title="Rolling Volatility", figsize=(12, 8), save_path=None):
        """
        Plot rolling volatility for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        window : int, default=60
            Rolling window size.
        title : str, default="Rolling Volatility"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info(f"Plotting {window}-day rolling volatility")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                
                # Calculate rolling volatility
                rolling_volatility = strategy_data['Daily_Return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
                
                ax.plot(strategy_data['Date'], rolling_volatility, 
                       label=strategy, linewidth=2)
            
            # Set title and labels
            ax.set_title(f"{title} ({window}-day Rolling Window)", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(f"{window}-day Rolling Volatility (Annualized)", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / f"rolling_volatility_{window}d.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rolling volatility plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting rolling volatility: {e}")
            raise
    
    def plot_rolling_sharpe(self, results_df, strategies=None, window=60, risk_free_rate=0,
                         title="Rolling Sharpe Ratio", figsize=(12, 8), save_path=None):
        """
        Plot rolling Sharpe ratio for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        window : int, default=60
            Rolling window size.
        risk_free_rate : float, default=0
            Risk-free rate (annualized).
        title : str, default="Rolling Sharpe Ratio"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info(f"Plotting {window}-day rolling Sharpe ratio")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Daily risk-free rate
            daily_rf = risk_free_rate / 252
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                
                # Calculate rolling excess returns
                excess_returns = strategy_data['Daily_Return'] - daily_rf
                
                # Calculate rolling mean and standard deviation
                rolling_mean = excess_returns.rolling(window=window).mean() * 252  # Annualized
                rolling_std = excess_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
                
                # Calculate rolling Sharpe ratio
                rolling_sharpe = rolling_mean / rolling_std
                
                ax.plot(strategy_data['Date'], rolling_sharpe, 
                       label=strategy, linewidth=2)
            
            # Set title and labels
            ax.set_title(f"{title} ({window}-day Rolling Window)", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(f"{window}-day Rolling Sharpe Ratio", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / f"rolling_sharpe_{window}d.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rolling Sharpe ratio plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting rolling Sharpe ratio: {e}")
            raise
    
    def plot_monthly_returns_heatmap(self, results_df, strategy, 
                                   title=None, figsize=(12, 8), save_path=None):
        """
        Plot monthly returns heatmap for a specific strategy.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        strategy : str
            Strategy to plot.
        title : str, optional
            Plot title. If None, use strategy name.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info(f"Plotting monthly returns heatmap for {strategy}")
            
            # Filter for the specified strategy
            strategy_data = results_df[results_df['Strategy'] == strategy].copy()
            
            # Extract year and month
            strategy_data['Year'] = strategy_data['Date'].dt.year
            strategy_data['Month'] = strategy_data['Date'].dt.month
            
            # Calculate monthly returns
            monthly_returns = strategy_data.groupby(['Year', 'Month'])['Daily_Return'].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            
            # Create a pivot table for the heatmap
            heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values='Daily_Return')
            
            # Replace month numbers with month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn', center=0,
                      linewidths=1, cbar_kws={'label': 'Monthly Return'}, ax=ax)
            
            # Set title
            if title is None:
                title = f"Monthly Returns: {strategy}"
            ax.set_title(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / f"monthly_returns_{strategy.replace(' ', '_').lower()}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monthly returns heatmap saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting monthly returns heatmap: {e}")
            raise
    
    def plot_performance_metrics(self, metrics_df, figsize=(14, 10), save_path=None):
        """
        Plot performance metrics for multiple strategies.
        
        Parameters
        ----------
        metrics_df : pandas.DataFrame
            DataFrame with performance metrics for each strategy.
        figsize : tuple, default=(14, 10)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting performance metrics")
            
            # Select relevant metrics to plot
            metrics_to_plot = [
                'annualized_return', 'volatility', 'sharpe_ratio', 'sortino_ratio', 
                'max_drawdown', 'calmar_ratio'
            ]
            
            # Add alpha and beta if available
            if 'alpha' in metrics_df.columns:
                metrics_to_plot.append('alpha')
            if 'beta' in metrics_df.columns:
                metrics_to_plot.append('beta')
            
            # Create a subset of the data with selected metrics
            plot_data = metrics_df[['Strategy'] + [m for m in metrics_to_plot if m in metrics_df.columns]]
            
            # Melt the data for easier plotting
            melted_data = pd.melt(plot_data, id_vars=['Strategy'], var_name='Metric', value_name='Value')
            
            # Create figure with subplots
            n_metrics = len(metrics_to_plot)
            n_cols = 2
            n_rows = (n_metrics + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()
            
            # Custom colors for each strategy
            strategies = metrics_df['Strategy'].unique()
            colors = sns.color_palette("Set2", len(strategies))
            
            # Plot each metric
            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_df.columns:
                    # Get data for this metric
                    metric_data = melted_data[melted_data['Metric'] == metric]
                    
                    # Sort by value
                    metric_data = metric_data.sort_values('Value', ascending=False)
                    
                    # Create bar plot
                    sns.barplot(x='Strategy', y='Value', data=metric_data, ax=axes[i],
                              palette={s: c for s, c in zip(strategies, colors)})
                    
                    # Format y-axis based on metric
                    if metric in ['annualized_return', 'volatility', 'max_drawdown']:
                        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                    
                    # Set title and labels
                    metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
                    axes[i].set_title(metric_name, fontsize=14)
                    axes[i].set_xlabel("")
                    
                    # Rotate x-axis labels for better readability
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            # Add a title to the figure
            plt.suptitle("Strategy Performance Metrics", fontsize=18, y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "performance_metrics.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance metrics plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting performance metrics: {e}")
            raise
    
    def plot_sector_allocation(self, results_df, title="Sector Allocation Over Time", 
                             figsize=(14, 8), save_path=None):
        """
        Plot sector allocation over time for sector rotation strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Selected_Sectors' columns.
        title : str, default="Sector Allocation Over Time"
            Plot title.
        figsize : tuple, default=(14, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting sector allocation over time")
            
            # Check if Selected_Sectors column exists
            if 'Selected_Sectors' not in results_df.columns:
                logger.error("Selected_Sectors column not found in results_df")
                raise ValueError("Selected_Sectors column is required for plotting sector allocation")
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Create a figure for each strategy
            figures = []
            
            for strategy in unique_strategies:
                # Filter for this strategy
                strategy_data = results_df[results_df['Strategy'] == strategy].copy()
                
                # Get unique dates (monthly)
                strategy_data['Month'] = strategy_data['Date'].dt.strftime('%Y-%m')
                monthly_data = strategy_data.drop_duplicates('Month').copy()
                
                # Create a matrix of sector allocations
                all_sectors = set()
                for sectors_str in monthly_data['Selected_Sectors']:
                    if isinstance(sectors_str, str):
                        sectors = [s.strip() for s in sectors_str.split(',')]
                        all_sectors.update(sectors)
                
                all_sectors = sorted(list(all_sectors))
                
                # Create a dataframe with binary sector allocations
                sector_matrix = pd.DataFrame(0, index=monthly_data.index, 
                                          columns=all_sectors)
                
                for i, sectors_str in enumerate(monthly_data['Selected_Sectors']):
                    if isinstance(sectors_str, str):
                        sectors = [s.strip() for s in sectors_str.split(',')]
                        for sector in sectors:
                            sector_matrix.loc[monthly_data.index[i], sector] = 1
                
                # Add date column
                sector_matrix['Date'] = monthly_data['Date'].values
                
                # Melt the dataframe for easier plotting
                melted_data = pd.melt(sector_matrix, id_vars=['Date'], 
                                    var_name='Sector', value_name='Allocation')
                
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Plot sector allocation
                sns.scatterplot(x='Date', y='Sector', size='Allocation', 
                              sizes=(0, 200), data=melted_data, ax=ax)
                
                # Set title and labels
                ax.set_title(f"{title}: {strategy}", fontsize=16)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Sector", fontsize=12)
                
                # Format x-axis as dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)
                
                # Remove legend
                ax.legend_.remove()
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure if path is specified
                if save_path is None:
                    strategy_filename = strategy.replace(' ', '_').lower()
                    fig_save_path = self.figures_dir / f"sector_allocation_{strategy_filename}.png"
                else:
                    # Append strategy name to save path
                    strategy_filename = strategy.replace(' ', '_').lower()
                    file_path = Path(save_path)
                    fig_save_path = file_path.parent / f"{file_path.stem}_{strategy_filename}{file_path.suffix}"
                
                plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Sector allocation plot for {strategy} saved to {fig_save_path}")
                
                figures.append(fig)
            
            return figures
        
        except Exception as e:
            logger.error(f"Error plotting sector allocation: {e}")
            raise
            
    def plot_returns_distribution(self, results_df, strategies=None, bins=50,
                                title="Returns Distribution", figsize=(12, 8), save_path=None):
        """
        Plot the distribution of daily returns for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Strategy' and 'Daily_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        bins : int, default=50
            Number of bins for the histogram.
        title : str, default="Returns Distribution"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting returns distribution")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                sns.histplot(strategy_data['Daily_Return'], label=strategy, bins=bins,
                           kde=True, alpha=0.5, ax=ax)
            
            # Add vertical line at 0
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Daily Return", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            
            # Format x-axis as percentages
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "returns_distribution.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
            raise

    def plot_correlation_heatmap(self, results_df, title="Strategy Correlation",
                              figsize=(10, 8), save_path=None):
        """
        Plot correlation heatmap between strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        title : str, default="Strategy Correlation"
            Plot title.
        figsize : tuple, default=(10, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting strategy correlation heatmap")
            
            # Pivot the data to get strategies as columns
            pivot_df = results_df.pivot_table(
                index='Date', columns='Strategy', values='Daily_Return'
            )
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                      vmin=-1, vmax=1, linewidths=0.5, ax=ax, fmt='.2f')
            
            # Set title
            ax.set_title(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "strategy_correlation.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy correlation heatmap saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting correlation heatmap: {e}")
            raise

    def plot_underwater(self, results_df, strategies=None, title="Underwater Plot",
                     figsize=(12, 8), save_path=None):
        """
        Plot underwater chart (continuous drawdowns) for multiple strategies.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        strategies : list, optional
            List of strategies to plot. If None, plot all strategies in results_df.
        title : str, default="Underwater Plot"
            Plot title.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting underwater chart")
            
            # Filter strategies if specified
            if strategies is not None:
                results_df = results_df[results_df['Strategy'].isin(strategies)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique strategies
            unique_strategies = results_df['Strategy'].unique()
            
            # Plot each strategy
            for strategy in unique_strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                
                # Calculate cumulative returns
                cum_returns = (1 + strategy_data['Daily_Return']).cumprod()
                
                # Calculate running maximum
                running_max = np.maximum.accumulate(cum_returns)
                
                # Calculate drawdowns
                underwater = (cum_returns / running_max) - 1
                
                ax.plot(strategy_data['Date'], underwater, 
                       label=strategy, linewidth=2)
            
            # Fill area under the curve
            ax.fill_between(results_df['Date'].unique(), 0, -1, color='red', alpha=0.1)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Drawdown", fontsize=12)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Set y-axis limits
            ax.set_ylim(bottom=min(underwater.min() * 1.1, -0.05), top=0.01)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "underwater_chart.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Underwater chart saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting underwater chart: {e}")
            raise

    def plot_risk_return_scatter(self, metrics_df, benchmark_strategy='Buy and Hold',
                              title="Risk vs Return", figsize=(10, 8), save_path=None):
        """
        Plot risk vs return scatter plot for strategies.
        
        Parameters
        ----------
        metrics_df : pandas.DataFrame
            DataFrame with performance metrics for each strategy.
            Should have 'Strategy', 'annualized_return', and 'volatility' columns.
        benchmark_strategy : str, default='Buy and Hold'
            Name of the benchmark strategy.
        title : str, default="Risk vs Return"
            Plot title.
        figsize : tuple, default=(10, 8)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info("Plotting risk vs return scatter plot")
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Check if required columns exist
            if 'annualized_return' not in metrics_df.columns or 'volatility' not in metrics_df.columns:
                logger.error("Required columns not found in metrics_df")
                raise ValueError("metrics_df must have 'annualized_return' and 'volatility' columns")
            
            # Plot each strategy
            for i, row in metrics_df.iterrows():
                strategy = row['Strategy']
                x = row['volatility']
                y = row['annualized_return']
                
                # Use different marker for benchmark
                if strategy == benchmark_strategy:
                    ax.scatter(x, y, s=150, marker='*', color='red', 
                              label=f"{strategy} (Benchmark)")
                else:
                    ax.scatter(x, y, s=100, marker='o')
                    
                # Add strategy label
                ax.annotate(strategy, (x, y), xytext=(5, 5), textcoords='offset points')
            
            # Add Sharpe ratio lines
            if 'risk_free_rate' in metrics_df.columns:
                rf = metrics_df['risk_free_rate'].iloc[0]
            else:
                rf = 0
                
            # Plot Sharpe ratio lines
            x_range = np.linspace(0, metrics_df['volatility'].max() * 1.2, 100)
            for sharpe in [0.5, 1.0, 1.5, 2.0]:
                y_values = rf + sharpe * x_range
                ax.plot(x_range, y_values, 'k--', alpha=0.3)
                # Label the line at the right edge
                ax.annotate(f'SR={sharpe}', 
                         xy=(x_range[-1], y_values[-1]),
                         xytext=(5, 0), 
                         textcoords='offset points',
                         fontsize=8)
            
            # Set title and labels
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Annualized Volatility", fontsize=12)
            ax.set_ylabel("Annualized Return", fontsize=12)
            
            # Format axes as percentages
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Set axis limits
            ax.set_xlim(left=0)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # If benchmark is specified, add a legend
            if benchmark_strategy in metrics_df['Strategy'].values:
                ax.legend(fontsize=10, loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "risk_return_scatter.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk vs return scatter plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting risk vs return scatter plot: {e}")
            raise

    def plot_regime_analysis(self, results_df, macro_df, regime_column='Economic_Cycle',
                          title="Performance by Market Regime", figsize=(12, 10), save_path=None):
        """
        Plot strategy performance across different market regimes.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', and 'Daily_Return' columns.
        macro_df : pandas.DataFrame
            DataFrame with macroeconomic data. Should have 'Date' and regime_column.
        regime_column : str, default='Economic_Cycle'
            Column in macro_df that defines the market regime.
        title : str, default="Performance by Market Regime"
            Plot title.
        figsize : tuple, default=(12, 10)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info(f"Plotting performance by market regime using {regime_column}")
            
            # Check if required column exists in macro_df
            if regime_column not in macro_df.columns:
                logger.error(f"{regime_column} not found in macro_df")
                raise ValueError(f"macro_df must have '{regime_column}' column")
            
            # Merge results with macro data to get regime information
            merged_df = pd.merge(results_df, macro_df[['Date', regime_column]], on='Date', how='left')
            
            # Get unique regimes and strategies
            unique_regimes = sorted(merged_df[regime_column].unique())
            unique_strategies = merged_df['Strategy'].unique()
            
            # Define a mapping for regime names if needed
            regime_names = {
                0: "Contraction",
                1: "Early Recovery",
                2: "Expansion",
                3: "Late Cycle"
            }
            
            # Calculate performance by regime for each strategy
            performance_by_regime = []
            
            for strategy in unique_strategies:
                for regime in unique_regimes:
                    # Filter data for this strategy and regime
                    regime_data = merged_df[(merged_df['Strategy'] == strategy) & 
                                          (merged_df[regime_column] == regime)]
                    
                    if not regime_data.empty:
                        # Calculate annualized return
                        returns = regime_data['Daily_Return'].values
                        cum_return = (1 + returns).prod() - 1
                        n_days = len(returns)
                        ann_return = (1 + cum_return) ** (252 / n_days) - 1 if n_days > 0 else 0
                        
                        # Calculate annualized volatility
                        vol = np.std(returns, ddof=1) * np.sqrt(252)
                        
                        # Calculate Sharpe ratio
                        sharpe = ann_return / vol if vol > 0 else 0
                        
                        # Add to results
                        regime_name = regime_names.get(regime, f"Regime {regime}")
                        performance_by_regime.append({
                            'Strategy': strategy,
                            'Regime': regime_name,
                            'Annualized_Return': ann_return,
                            'Volatility': vol,
                            'Sharpe_Ratio': sharpe,
                            'Days': n_days
                        })
            
            # Convert to DataFrame
            regime_performance = pd.DataFrame(performance_by_regime)
            
            # Create figure with subplots for returns, volatility, and Sharpe ratio
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # Plot annualized returns by regime
            sns.barplot(x='Regime', y='Annualized_Return', hue='Strategy', 
                      data=regime_performance, ax=axes[0])
            axes[0].set_title(f"{title}: Annualized Returns", fontsize=14)
            axes[0].set_ylabel("Annualized Return", fontsize=12)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend(fontsize=10, loc='upper right')
            
            # Plot volatility by regime
            sns.barplot(x='Regime', y='Volatility', hue='Strategy', 
                      data=regime_performance, ax=axes[1])
            axes[1].set_title("Annualized Volatility", fontsize=14)
            axes[1].set_ylabel("Volatility", fontsize=12)
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend().remove()  # Remove redundant legend
            
            # Plot Sharpe ratio by regime
            sns.barplot(x='Regime', y='Sharpe_Ratio', hue='Strategy', 
                      data=regime_performance, ax=axes[2])
            axes[2].set_title("Sharpe Ratio", fontsize=14)
            axes[2].set_ylabel("Sharpe Ratio", fontsize=12)
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].legend().remove()  # Remove redundant legend
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "regime_analysis.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regime analysis plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting regime analysis: {e}")
            raise

    def plot_benchmark_comparison(self, results_df, benchmark_strategy='Buy and Hold',
                               title="Strategy Performance vs. Benchmark", figsize=(12, 15), save_path=None):
        """
        Plot detailed comparison of strategies against a benchmark.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            DataFrame with strategy results. Should have 'Date', 'Strategy', 'Daily_Return',
            and 'Cumulative_Return' columns.
        benchmark_strategy : str, default='Buy and Hold'
            Name of the benchmark strategy.
        title : str, default="Strategy Performance vs. Benchmark"
            Plot title.
        figsize : tuple, default=(12, 15)
            Figure size.
        save_path : str or Path, optional
            Path to save the figure. If None, use default naming convention.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            logger.info(f"Plotting detailed benchmark comparison against {benchmark_strategy}")
            
            # Check if benchmark strategy exists in results
            if benchmark_strategy not in results_df['Strategy'].unique():
                logger.error(f"Benchmark strategy '{benchmark_strategy}' not found in results_df")
                raise ValueError(f"Benchmark strategy '{benchmark_strategy}' not found in results_df")
            
            # Filter benchmark data
            benchmark_data = results_df[results_df['Strategy'] == benchmark_strategy].copy()
            
            # Get unique non-benchmark strategies
            strategies = [s for s in results_df['Strategy'].unique() if s != benchmark_strategy]
            
            # Create a figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            
            # 1. Plot cumulative returns
            for strategy in strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy]
                axes[0].plot(strategy_data['Date'], strategy_data['Cumulative_Return'],
                         label=strategy, linewidth=2)
            
            # Add benchmark
            axes[0].plot(benchmark_data['Date'], benchmark_data['Cumulative_Return'],
                      label=f"{benchmark_strategy} (Benchmark)", linewidth=2,
                      color='black', linestyle='--')
            
            axes[0].set_title(f"{title}: Cumulative Returns", fontsize=14)
            axes[0].set_ylabel("Cumulative Return", fontsize=12)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=10, loc='upper left')
            
            # 2. Plot relative performance vs benchmark
            for strategy in strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy].copy()
                # Align dates
                dates = sorted(list(set(strategy_data['Date']) & set(benchmark_data['Date'])))
                strategy_data = strategy_data[strategy_data['Date'].isin(dates)]
                benchmark_subset = benchmark_data[benchmark_data['Date'].isin(dates)]
                
                # Calculate relative cumulative return
                relative_return = (
                    (1 + strategy_data['Cumulative_Return'].values) / 
                    (1 + benchmark_subset['Cumulative_Return'].values) - 1
                )
                
                axes[1].plot(strategy_data['Date'], relative_return,
                         label=f"{strategy} vs Benchmark", linewidth=2)
            
            # Add horizontal line at 0
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
            
            axes[1].set_title("Relative Performance vs Benchmark", fontsize=14)
            axes[1].set_ylabel("Relative Return", fontsize=12)
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=10, loc='upper left')
            
            # 3. Plot rolling correlation with benchmark
            window = 60  # 60-day rolling window
            
            for strategy in strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy].copy()
                # Align dates
                dates = sorted(list(set(strategy_data['Date']) & set(benchmark_data['Date'])))
                strategy_data = strategy_data[strategy_data['Date'].isin(dates)]
                benchmark_subset = benchmark_data[benchmark_data['Date'].isin(dates)]
                
                # Calculate rolling correlation
                rolling_corr = pd.Series(strategy_data['Daily_Return'].values).rolling(window=window).corr(
                    pd.Series(benchmark_subset['Daily_Return'].values)
                )
                
                axes[2].plot(strategy_data['Date'], rolling_corr,
                         label=f"{strategy} Correlation with Benchmark", linewidth=2)
            
            axes[2].set_title(f"{window}-Day Rolling Correlation with Benchmark", fontsize=14)
            axes[2].set_ylabel("Correlation", fontsize=12)
            axes[2].set_ylim(-1.1, 1.1)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=10, loc='upper left')
            
            # 4. Plot rolling beta with benchmark
            for strategy in strategies:
                strategy_data = results_df[results_df['Strategy'] == strategy].copy()
                # Align dates
                dates = sorted(list(set(strategy_data['Date']) & set(benchmark_data['Date'])))
                strategy_data = strategy_data[strategy_data['Date'].isin(dates)]
                benchmark_subset = benchmark_data[benchmark_data['Date'].isin(dates)]
                
                # Calculate rolling beta
                # Beta = Cov(r_a, r_b) / Var(r_b)
                rolling_cov = pd.Series(strategy_data['Daily_Return'].values).rolling(window=window).cov(
                    pd.Series(benchmark_subset['Daily_Return'].values)
                )
                rolling_var = pd.Series(benchmark_subset['Daily_Return'].values).rolling(window=window).var()
                
                rolling_beta = rolling_cov / rolling_var
                
                axes[3].plot(strategy_data['Date'], rolling_beta,
                         label=f"{strategy} Beta to Benchmark", linewidth=2)
            
            # Add horizontal line at 1
            axes[3].axhline(y=1, color='black', linestyle='--', alpha=0.7)
            
            axes[3].set_title(f"{window}-Day Rolling Beta to Benchmark", fontsize=14)
            axes[3].set_xlabel("Date", fontsize=12)
            axes[3].set_ylabel("Beta", fontsize=12)
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[3].tick_params(axis='x', rotation=45)
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(fontsize=10, loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is specified
            if save_path is None:
                save_path = self.figures_dir / "benchmark_comparison.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Benchmark comparison plot saved to {save_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting benchmark comparison: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the Visualization class
    visualization = Visualization()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252*2, freq='B')
    
    # Strategy 1: Momentum
    momentum_returns = np.random.normal(0.0008, 0.01, len(dates))
    momentum_cum_returns = (1 + momentum_returns).cumprod() - 1
    
    # Strategy 2: Value
    value_returns = np.random.normal(0.0006, 0.015, len(dates))
    value_cum_returns = (1 + value_returns).cumprod() - 1
    
    # Strategy 3: Equal Weight
    equal_returns = np.random.normal(0.0005, 0.012, len(dates))
    equal_cum_returns = (1 + equal_returns).cumprod() - 1
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Strategy': ['Momentum Rotation'] * len(dates),
        'Daily_Return': momentum_returns,
        'Cumulative_Return': momentum_cum_returns,
        'Selected_Sectors': np.random.choice([
            'Technology, Healthcare', 
            'Financials, Energy', 
            'Consumer Discretionary, Utilities'
        ], len(dates))
    })
    
    # Add other strategies
    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'Date': dates,
            'Strategy': ['Value Rotation'] * len(dates),
            'Daily_Return': value_returns,
            'Cumulative_Return': value_cum_returns,
            'Selected_Sectors': np.random.choice([
                'Industrials, Materials', 
                'Energy, Financials', 
                'Utilities, Real Estate'
            ], len(dates))
        }),
        pd.DataFrame({
            'Date': dates,
            'Strategy': ['Equal Weight'] * len(dates),
            'Daily_Return': equal_returns,
            'Cumulative_Return': equal_cum_returns,
            'Selected_Sectors': ['All Sectors'] * len(dates)
        })
    ], ignore_index=True)
    
    # Plot cumulative returns
    visualization.plot_cumulative_returns(results_df)
    
    # Plot drawdowns
    visualization.plot_drawdowns(results_df)
    
    # Plot rolling returns
    visualization.plot_rolling_returns(results_df, window=60)
    
    # Plot sector allocation for a specific strategy
    visualization.plot_sector_allocation(
        results_df[results_df['Strategy'] == 'Momentum Rotation']
    )
    
    # Plot returns distribution
    visualization.plot_returns_distribution(results_df)
    
    # Plot correlation heatmap
    visualization.plot_correlation_heatmap(results_df)
    
    # Plot underwater chart
    visualization.plot_underwater(results_df)
    
    # Create performance metrics for risk-return scatter plot
    performance_metrics = {
        'Strategy': ['Momentum Rotation', 'Value Rotation', 'Equal Weight'],
        'annualized_return': [0.12, 0.09, 0.07],
        'volatility': [0.15, 0.12, 0.10],
        'sharpe_ratio': [0.8, 0.75, 0.7],
        'risk_free_rate': [0.02, 0.02, 0.02]
    }
    performance_df = pd.DataFrame(performance_metrics)
    
    # Plot risk-return scatter
    visualization.plot_risk_return_scatter(performance_df)
    
    # Create sample macro data for regime analysis
    macro_df = pd.DataFrame({
        'Date': dates,
        'Economic_Cycle': np.random.choice([0, 1, 2, 3], len(dates))
    })
    
    # Plot regime analysis
    visualization.plot_regime_analysis(results_df, macro_df)
    
    # Plot benchmark comparison
    visualization.plot_benchmark_comparison(results_df, benchmark_strategy='Equal Weight')