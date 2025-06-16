"""
Visualization Module
Creates professional charts and plots for financial analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Optional, Tuple, Union
import logging

from config.settings import VIZ_CONFIG

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')  # Use default instead of seaborn-v0_8 for compatibility
sns.set_palette(VIZ_CONFIG['color_palette'])


class FinancialPlotter:
    """Creates financial analysis visualizations."""
    
    def __init__(self, config: Dict = None):
        self.config = config or VIZ_CONFIG
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.rcParams['figure.figsize'] = self.config['figure_size']
        plt.rcParams['font.size'] = self.config['font_size']
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
    
    def plot_cip_deviations(self, data: pd.DataFrame, 
                           save_format: str = 'base64') -> Union[str, None]:
        """Plot CIP deviations for all currencies."""
        try:
            cip_columns = {
                "x_usd": "USD",
                "x_gbp": "GBP", 
                "x_jpy": "JPY",
                "x_sek": "SEK",
                "x_chf": "CHF"
            }
            
            colors = self.config['color_palette']
            
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Plot each currency
            for i, ((col, label), color) in enumerate(zip(cip_columns.items(), colors)):
                if col not in data.columns:
                    logger.warning(f"Column {col} not found in data")
                    continue
                
                temp_data = data[["Date", col]].dropna()
                if temp_data.empty:
                    logger.warning(f"No data available for {label}")
                    continue
                
                ax.plot(temp_data["Date"], temp_data[col], 
                       label=label, color=color, linewidth=2)
            
            # Formatting
            ax.set_title("Covered Interest Parity (CIP) Deviations Over Time", 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("CIP Deviation", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Format x-axis
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=45)
            
            # Add reference line
            ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.7)
            
            # Legend
            ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
            
            plt.tight_layout()
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot CIP deviations: {str(e)}")
            plt.close()
            return None
    
    def plot_bandwidth_vs_volatility(self, data: pd.DataFrame, 
                                    currency: str,
                                    save_format: str = 'base64') -> Union[str, None]:
        """Plot bandwidth vs volatility comparison."""
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Check required columns
            required_cols = ["Band_Width_scaled", "FX_RealizedVol_scaled"]
            if not all(col in data.columns for col in required_cols):
                logger.error(f"Missing required columns: {required_cols}")
                return None
            
            clean_data = data[required_cols].dropna()
            
            if len(clean_data) == 0:
                logger.error("No valid data for plotting")
                return None
            
            # Plot both series
            ax.plot(clean_data.index, clean_data["Band_Width_scaled"], 
                   label="Band Width", color="blue", linewidth=2)
            ax.plot(clean_data.index, clean_data["FX_RealizedVol_scaled"], 
                   label="FX Realized Volatility", color="orange", linewidth=2)
            
            # Formatting
            ax.set_title(f"{currency.upper()} - Band Width vs FX Realized Volatility",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Scaled Values", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot bandwidth vs volatility: {str(e)}")
            plt.close()
            return None
    
    def plot_cip_deviation_vs_band(self, results: Dict, currencies: Dict,
                                  save_format: str = 'base64') -> Dict:
        """Plot CIP deviation vs neutral band for each currency."""
        images = {}
        
        for curr, result in results.items():
            try:
                if 'data' not in result:
                    logger.warning(f"No data available for {curr}")
                    continue
                
                df = result["data"].copy()
                x_col = currencies[curr]["x"]
                
                if x_col not in df.columns:
                    logger.warning(f"Column {x_col} not found for {curr}")
                    continue
                
                fig, ax = plt.subplots(figsize=self.config['figure_size'])
                
                # Plot CIP deviation
                ax.plot(df.index, df[x_col], label="CIP Deviation", 
                       color="black", linewidth=2)
                
                # Plot quantile bands
                if all(col in df.columns for col in ["Q5.0", "Q95.0"]):
                    ax.plot(df.index, df["Q5.0"], label="5th Percentile", 
                           color="blue", linestyle="--", alpha=0.7)
                    ax.plot(df.index, df["Q95.0"], label="95th Percentile", 
                           color="red", linestyle="--", alpha=0.7)
                    
                    # Fill between quantiles
                    ax.fill_between(df.index, df["Q5.0"], df["Q95.0"], 
                                   color="lightgray", alpha=0.5, 
                                   label="Neutral Band")
                
                # Formatting
                ax.set_title(f"{curr.upper()} - CIP Deviation vs Estimated Neutral Band",
                            fontsize=14, fontweight='bold')
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("CIP Deviation", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                
                if save_format == 'base64':
                    images[curr] = self._save_plot_base64(fig)
                else:
                    plt.show()
                    
            except Exception as e:
                logger.error(f"Failed to plot {curr} deviation vs band: {str(e)}")
                plt.close()
        
        return images
    
    def plot_ciss_indicator(self, ciss_data: pd.Series,
                           title: str = "CISS Indicator",
                           save_format: str = 'base64') -> Union[str, None]:
        """Plot CISS indicator."""
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Plot CISS
            ax.plot(ciss_data.index, ciss_data.values, 
                   color="red", linewidth=2, label=title)
            
            # Formatting
            ax.set_title("ECB Composite Indicator of Systemic Stress (CISS)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Normalized CISS", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            # Add stress level indicators
            ax.axhline(0.5, color='orange', linestyle=':', alpha=0.7, 
                      label='Medium Stress')
            ax.axhline(0.8, color='red', linestyle=':', alpha=0.7, 
                      label='High Stress')
            
            plt.tight_layout()
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot CISS indicator: {str(e)}")
            plt.close()
            return None
    
    def plot_ciss_comparison(self, comparison_data: pd.DataFrame,
                            save_format: str = 'base64') -> Union[str, None]:
        """Plot comparison between official and constructed CISS."""
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Plot both series
            ax.plot(comparison_data.index, comparison_data["Official_ECB_CISS"], 
                   label="Official ECB CISS", color="blue", linewidth=2)
            ax.plot(comparison_data.index, comparison_data["Constructed_CISS"], 
                   label="Constructed CISS", color="red", linewidth=2, alpha=0.8)
            
            # Formatting
            ax.set_title("ECB CISS - Official vs Constructed Comparison",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Normalized CISS", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            # Calculate and display correlation
            correlation = comparison_data["Official_ECB_CISS"].corr(
                comparison_data["Constructed_CISS"]
            )
            ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot CISS comparison: {str(e)}")
            plt.close()
            return None
    
    def plot_cross_correlation(self, lags: List[int], ccf_values: List[float],
                              save_format: str = 'base64') -> Union[str, None]:
        """Plot cross-correlation function."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create stem plot
            ax.stem(lags, ccf_values, basefmt=' ')
            
            # Add reference lines
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
            
            # Formatting
            ax.set_title("Cross-Correlation: Constructed CISS vs Official ECB CISS",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Lag (periods)", fontsize=12)
            ax.set_ylabel("Correlation", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add significance bands (optional)
            n = len([v for v in ccf_values if not np.isnan(v)])
            if n > 0:
                significance = 1.96 / np.sqrt(n)
                ax.axhline(significance, color='red', linestyle=':', alpha=0.5,
                          label=f'95% Confidence')
                ax.axhline(-significance, color='red', linestyle=':', alpha=0.5)
                ax.legend()
            
            plt.tight_layout()
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot cross-correlation: {str(e)}")
            plt.close()
            return None
    
    def create_summary_dashboard(self, analysis_results: Dict,
                               save_format: str = 'base64') -> Union[str, None]:
        """Create a comprehensive dashboard with multiple plots."""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Create subplots
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: CIP Deviations
            ax1 = fig.add_subplot(gs[0, :])
            if 'cip_data' in analysis_results:
                self._plot_cip_subplot(ax1, analysis_results['cip_data'])
            
            # Plot 2: CISS Comparison
            ax2 = fig.add_subplot(gs[1, 0])
            if 'ciss_comparison' in analysis_results:
                self._plot_ciss_comparison_subplot(ax2, analysis_results['ciss_comparison'])
              # Plot 3: Cross-correlation
            ax3 = fig.add_subplot(gs[1, 1])
            if 'cross_correlation' in analysis_results:
                self._plot_ccf_subplot(ax3, analysis_results['cross_correlation'])
            
            # Plot 4: Summary statistics
            ax4 = fig.add_subplot(gs[2, :])
            self._plot_summary_stats(ax4, analysis_results)
            
            plt.suptitle("Financial Analysis Summary Dashboard", 
                        fontsize=16, fontweight='bold')
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to create summary dashboard: {str(e)}")
            plt.close()
            return None
    
    def _save_plot_base64(self, fig) -> str:
        """Save plot as base64 encoded string."""
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.config['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_base64
        except Exception as e:
            logger.error(f"Failed to save plot as base64: {str(e)}")
            plt.close(fig)
            return ""
    
    def _plot_cip_subplot(self, ax, data):
        """Helper method for CIP subplot."""
        # Implementation similar to plot_cip_deviations but for subplot
        pass
    
    def _plot_ciss_comparison_subplot(self, ax, data):
        """Helper method for CISS comparison subplot."""
        # Implementation similar to plot_ciss_comparison but for subplot
        pass
    
    def _plot_ccf_subplot(self, ax, data):
        """Helper method for cross-correlation subplot."""
        # Implementation similar to plot_cross_correlation but for subplot
        pass
    
    def _plot_summary_stats(self, ax, analysis_results):
        """Helper method for summary statistics subplot."""
        # Create a table or chart showing key statistics
        ax.axis('off')
        ax.text(0.5, 0.5, "Summary Statistics", ha='center', va='center',
               fontsize=14, transform=ax.transAxes)
    
    def plot_ecb_ciss(self, data: pd.DataFrame, 
                     ciss_column: str = "CISS",
                     save_format: str = 'base64') -> Union[str, None]:
        """Plot ECB CISS data directly (for official ECB CISS data)."""
        try:
            if ciss_column not in data.columns:
                logger.error(f"Column {ciss_column} not found in data")
                return None
            
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Plot CISS
            ax.plot(data.index, data[ciss_column], 
                   label="ECB CISS Index", color="red", linewidth=2)
            
            # Formatting
            ax.set_title("ECB Composite Indicator of Systemic Stress (CISS)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Normalized CISS", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            plt.tight_layout()
            
            if save_format == 'base64':
                return self._save_plot_base64(fig)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot ECB CISS: {str(e)}")
            plt.close()
            return None
