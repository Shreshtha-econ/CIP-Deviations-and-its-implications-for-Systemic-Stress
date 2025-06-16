"""
Main Analysis Script
Orchestrates the complete financial analysis pipeline.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any

# Add project root to path (go up one level from scripts directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataMerger, save_processed_data, load_processed_data
from src.data.preprocessor import DataPreprocessor
from src.analysis.cip_analysis import CIPAnalyzer, CurrencyAnalyzer
from src.analysis.risk_indicators import SystemicRiskAnalyzer
from config.settings import CURRENCIES, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MasterThesisAnalyzer:
    """Main analyzer class that orchestrates the entire analysis."""
    
    def __init__(self):
        self.data_merger = DataMerger()
        self.preprocessor = DataPreprocessor()
        self.cip_analyzer = CIPAnalyzer()
        self.currency_analyzer = CurrencyAnalyzer()
        self.risk_analyzer = SystemicRiskAnalyzer()
        
        self.results = {}
        self.master_data = None
    
    def run_complete_analysis(self, force_reload: bool = False) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        logger.info("Starting Master Thesis Analysis Pipeline")
        
        try:
            # Step 1: Load and merge data
            self.master_data = self._load_or_create_master_data(force_reload)
            
            # Step 2: Preprocess data
            self.master_data = self._preprocess_data()
            
            # Step 3: Calculate CIP deviations
            self.master_data = self._calculate_cip_metrics()
            
            # Step 4: Analyze individual currencies
            currency_results = self._analyze_currencies()
            
            # Step 5: Construct systemic risk indicators
            risk_results = self._construct_risk_indicators()
            
            # Step 6: Save results
            self._save_analysis_results()
            
            # Compile final results
            final_results = {
                'master_data': self.master_data,
                'currency_analysis': currency_results,
                'risk_analysis': risk_results,
                'summary': self._generate_summary()
            }
            
            logger.info("Analysis pipeline completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise
    
    def _load_or_create_master_data(self, force_reload: bool) -> pd.DataFrame:
        """Load existing master data or create new."""
        master_file = PROCESSED_DATA_DIR / 'master_dataset.csv'
        
        if not force_reload and master_file.exists():
            logger.info("Loading existing master dataset")
            return load_processed_data('master_dataset.csv')
        else:
            logger.info("Creating new master dataset")
            master_data = self.data_merger.create_master_dataset()
            save_processed_data(master_data, 'master_dataset.csv')
            return master_data
    
    def _preprocess_data(self) -> pd.DataFrame:
        """Preprocess the master dataset."""
        logger.info("Preprocessing data")
        
        # Convert data types
        processed_data = self.preprocessor.convert_numeric_columns(self.master_data)
        
        # Handle missing values
        processed_data = self.preprocessor.handle_missing_values(processed_data)
        
        # Create additional features
        processed_data = self.preprocessor.create_time_features(processed_data)
        
        return processed_data
    
    def _calculate_cip_metrics(self) -> pd.DataFrame:
        """Calculate CIP-related metrics."""
        logger.info("Calculating CIP metrics")
        
        # Rate conversions
        data_with_conversions = self.cip_analyzer.calculate_rate_conversions(self.master_data)
        
        # CIP deviations
        data_with_cip = self.cip_analyzer.calculate_cip_deviations(data_with_conversions)
        
        # Trading costs
        data_with_costs = self.cip_analyzer.calculate_trading_costs(data_with_cip)
        
        # Clean trading costs
        final_data = self.cip_analyzer.clean_trading_costs(data_with_costs)
        
        return final_data
    
    def _analyze_currencies(self) -> Dict[str, Any]:
        """Analyze individual currencies."""
        logger.info("Analyzing individual currencies")
        
        currency_results = {}
        
        for currency_code in CURRENCIES.keys():
            try:
                logger.info(f"Analyzing {currency_code.upper()}")
                result = self.currency_analyzer.analyze_currency(self.master_data, currency_code)
                currency_results[currency_code] = result
            except Exception as e:
                logger.error(f"Failed to analyze {currency_code}: {str(e)}")
                currency_results[currency_code] = {'error': str(e)}
        
        return currency_results
    
    def _construct_risk_indicators(self) -> Dict[str, Any]:
        """Construct systemic risk indicators."""
        logger.info("Constructing systemic risk indicators")
        
        try:
            # Load market indicators if not already in master data
            risk_data = self._prepare_risk_data()
            
            # Construct CISS indicator
            ciss_result = self.risk_analyzer.construct_ciss_indicator(risk_data)
            
            # Compare with official ECB CISS
            comparison_result = self.risk_analyzer.compare_with_official_ciss(ciss_result)
            
            return {
                'ciss': ciss_result,
                'comparison': comparison_result
            }
            
        except Exception as e:
            logger.error(f"Risk indicator construction failed: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_risk_data(self) -> pd.DataFrame:
        """Prepare data for risk indicator construction."""
        # This would load additional market data needed for CISS construction
        # For now, we'll use the master data
        return self.master_data
    
    def _save_analysis_results(self):
        """Save analysis results to files."""
        logger.info("Saving analysis results")
        
        # Save master data
        save_processed_data(self.master_data, 'final_merged_data.csv')
        
        # Save individual currency results if available
        if hasattr(self, 'currency_results'):
            for currency, result in self.currency_results.items():
                if 'data' in result:
                    save_processed_data(
                        result['data'].reset_index(), 
                        f'{currency}_analysis_results.csv'
                    )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary."""
        summary = {
            'data_shape': self.master_data.shape,
            'date_range': {
                'start': str(self.master_data['Date'].min()),
                'end': str(self.master_data['Date'].max())
            },
            'currencies_analyzed': list(CURRENCIES.keys()),
            'missing_data_percentage': (
                self.master_data.isnull().sum().sum() / 
                (self.master_data.shape[0] * self.master_data.shape[1]) * 100
            )
        }
        
        return summary


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Master Thesis Financial Analysis')
    parser.add_argument('--force-reload', action='store_true', 
                       help='Force reload of master dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MasterThesisAnalyzer()
    
    try:
        # Run analysis
        results = analyzer.run_complete_analysis(force_reload=args.force_reload)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        summary = results['summary']
        print(f"Data shape: {summary['data_shape']}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Currencies analyzed: {', '.join(summary['currencies_analyzed'])}")
        print(f"Missing data: {summary['missing_data_percentage']:.2f}%")
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {PROCESSED_DATA_DIR}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
