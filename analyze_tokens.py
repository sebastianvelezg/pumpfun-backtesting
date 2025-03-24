import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from pathlib import Path
import seaborn as sns
from collections import defaultdict

class PumpfunAnalyzer:
    def __init__(self, data_dir="pumpfun_data"):
        self.data_dir = data_dir
        self.tokens_dir = os.path.join(data_dir, "tokens")
        self.index_file = os.path.join(data_dir, "token_index.json")
        self.token_index = {}
        self.analysis_results = {}
        
        # Load token index
        self._load_token_index()
    
    def _load_token_index(self):
        """Load the token index file"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.token_index = json.load(f)
                print(f"Loaded index with {len(self.token_index)} tokens")
            except Exception as e:
                print(f"Error loading token index: {e}")
                self.token_index = {}
        else:
            print(f"Token index file not found at {self.index_file}")
    
    def _get_token_filepath(self, token_address):
        """Get the filepath for a specific token's data file"""
        if len(token_address) >= 2:
            subdir = token_address[:2]
            return os.path.join(self.tokens_dir, subdir, f"{token_address}.json")
        else:
            return os.path.join(self.tokens_dir, f"{token_address}.json")
    
    def load_token_data(self, token_address):
        """Load data for a specific token"""
        filepath = self._get_token_filepath(token_address)
        if not os.path.exists(filepath):
            print(f"Token data file not found: {filepath}")
            return None
            
        try:
            with open(filepath, 'r') as f:
                token_data = json.load(f)
            print(f"Loaded data for token: {token_data.get('name', 'Unknown')} ({token_data.get('symbol', 'Unknown')})")
            return token_data
        except Exception as e:
            print(f"Error loading token data: {e}")
            return None
    
    def find_all_token_files(self):
        """Find all token JSON files in the data directory"""
        pattern = os.path.join(self.tokens_dir, "**", "*.json")
        return glob.glob(pattern, recursive=True)
    
    def analyze_token(self, token_address):
        """Analyze a single token's performance"""
        token_data = self.load_token_data(token_address)
        if not token_data or 'trades' not in token_data or not token_data['trades']:
            print(f"No trade data available for token {token_address}")
            return None
        
        # Basic token info
        token_info = {
            'address': token_address,
            'name': token_data.get('name', 'Unknown'),
            'symbol': token_data.get('symbol', 'Unknown'),
            'created_at': token_data.get('created_at', 'Unknown'),
        }
        
        # Extract trades and convert to DataFrame for analysis
        trades_df = pd.DataFrame(token_data['trades'])
        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('unix_time')
        
        # Calculate time since creation for each trade
        if 'creation_time' in token_data:
            creation_time = token_data['creation_time']
            trades_df['seconds_since_creation'] = trades_df['unix_time'] - creation_time
            trades_df['minutes_since_creation'] = trades_df['seconds_since_creation'] / 60
        
        # Price analysis
        first_price = trades_df['price'].iloc[0] if not trades_df.empty else 0
        last_price = trades_df['price'].iloc[-1] if not trades_df.empty else 0
        highest_price = trades_df['price'].max() if not trades_df.empty else 0
        lowest_price = trades_df['price'].min() if not trades_df.empty else 0
        
        # Overall price change
        price_change = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
        
        # High to low change (drawdown)
        drawdown = ((lowest_price - highest_price) / highest_price * 100) if highest_price > 0 else 0
        
        # Volume analysis
        total_volume_sol = trades_df['sol_amount'].sum() if not trades_df.empty else 0
        buy_volume = trades_df[trades_df['type'] == 'buy']['sol_amount'].sum() if not trades_df.empty else 0
        sell_volume = trades_df[trades_df['type'] == 'sell']['sol_amount'].sum() if not trades_df.empty else 0
        volume_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        
        # Trade count analysis
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['type'] == 'buy']) if not trades_df.empty else 0
        sell_trades = len(trades_df[trades_df['type'] == 'sell']) if not trades_df.empty else 0
        
        # Time analysis
        first_trade_time = pd.to_datetime(trades_df['timestamp'].iloc[0]) if not trades_df.empty else None
        last_trade_time = pd.to_datetime(trades_df['timestamp'].iloc[-1]) if not trades_df.empty else None
        
        if first_trade_time and last_trade_time:
            trading_duration = (last_trade_time - first_trade_time).total_seconds() / 60  # in minutes
        else:
            trading_duration = 0
        
        # Compile analysis results
        analysis = {
            'token_info': token_info,
            'price_metrics': {
                'first_price': first_price,
                'last_price': last_price,
                'highest_price': highest_price,
                'lowest_price': lowest_price,
                'price_change_pct': price_change,
                'drawdown_pct': drawdown
            },
            'volume_metrics': {
                'total_volume_sol': total_volume_sol,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': volume_ratio
            },
            'trade_metrics': {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'buy_sell_trade_ratio': buy_trades / sell_trades if sell_trades > 0 else float('inf')
            },
            'time_metrics': {
                'first_trade': first_trade_time.strftime('%Y-%m-%d %H:%M:%S') if first_trade_time else None,
                'last_trade': last_trade_time.strftime('%Y-%m-%d %H:%M:%S') if last_trade_time else None,
                'trading_duration_minutes': trading_duration
            },
            'trades_df': trades_df
        }
        
        # Store analysis results
        self.analysis_results[token_address] = analysis
        
        return analysis
    
    def generate_price_chart(self, token_address, save_path=None):
        """Generate a price chart for the token"""
        if token_address not in self.analysis_results:
            print(f"No analysis results for token {token_address}. Running analysis first.")
            analysis = self.analyze_token(token_address)
            if not analysis:
                return
        else:
            analysis = self.analysis_results[token_address]
        
        trades_df = analysis['trades_df']
        if trades_df.empty:
            print("No trades to plot")
            return
            
        token_info = analysis['token_info']
        price_metrics = analysis['price_metrics']
        
        # Create the figure and plot
        plt.figure(figsize=(12, 8))
        
        # Plot price over time
        if 'minutes_since_creation' in trades_df.columns:
            plt.plot(trades_df['minutes_since_creation'], trades_df['price'], marker='.', alpha=0.7)
            plt.xlabel('Minutes Since Creation')
        else:
            plt.plot(trades_df['datetime'], trades_df['price'], marker='.', alpha=0.7)
            plt.xlabel('Time')
            
        plt.ylabel('Price (SOL)')
        plt.title(f"{token_info['name']} ({token_info['symbol']}) Price History")
        
        # Add a grid
        plt.grid(True, alpha=0.3)
        
        # Add buy/sell markers
        buys = trades_df[trades_df['type'] == 'buy']
        sells = trades_df[trades_df['type'] == 'sell']
        
        if 'minutes_since_creation' in trades_df.columns:
            plt.scatter(buys['minutes_since_creation'], buys['price'], color='green', marker='^', alpha=0.7, label='Buy')
            plt.scatter(sells['minutes_since_creation'], sells['price'], color='red', marker='v', alpha=0.7, label='Sell')
        else:
            plt.scatter(buys['datetime'], buys['price'], color='green', marker='^', alpha=0.7, label='Buy')
            plt.scatter(sells['datetime'], sells['price'], color='red', marker='v', alpha=0.7, label='Sell')
        
        # Add price change annotation
        plt.annotate(
            f"Price Change: {price_metrics['price_change_pct']:.2f}%\nHighest: {price_metrics['highest_price']:.10f}\nLowest: {price_metrics['lowest_price']:.10f}", 
            xy=(0.02, 0.95), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def generate_volume_chart(self, token_address, save_path=None):
        """Generate a volume chart for the token"""
        if token_address not in self.analysis_results:
            print(f"No analysis results for token {token_address}. Running analysis first.")
            analysis = self.analyze_token(token_address)
            if not analysis:
                return
        else:
            analysis = self.analysis_results[token_address]
        
        trades_df = analysis['trades_df']
        if trades_df.empty:
            print("No trades to plot")
            return
            
        token_info = analysis['token_info']
        
        # Create the figure and plot
        plt.figure(figsize=(12, 8))
        
        # Prepare the data - group by minute and sum volumes
        trades_df['minute'] = (trades_df['seconds_since_creation'] // 60).astype(int) if 'seconds_since_creation' in trades_df.columns else trades_df['datetime'].dt.floor('min')
        
        buy_volumes = trades_df[trades_df['type'] == 'buy'].groupby('minute')['sol_amount'].sum()
        sell_volumes = trades_df[trades_df['type'] == 'sell'].groupby('minute')['sol_amount'].sum()
        
        # Plot the volumes
        if 'seconds_since_creation' in trades_df.columns:
            minutes = sorted(trades_df['minute'].unique())
            
            buy_data = [buy_volumes.get(m, 0) for m in minutes]
            sell_data = [-sell_volumes.get(m, 0) for m in minutes]  # Negative for sells
            
            plt.bar(minutes, buy_data, color='green', alpha=0.7, label='Buy Volume')
            plt.bar(minutes, sell_data, color='red', alpha=0.7, label='Sell Volume')
            plt.xlabel('Minutes Since Creation')
        else:
            # If we don't have seconds_since_creation, use datetime
            buy_df = pd.DataFrame(buy_volumes).reset_index()
            sell_df = pd.DataFrame(sell_volumes).reset_index()
            
            plt.bar(buy_df['minute'], buy_df['sol_amount'], color='green', alpha=0.7, label='Buy Volume')
            plt.bar(sell_df['minute'], -sell_df['sol_amount'], color='red', alpha=0.7, label='Sell Volume')
            plt.xlabel('Time')
            
        plt.ylabel('Volume (SOL)')
        plt.title(f"{token_info['name']} ({token_info['symbol']}) Trading Volume")
        
        # Add a grid
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Volume chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def analyze_all_tokens(self, limit=None):
        """Analyze all tokens in the dataset"""
        token_files = self.find_all_token_files()
        if limit:
            token_files = token_files[:limit]
            
        print(f"Analyzing {len(token_files)} token files...")
        
        results = []
        for file_path in token_files:
            token_address = os.path.basename(file_path).replace('.json', '')
            analysis = self.analyze_token(token_address)
            if analysis:
                # Extract key metrics for summary
                summary = {
                    'address': token_address,
                    'name': analysis['token_info']['name'],
                    'symbol': analysis['token_info']['symbol'],
                    'price_change_pct': analysis['price_metrics']['price_change_pct'],
                    'highest_price': analysis['price_metrics']['highest_price'],
                    'total_volume_sol': analysis['volume_metrics']['total_volume_sol'],
                    'buy_sell_ratio': analysis['volume_metrics']['buy_sell_ratio'],
                    'total_trades': analysis['trade_metrics']['total_trades'],
                    'trading_duration_minutes': analysis['time_metrics']['trading_duration_minutes']
                }
                results.append(summary)
        
        # Convert to DataFrame for easier analysis
        summary_df = pd.DataFrame(results)
        return summary_df
    
    def generate_summary_report(self, output_file="token_analysis_summary.csv"):
        """Generate a summary report of all token analyses"""
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available for summary")
            return
            
        # Save to CSV
        summary_df.to_csv(output_file, index=False)
        print(f"Summary report saved to {output_file}")
        
        # Some basic statistics
        print("\n=== Token Analysis Summary ===")
        print(f"Total tokens analyzed: {len(summary_df)}")
        print(f"Average price change: {summary_df['price_change_pct'].mean():.2f}%")
        print(f"Median price change: {summary_df['price_change_pct'].median():.2f}%")
        print(f"Average trading volume: {summary_df['total_volume_sol'].mean():.4f} SOL")
        print(f"Average number of trades: {summary_df['total_trades'].mean():.1f}")
        
        # Return the summary DataFrame
        return summary_df
    
    def find_best_performing_tokens(self, n=10):
        """Find the best performing tokens by price change"""
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available")
            return
            
        # Sort by price change
        best_tokens = summary_df.sort_values('price_change_pct', ascending=False).head(n)
        
        print("\n=== Best Performing Tokens ===")
        for _, token in best_tokens.iterrows():
            print(f"{token['name']} ({token['symbol']}): {token['price_change_pct']:.2f}% change, {token['total_volume_sol']:.4f} SOL volume")
            
        return best_tokens
    
    def find_worst_performing_tokens(self, n=10):
        """Find the worst performing tokens by price change"""
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available")
            return
            
        # Sort by price change (ascending)
        worst_tokens = summary_df.sort_values('price_change_pct', ascending=True).head(n)
        
        print("\n=== Worst Performing Tokens ===")
        for _, token in worst_tokens.iterrows():
            print(f"{token['name']} ({token['symbol']}): {token['price_change_pct']:.2f}% change, {token['total_volume_sol']:.4f} SOL volume")
            
        return worst_tokens
    
    def find_highest_volume_tokens(self, n=10):
        """Find tokens with the highest trading volume"""
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available")
            return
            
        # Sort by volume
        highest_volume = summary_df.sort_values('total_volume_sol', ascending=False).head(n)
        
        print("\n=== Highest Volume Tokens ===")
        for _, token in highest_volume.iterrows():
            print(f"{token['name']} ({token['symbol']}): {token['total_volume_sol']:.4f} SOL volume, {token['price_change_pct']:.2f}% change")
            
        return highest_volume
    
    def analyze_patterns(self):
        """Analyze common patterns across all tokens"""
        all_tokens = self.analyze_all_tokens()
        if all_tokens.empty:
            print("No token data available for pattern analysis")
            return
            
        # Categorize tokens based on performance
        all_tokens['performance_category'] = pd.cut(
            all_tokens['price_change_pct'],
            bins=[-float('inf'), -50, -10, 10, 50, float('inf')],
            labels=['Large Drop', 'Small Drop', 'Stable', 'Small Gain', 'Large Gain']
        )
        
        # Count tokens in each category
        performance_counts = all_tokens['performance_category'].value_counts()
        
        print("\n=== Token Performance Categories ===")
        for category, count in performance_counts.items():
            print(f"{category}: {count} tokens ({count/len(all_tokens)*100:.1f}%)")
            
        # Analyze correlation between metrics
        correlation = all_tokens[['price_change_pct', 'total_volume_sol', 'buy_sell_ratio', 'total_trades', 'trading_duration_minutes']].corr()
        
        print("\n=== Metric Correlations ===")
        print(correlation)
        
        # Average metrics by performance category
        category_means = all_tokens.groupby('performance_category')[
            ['price_change_pct', 'total_volume_sol', 'buy_sell_ratio', 'total_trades']
        ].mean()
        
        print("\n=== Average Metrics by Performance Category ===")
        print(category_means)
        
        return {
            'performance_counts': performance_counts,
            'correlation': correlation,
            'category_means': category_means,
            'all_tokens': all_tokens
        }
    
    def analyze_specific_token(self, token_address):
        """Comprehensive analysis of a specific token"""
        analysis = self.analyze_token(token_address)
        if not analysis:
            print(f"Could not analyze token {token_address}")
            return
            
        token_info = analysis['token_info']
        price_metrics = analysis['price_metrics']
        volume_metrics = analysis['volume_metrics']
        trade_metrics = analysis['trade_metrics']
        time_metrics = analysis['time_metrics']
        
        print("\n" + "="*50)
        print(f"DETAILED ANALYSIS FOR: {token_info['name']} ({token_info['symbol']})")
        print("="*50)
        
        print(f"\nToken Address: {token_info['address']}")
        print(f"Created At: {token_info['created_at']}")
        
        print("\n--- PRICE METRICS ---")
        print(f"First Price: {price_metrics['first_price']:.10f} SOL")
        print(f"Last Price: {price_metrics['last_price']:.10f} SOL")
        print(f"Highest Price: {price_metrics['highest_price']:.10f} SOL")
        print(f"Lowest Price: {price_metrics['lowest_price']:.10f} SOL")
        print(f"Price Change: {price_metrics['price_change_pct']:.2f}%")
        print(f"Max Drawdown: {price_metrics['drawdown_pct']:.2f}%")
        
        print("\n--- VOLUME METRICS ---")
        print(f"Total Volume: {volume_metrics['total_volume_sol']:.4f} SOL")
        print(f"Buy Volume: {volume_metrics['buy_volume']:.4f} SOL")
        print(f"Sell Volume: {volume_metrics['sell_volume']:.4f} SOL")
        print(f"Buy/Sell Volume Ratio: {volume_metrics['buy_sell_ratio']:.2f}")
        
        print("\n--- TRADE METRICS ---")
        print(f"Total Trades: {trade_metrics['total_trades']}")
        print(f"Buy Trades: {trade_metrics['buy_trades']}")
        print(f"Sell Trades: {trade_metrics['sell_trades']}")
        print(f"Buy/Sell Trade Ratio: {trade_metrics['buy_sell_trade_ratio']:.2f}")
        
        print("\n--- TIME METRICS ---")
        print(f"First Trade: {time_metrics['first_trade']}")
        print(f"Last Trade: {time_metrics['last_trade']}")
        print(f"Trading Duration: {time_metrics['trading_duration_minutes']:.2f} minutes")
        
        # Generate the charts
        os.makedirs("analysis_output", exist_ok=True)
        
        print("\nGenerating price chart...")
        self.generate_price_chart(token_address, f"analysis_output/{token_info['symbol']}_price_chart.png")
        
        print("Generating volume chart...")
        self.generate_volume_chart(token_address, f"analysis_output/{token_info['symbol']}_volume_chart.png")
        
        return analysis
    
    def generate_comprehensive_report(self, output_file="analysis_output/comprehensive_report.md"):
        """Generate a comprehensive markdown report of all analysis results"""
        print("\nGenerating comprehensive report...")
        os.makedirs("analysis_output", exist_ok=True)
        
        # Get pattern analysis data
        pattern_results = self.analyze_patterns()
        if not pattern_results:
            print("No data available for comprehensive report")
            return
        
        performance_counts = pattern_results['performance_counts']
        correlation = pattern_results['correlation']
        category_means = pattern_results['category_means']
        all_tokens = pattern_results['all_tokens']
        
        # Get best and worst performers
        best_tokens = all_tokens.sort_values('price_change_pct', ascending=False).head(10)
        worst_tokens = all_tokens.sort_values('price_change_pct', ascending=True).head(10)
        highest_volume = all_tokens.sort_values('total_volume_sol', ascending=False).head(10)
        
        # Start building the markdown report
        with open(output_file, 'w') as f:
            # Title and introduction
            f.write("# PumpFun Token Analysis Report\n\n")
            f.write(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"This report provides a comprehensive analysis of {len(all_tokens)} tokens collected from the PumpPortal platform.\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Total tokens analyzed:** {len(all_tokens)}\n")
            f.write(f"- **Average price change:** {all_tokens['price_change_pct'].mean():.2f}%\n")
            f.write(f"- **Median price change:** {all_tokens['price_change_pct'].median():.2f}%\n")
            f.write(f"- **Average trading volume:** {all_tokens['total_volume_sol'].mean():.4f} SOL\n")
            f.write(f"- **Average number of trades:** {all_tokens['total_trades'].mean():.1f}\n")
            f.write(f"- **Total trading volume:** {all_tokens['total_volume_sol'].sum():.4f} SOL\n\n")
            
            # Performance categories
            f.write("## Token Performance Categories\n\n")
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")
            for category, count in performance_counts.items():
                f.write(f"| {category} | {count} | {count/len(all_tokens)*100:.1f}% |\n")
            f.write("\n")
            
            # Performance metrics by category
            f.write("## Average Metrics by Performance Category\n\n")
            f.write("| Category | Avg Price Change | Avg Volume (SOL) | Buy/Sell Ratio | Avg # Trades |\n")
            f.write("|----------|-----------------|------------------|----------------|-------------|\n")
            for category, row in category_means.iterrows():
                f.write(f"| {category} | {row['price_change_pct']:.2f}% | {row['total_volume_sol']:.4f} | {row['buy_sell_ratio']:.2f} | {row['total_trades']:.1f} |\n")
            f.write("\n")
            
            # Correlation analysis
            f.write("## Metric Correlations\n\n")
            f.write("Correlation between key metrics:\n\n")
            f.write("```\n")
            f.write(correlation.to_string())
            f.write("\n```\n\n")
            
            # Top performers
            f.write("## Top 10 Performing Tokens\n\n")
            f.write("| Token | Symbol | Price Change | Volume (SOL) | Trades | Duration (min) |\n")
            f.write("|-------|--------|-------------|--------------|--------|----------------|\n")
            for _, token in best_tokens.iterrows():
                f.write(f"| {token['name']} | {token['symbol']} | {token['price_change_pct']:.2f}% | {token['total_volume_sol']:.4f} | {token['total_trades']} | {token['trading_duration_minutes']:.1f} |\n")
            f.write("\n")
            
            # Worst performers
            f.write("## Worst 10 Performing Tokens\n\n")
            f.write("| Token | Symbol | Price Change | Volume (SOL) | Trades | Duration (min) |\n")
            f.write("|-------|--------|-------------|--------------|--------|----------------|\n")
            for _, token in worst_tokens.iterrows():
                f.write(f"| {token['name']} | {token['symbol']} | {token['price_change_pct']:.2f}% | {token['total_volume_sol']:.4f} | {token['total_trades']} | {token['trading_duration_minutes']:.1f} |\n")
            f.write("\n")
            
            # Highest volume
            f.write("## Highest Volume Tokens\n\n")
            f.write("| Token | Symbol | Volume (SOL) | Price Change | Trades | Duration (min) |\n")
            f.write("|-------|--------|--------------|-------------|--------|----------------|\n")
            for _, token in highest_volume.iterrows():
                f.write(f"| {token['name']} | {token['symbol']} | {token['total_volume_sol']:.4f} | {token['price_change_pct']:.2f}% | {token['total_trades']} | {token['trading_duration_minutes']:.1f} |\n")
            f.write("\n")
            
            # Insights and conclusions
            f.write("## Key Insights\n\n")
            
            # Correlation insights
            price_volume_corr = correlation.loc['price_change_pct', 'total_volume_sol']
            price_trades_corr = correlation.loc['price_change_pct', 'total_trades']
            volume_trades_corr = correlation.loc['total_volume_sol', 'total_trades']
            
            f.write("### Correlation Analysis\n\n")
            f.write(f"- Price change and trading volume have a correlation of {price_volume_corr:.2f}. ")
            if price_volume_corr > 0.7:
                f.write("This indicates a strong positive relationship - tokens with higher trading volumes tend to perform better.\n")
            elif price_volume_corr > 0.3:
                f.write("This indicates a moderate positive relationship between volume and price performance.\n")
            else:
                f.write("This suggests a weak relationship between trading volume and price performance.\n")
            
            f.write(f"- Price change and number of trades have a correlation of {price_trades_corr:.2f}. ")
            if price_trades_corr > 0.7:
                f.write("This shows a strong connection between trading activity and token performance.\n")
            elif price_trades_corr > 0.3:
                f.write("This shows a moderate connection between trading activity and token performance.\n")
            else:
                f.write("This suggests trade count alone is not strongly predictive of performance.\n")
            
            f.write(f"- Volume and trade count correlation is {volume_trades_corr:.2f}. ")
            if volume_trades_corr > 0.9:
                f.write("These metrics are very highly correlated as expected.\n\n")
            else:
                f.write("Interestingly, these metrics are not as strongly correlated as might be expected.\n\n")
            
            # Performance distribution insights
            stable_pct = performance_counts.get('Stable', 0) / len(all_tokens) * 100
            gain_pct = (performance_counts.get('Small Gain', 0) + performance_counts.get('Large Gain', 0)) / len(all_tokens) * 100
            loss_pct = (performance_counts.get('Small Drop', 0) + performance_counts.get('Large Drop', 0)) / len(all_tokens) * 100
            
            f.write("### Performance Distribution\n\n")
            f.write(f"- {stable_pct:.1f}% of tokens remain relatively stable (between -10% and +10% change)\n")
            f.write(f"- {gain_pct:.1f}% of tokens show gains (>10% price increase)\n")
            f.write(f"- {loss_pct:.1f}% of tokens show losses (>10% price decrease)\n\n")
            
            # Volume analysis insights
            avg_volume = all_tokens['total_volume_sol'].mean()
            median_volume = all_tokens['total_volume_sol'].median()
            max_volume = all_tokens['total_volume_sol'].max()
            min_volume = all_tokens['total_volume_sol'].min()
            
            f.write("### Volume Analysis\n\n")
            f.write(f"- The average trading volume is {avg_volume:.4f} SOL, while the median is {median_volume:.4f} SOL\n")
            f.write(f"- The highest volume token has {max_volume:.4f} SOL in trades, while the lowest has {min_volume:.4f} SOL\n")
            f.write(f"- The large difference between average and median suggests a few high-volume tokens skew the distribution\n\n")
            
            # Final conclusions
            f.write("## Conclusions\n\n")
            
            # Determine most common outcome
            most_common_cat = performance_counts.idxmax()
            most_common_pct = performance_counts.max() / len(all_tokens) * 100
            
            f.write(f"1. The most common outcome for tokens is to remain in the '{most_common_cat}' category ({most_common_pct:.1f}% of tokens)\n")
            
            if price_volume_corr > 0.5 and price_trades_corr > 0.5:
                f.write("2. There is a strong correlation between trading activity (volume and count) and price performance\n")
            
            # Compare top performers
            top_performer = best_tokens.iloc[0]
            top_volume = highest_volume.iloc[0]
            
            if top_performer['name'] == top_volume['name']:
                f.write(f"3. The best performing token ({top_performer['name']}) is also the highest volume token, reinforcing the volume-performance relationship\n")
            else:
                f.write(f"3. The best performing token ({top_performer['name']}) and highest volume token ({top_volume['name']}) are different, suggesting factors beyond just trading volume influence performance\n")
            
            # Add chart references
            f.write("\n## Chart References\n\n")
            f.write("Individual token price and volume charts can be found in the analysis_output directory.\n")
            
        print(f"Comprehensive report saved to {output_file}")
        return output_file

    def generate_performance_distribution(self, save_path="analysis_output/performance_distribution.png"):
        """Generate a histogram of token price changes"""
        print("\nGenerating performance distribution chart...")
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create bins for histogram
        bins = [-100, -50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50, 100, 400]
        
        # Plot histogram with custom bins
        n, bins, patches = plt.hist(summary_df['price_change_pct'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add value labels above each bar
        for i in range(len(n)):
            if n[i] > 0:
                plt.text(
                    (bins[i] + bins[i+1]) / 2,  # x position (middle of bin)
                    n[i] + 0.5,                 # y position (slightly above bar)
                    f'{int(n[i])}',              # label (count)
                    ha='center',                # horizontal alignment
                    va='bottom'                 # vertical alignment
                )
        
        # Add grid, title and labels
        plt.grid(axis='y', alpha=0.3)
        plt.title('Distribution of Token Price Changes', fontsize=16)
        plt.xlabel('Price Change (%)', fontsize=14)
        plt.ylabel('Number of Tokens', fontsize=14)
        
        # Add mean and median lines
        mean_change = summary_df['price_change_pct'].mean()
        median_change = summary_df['price_change_pct'].median()
        
        plt.axvline(mean_change, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_change:.2f}%)')
        plt.axvline(median_change, color='green', linestyle='dashed', linewidth=2, label=f'Median ({median_change:.2f}%)')
        
        # Add legend
        plt.legend()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Performance distribution chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return save_path
    
    def generate_correlation_heatmap(self, save_path="analysis_output/correlation_heatmap.png"):
        """Generate a heatmap of correlations between token metrics"""
        print("\nGenerating correlation heatmap...")
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available")
            return
            
        # Select numeric columns for correlation
        numeric_cols = ['price_change_pct', 'total_volume_sol', 'buy_sell_ratio', 'total_trades', 'trading_duration_minutes']
        corr_data = summary_df[numeric_cols].corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            corr_data, 
            annot=True,           # Show correlation values
            cmap='coolwarm',      # Red-blue color scheme
            vmin=-1, vmax=1,      # Correlation ranges from -1 to 1
            center=0,             # Center color map at 0
            square=True,          # Make cells square
            linewidths=0.5,       # Add lines between cells
            fmt='.2f',            # Format for annotation (2 decimal places)
            cbar_kws={'label': 'Correlation Coefficient'}  # Add label to color bar
        )
        
        # Add title and adjust labels
        plt.title('Correlation Between Token Metrics', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return save_path
    
    def generate_compare_top_tokens(self, n=5, metric='price_change_pct', save_path="analysis_output/top_tokens_comparison.png"):
        """Generate a chart comparing price movements of top tokens"""
        print(f"\nGenerating comparison chart for top {n} tokens by {metric}...")
        summary_df = self.analyze_all_tokens()
        
        if summary_df.empty:
            print("No token data available")
            return
            
        # Get top tokens by the specified metric
        if metric == 'price_change_pct':
            top_tokens = summary_df.sort_values(metric, ascending=False).head(n)
            title_metric = "Price Change"
        elif metric == 'total_volume_sol':
            top_tokens = summary_df.sort_values(metric, ascending=False).head(n)
            title_metric = "Trading Volume"
        else:
            # Default to price change if invalid metric
            top_tokens = summary_df.sort_values('price_change_pct', ascending=False).head(n)
            title_metric = "Price Change"
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # For each top token, plot its price movement
        for _, token_row in top_tokens.iterrows():
            token_address = token_row['address']
            token_symbol = token_row['symbol']
            
            # Get detailed data for this token if not already analyzed
            if token_address not in self.analysis_results:
                self.analyze_token(token_address)
                
            if token_address in self.analysis_results:
                token_data = self.analysis_results[token_address]
                trades_df = token_data['trades_df']
                
                # Normalize price to percentage of initial price for comparison
                if not trades_df.empty and 'minutes_since_creation' in trades_df.columns:
                    initial_price = trades_df['price'].iloc[0]
                    
                    # Calculate percentage change from initial price
                    normalized_prices = (trades_df['price'] / initial_price - 1) * 100
                    
                    # Plot this token's price movement
                    plt.plot(
                        trades_df['minutes_since_creation'], 
                        normalized_prices, 
                        marker='.', 
                        alpha=0.7,
                        label=f"{token_symbol} ({token_row['price_change_pct']:.1f}%)"
                    )
        
        # Add labels and title
        plt.xlabel('Minutes Since Creation', fontsize=12)
        plt.ylabel('Price Change from Initial (%)', fontsize=12)
        plt.title(f'Price Movement Comparison: Top {n} Tokens by {title_metric}', fontsize=16)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add horizontal line at 0% change
        plt.axhline(0, color='black', linestyle='-', alpha=0.2)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Top tokens comparison chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return save_path
    
    def generate_time_since_creation_analysis(self, save_path="analysis_output/time_performance_analysis.png"):
        """Generate a chart showing average performance at different time points after creation"""
        print("\nGenerating time-based performance analysis...")
        all_tokens = self.analyze_all_tokens()
        
        if all_tokens.empty:
            print("No token data available")
            return
        
        # We need detailed trade data for each token
        time_points = [1, 2, 3, 5, 10, 15, 30]  # minutes after creation
        performance_data = {t: [] for t in time_points}
        
        # For each token, analyze its performance at different time points
        for _, token_row in all_tokens.iterrows():
            token_address = token_row['address']
            
            # Get detailed data for this token if not already analyzed
            if token_address not in self.analysis_results:
                self.analyze_token(token_address)
                
            if token_address in self.analysis_results:
                token_data = self.analysis_results[token_address]
                trades_df = token_data['trades_df']
                
                if not trades_df.empty and 'minutes_since_creation' in trades_df.columns:
                    initial_price = trades_df['price'].iloc[0]
                    
                    # For each time point, find the price if available
                    for time_point in time_points:
                        # Find the price closest to this time point
                        closest_rows = trades_df[trades_df['minutes_since_creation'] <= time_point]
                        if not closest_rows.empty:
                            latest_price = closest_rows.iloc[-1]['price']
                            pct_change = ((latest_price / initial_price) - 1) * 100
                            performance_data[time_point].append(pct_change)
        
        # Calculate averages and statistics
        avg_performance = {t: np.mean(perf) if perf else 0 for t, perf in performance_data.items()}
        median_performance = {t: np.median(perf) if perf else 0 for t, perf in performance_data.items()}
        count_tokens = {t: len(perf) for t, perf in performance_data.items()}
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create time point labels
        time_labels = [f"{t} min\n({count_tokens[t]} tokens)" for t in time_points]
        
        # Plot average performance
        plt.bar(
            time_labels,
            [avg_performance[t] for t in time_points],
            alpha=0.7,
            color='skyblue',
            width=0.6,
            label='Mean Performance'
        )
        
        # Add median line
        plt.plot(
            time_labels,
            [median_performance[t] for t in time_points],
            marker='o',
            color='red',
            linestyle='-',
            linewidth=2,
            label='Median Performance'
        )
        
        # Add value labels above each bar
        for i, t in enumerate(time_points):
            plt.text(
                i,                               # x position (bar index)
                avg_performance[t] + 1,          # y position (slightly above bar)
                f'{avg_performance[t]:.1f}%',    # label (avg performance)
                ha='center',                     # horizontal alignment
                va='bottom',                     # vertical alignment
                fontweight='bold'
            )
        
        # Add labels and title
        plt.xlabel('Time Since Token Creation', fontsize=12)
        plt.ylabel('Average Price Change (%)', fontsize=12)
        plt.title('Token Performance at Different Time Points After Creation', fontsize=16)
        
        # Add grid and legend
        plt.grid(axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add horizontal line at 0% change
        plt.axhline(0, color='black', linestyle='-', alpha=0.2)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Time-based performance analysis saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return save_path

    def generate_all_visualizations(self):
        """Generate all available visualizations"""
        print("\nGenerating all visualizations...")
        os.makedirs("analysis_output", exist_ok=True)
        
        # Specific token analysis
        token_address = "V9MfAsmXmEywPdSmopXJgVPLjBoybQuGp1BHZDYpump"
        if token_address not in self.analysis_results:
            self.analyze_token(token_address)
        
        # Generate individual token charts
        if token_address in self.analysis_results:
            token_info = self.analysis_results[token_address]['token_info']
            self.generate_price_chart(token_address, f"analysis_output/{token_info['symbol']}_price_chart.png")
            self.generate_volume_chart(token_address, f"analysis_output/{token_info['symbol']}_volume_chart.png")
        
        # Generate aggregate visualizations
        self.generate_performance_distribution()
        self.generate_correlation_heatmap()
        self.generate_compare_top_tokens(n=5, metric='price_change_pct')
        self.generate_compare_top_tokens(n=5, metric='total_volume_sol', save_path="analysis_output/top_volume_tokens_comparison.png")
        self.generate_time_since_creation_analysis()
        
        print("All visualizations completed and saved to the analysis_output directory.")

def main():
    analyzer = PumpfunAnalyzer()
    
    # Create output directory
    os.makedirs("analysis_output", exist_ok=True)
    
    # Example 1: Analyze a specific token
    print("\nAnalyzing specific token...")
    token_address = "V9MfAsmXmEywPdSmopXJgVPLjBoybQuGp1BHZDYpump"
    analyzer.analyze_specific_token(token_address)
    
    # Example 2: Generate a summary report
    print("\nGenerating summary report...")
    summary_df = analyzer.generate_summary_report("analysis_output/token_summary.csv")
    
    # Example 3: Find best performing tokens
    print("\nFinding best performers...")
    best_tokens = analyzer.find_best_performing_tokens(5)
    
    # Example 4: Analyze patterns
    print("\nAnalyzing patterns...")
    patterns = analyzer.analyze_patterns()
    
    # New: Generate comprehensive document report
    analyzer.generate_comprehensive_report("analysis_output/comprehensive_report.md")
    
    # New: Generate all visualizations
    analyzer.generate_all_visualizations()
    
    print("\nAnalysis complete! Results saved to the analysis_output directory.")

if __name__ == "__main__":
    main() 