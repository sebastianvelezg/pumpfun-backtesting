import asyncio
import websockets
import json
import time
from datetime import datetime
import os
import signal
import shutil
from pathlib import Path

class PumpfunDataCollector:
    def __init__(self, data_dir="pumpfun_data"):
        self.uri = "wss://pumpportal.fun/api/data"
        self.active_tokens = {}  # In-memory token data
        self.modified_tokens = set()  # Track which tokens have new data
        self.data_dir = data_dir
        self.tokens_dir = os.path.join(data_dir, "tokens")
        self.index_file = os.path.join(data_dir, "token_index.json")
        self.token_index = {}  # Basic info about all tokens
        self.last_save_time = time.time()
        self.save_interval = 120  # Save every 2 minutes (120 seconds)
        
        # Create data directories if they don't exist
        self._create_data_directories()
        
        # Load existing token index if available
        self._load_token_index()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def _create_data_directories(self):
        """Create the directory structure for data storage"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.tokens_dir, exist_ok=True)
        print(f"[{self.get_timestamp()}] 📁 Data directory structure created at {self.data_dir}")
    
    def signal_handler(self, sig, frame):
        """Handle exit signals gracefully by saving data"""
        print(f"\n[{self.get_timestamp()}] 💾 Received exit signal, saving data before exiting...")
        self.save_all_data()
        print(f"[{self.get_timestamp()}] 👋 Data saved. Exiting.")
        exit(0)
        
    def _load_token_index(self):
        """Load the token index file if it exists"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.token_index = json.load(f)
                    token_count = len(self.token_index)
                    print(f"[{self.get_timestamp()}] 📂 Loaded index for {token_count} tokens")
            except Exception as e:
                print(f"[{self.get_timestamp()}] ⚠️ Error loading token index: {e}")
                self.token_index = {}
    
    def _get_token_filepath(self, token_address):
        """Get the filepath for a specific token's data file"""
        # Use first 2 chars of address as subdirectory to spread files around
        if len(token_address) >= 2:
            subdir = token_address[:2]
            token_dir = os.path.join(self.tokens_dir, subdir)
            os.makedirs(token_dir, exist_ok=True)
            return os.path.join(token_dir, f"{token_address}.json")
        else:
            return os.path.join(self.tokens_dir, f"{token_address}.json")
    
    def load_token_data(self, token_address):
        """Load data for a specific token"""
        filepath = self._get_token_filepath(token_address)
        if not os.path.exists(filepath):
            return {}
            
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[{self.get_timestamp()}] ⚠️ Error loading token data for {token_address}: {e}")
            return {}
    
    def save_token_data(self, token_address):
        """Save data for a specific token"""
        if token_address not in self.active_tokens:
            return
            
        filepath = self._get_token_filepath(token_address)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.active_tokens[token_address], f, indent=2)
            # Update the last saved time for this token in the index
            if token_address in self.token_index:
                self.token_index[token_address]['last_saved'] = time.time()
            self.modified_tokens.discard(token_address)  # Remove from modified tokens set
        except Exception as e:
            print(f"[{self.get_timestamp()}] ❌ Error saving token data for {token_address}: {e}")
    
    def save_token_index(self):
        """Save the token index file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.token_index, f, indent=2)
            print(f"[{self.get_timestamp()}] 💾 Saved index for {len(self.token_index)} tokens")
        except Exception as e:
            print(f"[{self.get_timestamp()}] ❌ Error saving token index: {e}")
    
    def save_modified_tokens(self):
        """Save only tokens that have been modified since last save"""
        if not self.modified_tokens:
            print(f"[{self.get_timestamp()}] No modified tokens to save")
            return
            
        for token_address in list(self.modified_tokens):
            self.save_token_data(token_address)
            
        # Update the token index after saving all modified tokens
        self.save_token_index()
        print(f"[{self.get_timestamp()}] 💾 Saved {len(self.modified_tokens)} modified tokens")
        self.modified_tokens.clear()
        self.last_save_time = time.time()
    
    def save_all_data(self):
        """Save all token data and index"""
        # Save all active tokens
        tokens_saved = 0
        for token_address in self.active_tokens:
            self.save_token_data(token_address)
            tokens_saved += 1
            
        # Save the token index
        self.save_token_index()
        
        print(f"[{self.get_timestamp()}] 💾 Full save completed: {tokens_saved} tokens")
        self.modified_tokens.clear()
        self.last_save_time = time.time()
    
    async def connect_and_monitor(self):
        """Connect to WebSocket and monitor token activity"""
        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"[{self.get_timestamp()}] Connected to PumpPortal WebSocket")
                
                # Subscribe to new token creation events
                await self.subscribe_new_tokens(websocket)
                
                # Main processing loop
                async for message in websocket:
                    await self.process_message(websocket, message)
                    
                    # Check if it's time to save data
                    if time.time() - self.last_save_time > self.save_interval:
                        self.save_modified_tokens()
                    
        except Exception as e:
            print(f"[{self.get_timestamp()}] Connection error: {e}")
            # Save data before attempting to reconnect
            self.save_modified_tokens()
            # Try to reconnect after a delay
            print(f"[{self.get_timestamp()}] Attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)
            return await self.connect_and_monitor()
    
    async def subscribe_new_tokens(self, websocket):
        """Subscribe to new token creation events"""
        payload = {
            "method": "subscribeNewToken"
        }
        await websocket.send(json.dumps(payload))
        print(f"[{self.get_timestamp()}] Sent subscription request for new tokens")
    
    async def subscribe_token_trades(self, websocket, token_address):
        """Subscribe to trades for a specific token"""
        payload = {
            "method": "subscribeTokenTrade",
            "keys": [token_address]
        }
        await websocket.send(json.dumps(payload))
        print(f"[{self.get_timestamp()}] Subscribed to trades for token: {token_address}")
    
    async def process_message(self, websocket, message_str):
        """Process incoming WebSocket messages"""
        try:
            # Print raw message for debugging
            print(f"[{self.get_timestamp()}] RAW MESSAGE: {message_str[:200]}...")
            
            message = json.loads(message_str)
            
            # Check if message is a success confirmation
            if "message" in message and "Successfully subscribed" in message["message"]:
                print(f"[{self.get_timestamp()}] ✅ {message['message']}")
                return
                
            # Check for error messages
            if "errors" in message:
                print(f"[{self.get_timestamp()}] ❌ Error: {message['errors']}")
                return
                
            # Check if message is a token creation event (has "txType": "create")
            if "txType" in message and message["txType"] == "create" and "mint" in message:
                await self.handle_new_token(websocket, message)
                return
                
            # Check if message is a trade event (has "txType" but not "create")
            if "txType" in message and message["txType"] != "create" and "mint" in message:
                await self.handle_trade(message)
                return
                
            # Handle unrecognized message format
            print(f"[{self.get_timestamp()}] ❓ Unrecognized message format")
            
        except json.JSONDecodeError:
            print(f"[{self.get_timestamp()}] Failed to parse message: {message_str[:100]}...")
        except Exception as e:
            print(f"[{self.get_timestamp()}] Error processing message: {e}")
    
    async def handle_new_token(self, websocket, token_data):
        """Handle new token creation events"""
        token_address = token_data.get('mint')
        token_name = token_data.get('name', 'Unknown')
        token_symbol = token_data.get('symbol', 'UNKNOWN')
        
        print(f"\n[{self.get_timestamp()}] 🆕 NEW TOKEN CREATED:")
        print(f"  Name: {token_name} ({token_symbol})")
        print(f"  Address: {token_address}")
        print(f"  Initial Buy: {token_data.get('initialBuy', 'Unknown')} tokens")
        print(f"  SOL Amount: {token_data.get('solAmount', 'Unknown')} SOL")
        print(f"  Market Cap: {token_data.get('marketCapSol', 'Unknown')} SOL")
        
        # Check if we already have data for this token
        existing_data = self.load_token_data(token_address)
        
        # Store token info with creation timestamp
        token_info = {
            'name': token_name,
            'symbol': token_symbol,
            'created_at': self.get_timestamp(),
            'creation_time': time.time(),
            'creation_data': token_data,
            'trades': existing_data.get('trades', [])  # Preserve existing trades if any
        }
        
        # Update in-memory representation
        self.active_tokens[token_address] = token_info
        
        # Add to index with basic info
        self.token_index[token_address] = {
            'name': token_name,
            'symbol': token_symbol,
            'created_at': self.get_timestamp(),
            'creation_time': time.time(),
            'first_seen': self.get_timestamp(),
            'trade_count': len(token_info['trades']),
            'last_saved': time.time()
        }
        
        # Mark as modified for next save
        self.modified_tokens.add(token_address)
        
        # Subscribe to trades for this token
        await self.subscribe_token_trades(websocket, token_address)
    
    async def handle_trade(self, trade_data):
        """Handle trade events for tokens"""
        token_address = trade_data.get('mint')
        
        if not token_address:
            return
            
        # If token not in active memory, load it from disk if it exists
        if token_address not in self.active_tokens:
            token_data = self.load_token_data(token_address)
            if token_data:
                self.active_tokens[token_address] = token_data
            else:
                # Create new entry if we've never seen this token before
                self.active_tokens[token_address] = {
                    'name': 'Unknown',
                    'symbol': 'UNKNOWN',
                    'first_seen': self.get_timestamp(),
                    'first_seen_time': time.time(),
                    'trades': []
                }
                
                # Add to index
                self.token_index[token_address] = {
                    'name': 'Unknown',
                    'symbol': 'UNKNOWN',
                    'first_seen': self.get_timestamp(),
                    'first_seen_time': time.time(),
                    'trade_count': 0,
                    'last_saved': time.time()
                }
            
        # Extract trade information
        token = self.active_tokens[token_address]
        token_amount = trade_data.get('tokenAmount', 0)
        sol_amount = trade_data.get('solAmount', 0)
        tx_type = trade_data.get('txType', 'unknown')
        
        # Calculate approximate price
        price = sol_amount / token_amount if token_amount > 0 else 0
        
        # Create trade record
        trade_record = {
            'timestamp': self.get_timestamp(),
            'unix_time': time.time(),
            'type': tx_type,
            'token_amount': token_amount,
            'sol_amount': sol_amount,
            'price': price,
            'market_cap': trade_data.get('marketCapSol', 0),
            'data': trade_data  # Store full trade data
        }
        
        # Add trade to token's trade history
        if 'trades' not in token:
            token['trades'] = []
        token['trades'].append(trade_record)
        
        # Update the token index
        if token_address in self.token_index:
            self.token_index[token_address]['trade_count'] = len(token['trades'])
            self.token_index[token_address]['last_price'] = price
            self.token_index[token_address]['last_trade_time'] = self.get_timestamp()
        
        # Mark as modified for next save
        self.modified_tokens.add(token_address)
        
        # Print trade information
        print(f"\n[{self.get_timestamp()}] 💰 TRADE: {token.get('name', 'Unknown')} ({token.get('symbol', 'Unknown')})")
        print(f"  Type: {tx_type.upper()}")
        print(f"  Tokens: {token_amount}")
        print(f"  SOL: {sol_amount}")
        print(f"  Price: {price:.10f} SOL per token")
        print(f"  Market Cap: {trade_data.get('marketCapSol', 'Unknown')} SOL")
        
        # Update token's last trade info
        token['last_price'] = price
        token['last_trade_time'] = self.get_timestamp()
        
        # If this is the first trade we've seen, note it
        if 'first_trade_time' not in token:
            token['first_trade_time'] = self.get_timestamp()
            token['first_price'] = price
            print(f"  ⭐ First trade recorded for this token")
    
    @staticmethod
    def get_timestamp():
        """Get current timestamp in readable format"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def get_collection_stats(self):
        """Get statistics about the data collection"""
        total_tokens = len(self.token_index)
        total_trades = sum(info.get('trade_count', 0) for info in self.token_index.values())
        
        return {
            'total_tokens': total_tokens,
            'total_trades': total_trades,
            'active_tokens': len(self.active_tokens),
            'modified_tokens': len(self.modified_tokens),
            'last_save_time': datetime.fromtimestamp(self.last_save_time).strftime('%Y-%m-%d %H:%M:%S') 
        }

async def main():
    print("Starting PumpPortal Data Collector with per-token storage...")
    data_dir = "pumpfun_data"
    collector = PumpfunDataCollector(data_dir)
    
    # Start collection time
    start_time = time.time()
    
    try:
        # Show periodic stats while running
        stats_task = asyncio.create_task(show_periodic_stats(collector))
        
        # Run the main collector
        await collector.connect_and_monitor()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user. Saving data before exiting...")
        collector.save_all_data()
        print("Data saved. Exiting.")
    except Exception as e:
        print(f"Error in monitor: {e}")
        collector.save_all_data()
    
    # Calculate total run time
    run_time = time.time() - start_time
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total run time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Show final stats
    stats = collector.get_collection_stats()
    print("\nFinal collection statistics:")
    print(f"  Total tokens tracked: {stats['total_tokens']}")
    print(f"  Total trades recorded: {stats['total_trades']}")

async def show_periodic_stats(collector, interval=600):
    """Show periodic statistics about the data collection"""
    try:
        while True:
            await asyncio.sleep(interval)  # Show stats every 10 minutes
            stats = collector.get_collection_stats()
            print("\n" + "="*50)
            print(f"[{collector.get_timestamp()}] COLLECTION STATISTICS:")
            print(f"  Tokens tracked: {stats['total_tokens']}")
            print(f"  Trades recorded: {stats['total_trades']}")
            print(f"  Tokens in memory: {stats['active_tokens']}")
            print(f"  Last save: {stats['last_save_time']}")
            print("="*50 + "\n")
    except asyncio.CancelledError:
        pass  # Task was cancelled, exit quietly

if __name__ == "__main__":
    asyncio.run(main())