"""
Main Entrypoint
===============
Production entrypoint for Railway deployment.
Runs the bot continuously with Bybit WebSocket and persistent state storage.
"""

import os
import sys
import time
import logging
import signal
from pathlib import Path
from datetime import datetime

# Add mudrex-sdk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mudrex-sdk'))

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import get_config
from supertrend_mudrex_bot import SupertrendMudrexBot
from data_manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("main")

class StateManager:
    """Manages bot state in a persistent JSON file."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> dict:
        """Load state from file."""
        if self.filepath.exists():
            try:
                import json
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {}
    
    def save(self, state: dict) -> None:
        """Save state to file."""
        try:
            import json
            tmp_path = self.filepath.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            tmp_path.replace(self.filepath)
            logger.debug(f"State saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, stopping bot...")
    sys.exit(0)

def main():
    """Main execution loop."""
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    logger.info("Starting Mudrex Supertrend Bot (Bybit WebSocket Edition)")
    
    # Load configuration
    try:
        config = get_config()
        config.validate()
    except ValueError as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
        
    logger.info(f"Mode: {'DRY RUN' if config.trading.dry_run else 'LIVE TRADING'}")
    logger.info("Config: timeframe=%s, margin_percent=%s%%, max_positions=%s", config.trading.timeframe, config.trading.margin_percent, config.trading.max_positions)

    # Initialize Data Manager
    data_manager = DataManager(
        interval=config.trading.timeframe,
        lookback=config.trading.lookback_periods
    )
    data_manager.start()
    
    # Initialize state manager
    data_dir = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/app/data")
    state_file = os.path.join(data_dir, "bot_state.json")
    state_manager = StateManager(state_file)
    
    # Initialize bot
    bot = SupertrendMudrexBot(config, data_manager=data_manager)
    
    # Load initial state
    state = state_manager.load()
    if state:
        bot.load_state(state)
        logger.info("Restored previous state")
    
    # Main loop
    interval = 300  # 5 minutes
    
    try:
        # Give some time for initial data to arrive
        logger.info("Waiting for initial market data from Bybit...")
        time.sleep(15)
        
        while True:
            start_time = time.time()
            
            try:
                # Run one cycle
                result = bot.run_once()
                
                # Save state
                new_state = bot.save_state()
                state_manager.save(new_state)
                
            except Exception as e:
                logger.exception(f"Error in execution cycle: {e}")
            
            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = max(10, interval - elapsed)
            logger.info(f"Cycle complete. Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            
    finally:
        bot.close()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    main()
