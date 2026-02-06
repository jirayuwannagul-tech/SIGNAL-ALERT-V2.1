"""Signal History Manager - Persistent storage for 1D signals"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SignalHistoryManager:
    """Manage 1D signal history with persistent JSON storage"""
    
    def __init__(self, data_dir: str = "/data"):
        """Initialize signal history manager
        
        Args:
            data_dir: Directory for storing signal history (default: /data for Railway Volume)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.data_dir / "signal_history_1d.json"
        self.signal_history = self._load_history()
        
        logger.info(f"✅ SignalHistoryManager initialized (File: {self.history_file})")
    
    def _load_history(self) -> Dict:
        """Load signal history from JSON file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded {len(data)} signal histories")
                    return data
            else:
                logger.info("📂 No existing history file, starting fresh")
                return {}
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return {}
    
    def _save_history(self):
        """Save signal history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.signal_history, f, indent=2)
            logger.debug(f"💾 Saved signal history ({len(self.signal_history)} entries)")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def should_notify(self, symbol: str, timeframe: str, signal_type: str, current_price: float) -> bool:
        """Check if should notify for this signal
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 1d)
            signal_type: Signal type (LONG or SHORT)
            current_price: Current price
            
        Returns:
            True if should notify, False if already notified
        """
        key = f"{symbol}_{timeframe}_{signal_type}"
        
        # Check if exists
        if key not in self.signal_history:
            return True  # New signal, should notify
        
        last_signal = self.signal_history[key]
        
        # Check if same signal type
        if last_signal.get("signal_type") != signal_type:
            return True  # Signal type changed, should notify
        
        # Already notified for same signal
        logger.debug(f"⏭️ Skip: {key} already notified on {last_signal.get('date')}")
        return False
    
    def record_signal(self, symbol: str, timeframe: str, signal_type: str, current_price: float):
        """Record that signal was notified
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal_type: Signal type (LONG or SHORT)
            current_price: Current price
        """
        key = f"{symbol}_{timeframe}_{signal_type}"
        
        self.signal_history[key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_type": signal_type,
            "price": current_price,
            "date": datetime.now().isoformat(),
            "notified": True
        }
        
        self._save_history()
        logger.info(f"📝 Recorded: {key} @ {current_price}")
    
    def clear_opposite_signal(self, symbol: str, timeframe: str, signal_type: str):
        """Clear opposite signal when trend changes
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal_type: Current signal type (will clear opposite)
        """
        opposite = "SHORT" if signal_type == "LONG" else "LONG"
        key = f"{symbol}_{timeframe}_{opposite}"
        
        if key in self.signal_history:
            del self.signal_history[key]
            self._save_history()
            logger.info(f"🗑️ Cleared opposite signal: {key}")
    
    def get_history(self, symbol: Optional[str] = None) -> Dict:
        """Get signal history
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dictionary of signal history
        """
        if symbol:
            return {k: v for k, v in self.signal_history.items() if v['symbol'] == symbol}
        return self.signal_history
    
    def clear_history(self):
        """Clear all signal history (for testing)"""
        self.signal_history = {}
        self._save_history()
        logger.info("🗑️ Cleared all signal history")
    
    def get_stats(self) -> Dict:
        """Get statistics about signal history"""
        total = len(self.signal_history)
        long_count = sum(1 for v in self.signal_history.values() if v['signal_type'] == 'LONG')
        short_count = sum(1 for v in self.signal_history.values() if v['signal_type'] == 'SHORT')
        
        return {
            "total_signals": total,
            "long_signals": long_count,
            "short_signals": short_count,
            "file_path": str(self.history_file),
            "file_exists": self.history_file.exists()
        }