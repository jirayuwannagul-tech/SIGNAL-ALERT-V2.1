"""Signal detection logic combining all indicators - CONSERVATIVE MODE v2.0"""

import logging
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple

from .indicators import TechnicalIndicators
from ..utils.core_utils import ErrorHandler
from ..utils.data_types import DataConverter

logger = logging.getLogger(__name__)


class SignalDetector:
    """Detect trading signals using Squeeze + MACD Uncle Cholok + RSI strategy - CONSERVATIVE"""

    def __init__(self, config: Dict):
        """Initialize signal detector with refactored services"""
        # Extract refactored services
        self.data_manager = config["data_manager"]
        self.position_manager = config["position_manager"]
        self.config_manager = config["config_manager"]
        
        # Initialize utilities
        self.indicators = TechnicalIndicators()
        self.data_converter = DataConverter()
        self.active_positions = set()
        
        # Get configuration from ConfigManager
        self.risk_management = self._load_risk_config()
        self.indicator_settings = self._load_indicator_config()
        
        logger.info("✅ SignalDetector initialized - CONSERVATIVE MODE")
    
    def _load_risk_config(self) -> Dict:
        """Load risk management configuration from Config"""
        try:
            # ✅ อ่านจาก Config class
            from config.settings import Config
            return Config.RISK_MANAGEMENT
        except Exception as e:
            logger.warning(f"Error loading risk config, using defaults: {e}")
            return {
                "4h": {"tp_levels": [1.2, 1.5, 2.0], "sl_level": 1.5},
                "1d": {"tp_levels": [3.0, 5.0, 7.0], "sl_level": 3.0}
            }
    
    def _load_indicator_config(self) -> Dict:
        try:
            return {
                "squeeze": {"length": 20, "bb_mult": 2.0, "kc_mult": 1.5},
                "macd": {"fast": 8, "slow": 17, "signal": 9}, 
                "rsi": {
                    "period": 14, 
                    "oversold": 35,  # ✅ แก้จาก 40 เป็น 35
                    "overbought": 65 # ✅ แก้จาก 60 เป็น 65
                }
            }
            
        except Exception as e:
            logger.warning(f"Error loading indicator config, using defaults: {e}")
            return {
                "squeeze": {"length": 20, "bb_mult": 2.0, "kc_mult": 1.5},
                "macd": {"fast": 8, "slow": 17, "signal": 9}, 
                "rsi": {"period": 14, "oversold": 40, "overbought": 60}
            }

    @ErrorHandler.service_error_handler("SignalDetector")
    def analyze_symbol(self, symbol: str, timeframe: str = "4h") -> Optional[Dict]:
        """Analyze symbol using refactored data flow"""
        try:
            logger.info(f"🔍 Analyzing {symbol} on {timeframe} (CONSERVATIVE)")

            # Get data from DataManager
            df = self.data_manager.get_klines(symbol, timeframe, limit=100)

            df_1d = self.data_manager.get_klines(symbol, "1d", limit=50)
            trend_1d = self._detect_signals_improved_fixed(None, "1d", df_1d)

            if df is None:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return {
                    "error": f"Failed to fetch data for {symbol}",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0-conservative"
                }

            # Validate data quality
            if not self.data_converter.validate_dataframe(df):
                logger.warning(f"Invalid DataFrame for {symbol} {timeframe}")
                return None

            # Calculate all indicators
            analysis = self.indicators.analyze_all_indicators(df, self.indicator_settings)

            # Detect trading signals with CONSERVATIVE logic
            signals = self._detect_signals_improved_fixed(analysis, timeframe, df, trend_1d=trend_1d)

            # Calculate risk management levels  
            risk_levels = self._calculate_risk_levels(analysis["current_price"], timeframe, signals, symbol)

            # Handle position creation with duplicate prevention
            position_created = self._handle_signal_position_fixed(
                symbol, timeframe, signals, analysis["current_price"], risk_levels
            )

            # Create comprehensive result
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "current_price": analysis["current_price"],
                "version": "2.0-conservative",
                
                # Indicator values
                "indicators": {
                    "squeeze": analysis["squeeze"],
                    "macd": analysis["macd"],
                    "rsi": analysis["rsi"],
                },
                
                # Trading signals
                "signals": signals,
                
                # Risk management
                "risk_levels": risk_levels,
                
                # Overall assessment
                "signal_strength": self._calculate_signal_strength_improved(signals),
                "recommendation": self._get_recommendation_improved(signals),
                
                # Position info
                "position_created": position_created,
                "has_active_position": self._has_active_position_strict(symbol, timeframe),
            }

            # Convert NumPy types for JSON serialization
            result = self.data_converter.sanitize_signal_data(result)

            # Log significant results
            if result.get('recommendation'):
                logger.info(f"Analysis complete for {symbol}: {result['recommendation']}")
                if position_created:
                    logger.info(f"🆕 Created position for {symbol} {timeframe}")
            
            return result

        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {str(e)}")
            return {
                "error": f"Analysis error for {symbol}: {str(e)}",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "version": "2.0-conservative"
            }

    def _has_active_position_strict(self, symbol: str, timeframe: str) -> bool:
        """✅ STRICT check if position exists - prevent duplicates"""
        try:
            # Check 1: Via PositionManager
            position = self.position_manager.get_position_status(symbol, timeframe)
            if position is not None:
                logger.debug(f"Found active position via PositionManager: {symbol} {timeframe}")
                return True
            
            # Check 2: Check all direction combinations
            for direction in ["LONG", "SHORT"]:
                position_id = f"{symbol}_{timeframe}_{direction}"
                if position_id in self.position_manager.positions:
                    pos_data = self.position_manager.positions[position_id]
                    if pos_data.get('status') == 'ACTIVE':
                        logger.debug(f"Found active position by ID: {position_id}")
                        return True
            
            # Check 3: In active_positions set
            if symbol in self.active_positions:
                logger.debug(f"Found in active_positions set: {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking active position: {e}")
            return True  # Return True on error to prevent duplicate

    @ErrorHandler.service_error_handler("SignalDetector") 
    def _handle_signal_position_fixed(
        self, symbol: str, timeframe: str, signals: Dict, current_price: float, risk_levels: Dict
    ) -> bool:
        """Handle position creation - only blocks if active position exists"""
        try:
            # Check if we have a valid signal
            has_signal = signals.get("buy") or signals.get("short")
        
            if not has_signal:
                return False
        
            # Check if active position exists (no cooldown check)
            if symbol in self.active_positions:
                logger.warning(f"⚠️ {symbol} already in active positions set")
                return False

            # Check via PositionManager strictly
            existing_position = self.position_manager.get_position_status(symbol, timeframe)
            if existing_position:
                logger.warning(f"⚠️ Position already exists: {symbol} {timeframe}")
                return False
        
            # Check all possible position IDs
            direction = "LONG" if signals.get("buy") else "SHORT"
            for dir_check in ["LONG", "SHORT"]:
                check_id = f"{symbol}_{timeframe}_{dir_check}"
                if check_id in self.position_manager.positions:
                    if self.position_manager.positions[check_id].get('status') == 'ACTIVE':
                        logger.warning(f"⚠️ Found active {dir_check} position: {symbol} {timeframe}")
                        return False
        
            # Create signal data for position creation
            signal_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "current_price": current_price,
                "signal_strength": self._calculate_signal_strength_improved(signals)
            }
        
            # Create position using PositionManager
            position_id = self.position_manager.create_position(signal_data)
        
            if position_id:
                logger.info(f"✅ Created {direction} position: {symbol} {timeframe} @ {current_price}")
            
                # Update tracking (no cooldown timestamp)
                self.active_positions.add(symbol)
            
                return True
            else:
                logger.warning(f"Failed to create position for {symbol} {timeframe}")
                return False
            
        except Exception as e:
            logger.error(f"Error handling signal position: {e}")
            return False

    def _detect_signals_improved_fixed(self, analysis: Dict, timeframe: str = "1d", df=None, trend_1d=None) -> Dict[str, bool]:
        """
        1D: CDC ActionZone (EMA 12/26 Crossover)
        4H: RSI + MACD + Enhanced filters + STRONG MOMENTUM MODE + PULLBACK MODE
        """
        try:
            import pandas as pd

            # Validate dataframe
            if df is None or 'close' not in df.columns:
                logger.warning("Invalid dataframe")
                return {"buy": False, "short": False, "sell": False, "cover": False}
            
            # ========================================
            # 1D: CDC ACTIONZONE (EMA CROSSOVER)
            # ========================================
            if timeframe == "1d":
                if len(df) < 30:
                    logger.warning(f"Insufficient data: {len(df)} candles")
                    return {"buy": False, "short": False, "sell": False, "cover": False}
                
                # Calculate EMA 12 and 26
                df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
                
                # Current values
                ema12_curr = df['ema12'].iloc[-1]
                ema26_curr = df['ema26'].iloc[-1]
                ema12_prev = df['ema12'].iloc[-2]
                ema26_prev = df['ema26'].iloc[-2]
                price_curr = df['close'].iloc[-1]
                
                # Check crossover
                ema_cross_up = (ema12_prev <= ema26_prev) and (ema12_curr > ema26_curr)
                ema_cross_down = (ema12_prev >= ema26_prev) and (ema12_curr < ema26_curr)
                
                # CDC ActionZone conditions
                buy_signal = ema_cross_up and (price_curr > ema12_curr)
                short_signal = ema_cross_down and (price_curr < ema12_curr)
                
                # Log
                if buy_signal:
                    logger.info(
                        f"🟢 1D CDC BUY | "
                        f"EMA12: {ema12_curr:.2f} > EMA26: {ema26_curr:.2f} | "
                        f"Price: {price_curr:.2f} > EMA12"
                    )
                elif short_signal:
                    logger.info(
                        f"🔴 1D CDC SELL | "
                        f"EMA12: {ema12_curr:.2f} < EMA26: {ema26_curr:.2f} | "
                        f"Price: {price_curr:.2f} < EMA12"
                    )
                
                return {
                    "buy": buy_signal,
                    "short": short_signal,
                    "sell": False,
                    "cover": False
                }
            
            # ========================================
            # 4H: IMPROVED SIGNALS
            # ========================================
            else:
                if len(df) < 30:
                    logger.warning(f"Insufficient data: {len(df)} candles")
                    return {"buy": False, "short": False, "sell": False, "cover": False}
                
                # Calculate RSI
                from ta.momentum import RSIIndicator
                rsi_indicator = RSIIndicator(df['close'], window=14)
                df['rsi'] = rsi_indicator.rsi()
                df['rsi_ma'] = df['rsi'].rolling(window=14).mean()
                
                rsi_current = df['rsi'].iloc[-1]
                rsi_ma_current = df['rsi_ma'].iloc[-1]
                rsi_prev = df['rsi'].iloc[-2]
                rsi_ma_prev = df['rsi_ma'].iloc[-2]
                
                # MACD values
                macd_data = analysis.get("macd", {})
                macd_cross = macd_data.get("cross_direction", "NONE")
                macd_line = macd_data.get("macd_line", 0)
                
                # Calculate MACD previous value
                from ta.trend import MACD
                macd_indicator = MACD(df['close'], window_slow=17, window_fast=8, window_sign=9)
                df['macd'] = macd_indicator.macd()
                macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd_line
                
                # Squeeze
                squeeze_data = analysis.get("squeeze", {})
                squeeze_off = squeeze_data.get("squeeze_off", False)
                
                # Check NaN
                if any(pd.isna([rsi_current, rsi_ma_current, rsi_prev, rsi_ma_prev])):
                    logger.warning("NaN values in RSI")
                    return {"buy": False, "short": False, "sell": False, "cover": False}
                
                # RSI Crossovers
                rsi_cross_up = (rsi_prev <= rsi_ma_prev) and (rsi_current > rsi_ma_current)
                rsi_cross_down = (rsi_prev >= rsi_ma_prev) and (rsi_current < rsi_ma_current)
                
                # ========================================
                # 🔥 ORIGINAL SIGNALS (Crossover Based)
                # ========================================
                original_buy = (
                    rsi_cross_up and 
                    macd_cross == "UP" and 
                    macd_line > -20 and        # ✅ แก้แล้ว: -20 แทน 0
                    squeeze_off
                )
                
                original_short = (
                    rsi_cross_down and 
                    macd_cross == "DOWN" and 
                    macd_line < 20 and         # ✅ แก้แล้ว: 20 แทน 0
                    squeeze_off
                )
                
                # ========================================
                # 🔥 STRONG MOMENTUM MODE
                # ========================================
                strong_momentum_buy = (
                    rsi_current > 65 and           # ✅ ลดจาก 70
                    rsi_current > rsi_prev and
                    macd_line > 80 and             # ✅ ลดจาก 100
                    macd_line > macd_prev and
                    squeeze_off
                )
                
                strong_momentum_short = (
                    rsi_current < 35 and           # ✅ เพิ่มจาก 30
                    rsi_current < rsi_prev and
                    macd_line < -80 and            # ✅ เพิ่มจาก -100
                    macd_line < macd_prev and
                    squeeze_off
                )
                
                # ========================================
                # 🆕 PULLBACK MODE
                # ========================================
                pullback_buy = (
                    macd_line > 50 and
                    rsi_current > 45 and rsi_current < 55 and
                    rsi_current > rsi_prev and
                    squeeze_off
                )
                
                pullback_short = (
                    macd_line < -50 and
                    rsi_current > 45 and rsi_current < 55 and
                    rsi_current < rsi_prev and
                    squeeze_off
                )
                
                # ========================================
                # รวมสัญญาณทั้ง 3 โหมด ✅
                # ========================================
                buy_signal = original_buy or strong_momentum_buy or pullback_buy
                short_signal = original_short or strong_momentum_short or pullback_short
                
                # ========================================
                # Multi-Timeframe Filter
                # ========================================
                if trend_1d:
                    raw_buy = buy_signal
                    raw_short = short_signal
                    
                    buy_signal = buy_signal and trend_1d.get("buy", False)
                    short_signal = short_signal and trend_1d.get("short", False)
                    
                    # Log ถ้าโดนบล็อก
                    if raw_buy and not buy_signal:
                        logger.info(f"🚫 LONG Blocked by 1D Trend (CDC is RED)")
                    if raw_short and not short_signal:
                        logger.info(f"🚫 SHORT Blocked by 1D Trend (CDC is GREEN)")
                
                # ========================================
                # 📊 LOGGING
                # ========================================
                if original_buy:
                    logger.info(
                        f"🟢 4H LONG (Crossover) | "
                        f"RSI: {rsi_prev:.2f}→{rsi_current:.2f} | "
                        f"MACD: {macd_cross} ({macd_line:.6f}) | "
                        f"Squeeze: OFF"
                    )
                elif strong_momentum_buy:
                    logger.info(
                        f"🔥 4H LONG (Strong Momentum) | "
                        f"RSI: {rsi_current:.2f} (rising, >65) | "
                        f"MACD: {macd_line:.6f} (rising, >80) | "
                        f"Squeeze: OFF"
                    )
                elif pullback_buy:
                    logger.info(
                        f"📈 4H LONG (Pullback) | "
                        f"RSI: {rsi_current:.2f} (mid, rising) | "
                        f"MACD: {macd_line:.6f} (>50) | "
                        f"Squeeze: OFF"
                    )
                elif original_short:
                    logger.info(
                        f"🔴 4H SHORT (Crossover) | "
                        f"RSI: {rsi_prev:.2f}→{rsi_current:.2f} | "
                        f"MACD: {macd_cross} ({macd_line:.6f}) | "
                        f"Squeeze: OFF"
                    )
                elif strong_momentum_short:
                    logger.info(
                        f"🔥 4H SHORT (Strong Momentum) | "
                        f"RSI: {rsi_current:.2f} (falling, <35) | "
                        f"MACD: {macd_line:.6f} (falling, <-80) | "
                        f"Squeeze: OFF"
                    )
                elif pullback_short:
                    logger.info(
                        f"📉 4H SHORT (Pullback) | "
                        f"RSI: {rsi_current:.2f} (mid, falling) | "
                        f"MACD: {macd_line:.6f} (<-50) | "
                        f"Squeeze: OFF"
                    )
                else:
                    logger.debug(
                        f"4H No signal | RSI: {rsi_current:.2f}, "
                        f"MACD: {macd_cross} ({macd_line:.6f}), Squeeze: {squeeze_off}"
                    )
                
                return {
                    "buy": buy_signal,
                    "short": short_signal,
                    "sell": False,
                    "cover": False
                }
        
        except Exception as e:
            logger.error(f"Error detecting signals: {e}", exc_info=True)
            return {"buy": False, "short": False, "sell": False, "cover": False}

    def _check_market_trend_enhanced(self, df) -> str:
        """Conservative trend detection using MA20 and MA50"""
        try:
            close = df['close']
            
            # Calculate MAs
            ma_20 = close.rolling(20).mean()
            ma_50 = close.rolling(50).mean() if len(close) >= 50 else None
            
            current_price = close.iloc[-1]
            ma_20_current = ma_20.iloc[-1]
            
            # Case 1: Have MA50 - strict check
            if ma_50 is not None:
                ma_50_current = ma_50.iloc[-1]
                
                # Uptrend: Price > MA20 AND MA20 > MA50
                if current_price > ma_20_current and ma_20_current > ma_50_current:
                    return "UP"
                
                # Downtrend: Price < MA20 AND MA20 < MA50
                elif current_price < ma_20_current and ma_20_current < ma_50_current:
                    return "DOWN"
                
                else:
                    return "NEUTRAL"
            
            # Case 2: No MA50 - use only MA20
            else:
                if current_price > ma_20_current:
                    return "UP"
                elif current_price < ma_20_current:
                    return "DOWN"
                else:
                    return "NEUTRAL"
                    
        except Exception as e:
            logger.error(f"Error checking market trend: {e}")
            return "NEUTRAL"

    def _get_recommendation_improved(self, signals: Dict[str, bool]) -> str:
        """Generate recommendation based on signals"""
        if signals.get("buy"):
            return "LONG"
        elif signals.get("short"):
            return "SHORT"
        else:
            return ""

    def _calculate_signal_strength_improved(self, signals: Dict[str, bool]) -> int:
        """Calculate signal strength (0-100)"""
        if signals.get("buy") or signals.get("short"):
            return 100  # Signals that pass conservative conditions = 100%
        else:
            return 0

    def _calculate_risk_levels(self, current_price: float, timeframe: str, signals: Dict, symbol: str) -> Dict:
        """Calculate Stop Loss and Take Profit levels"""
        try:
            risk_config = self.risk_management.get(
                timeframe, self.risk_management.get("4h", {})
            )

            tp_percentages = risk_config.get("tp_levels", [3.0, 5.0, 7.0])
            
            # --- 🆕 เริ่มส่วนที่แก้ไขใหม่: คำนวณ SL ตามความซิ่ง (Volatility) ---
            # ดึงข้อมูล 10 แท่งล่าสุดมาดูระยะเหวี่ยง (ใช้ข้อมูลจาก signals หรือ symbol ที่เกี่ยวข้อง)
            symbol = symbol
            df_recent = self.data_manager.get_klines(symbol, timeframe, limit=10)
            
            if df_recent is not None and not df_recent.empty:
                # คำนวณความเหวี่ยงเฉลี่ยเป็น % (High-Low)
                volatility = ((df_recent['high'] - df_recent['low']) / df_recent['close']).mean() * 100
                # ใช้ 1.5 เท่าของความเหวี่ยง แต่ไม่ต่ำกว่า 2% และไม่เกิน 5%
                sl_percentage = min(max(volatility * 1.5, 2.0), 5.0)
                logger.info(f"🛡️ Dynamic SL set at {sl_percentage:.2f}% (Volatility: {volatility:.2f}%)")
            else:
                # ถ้าดึงข้อมูลไม่ได้ ให้ใช้ค่าจาก Config ตามเดิม
                sl_percentage = risk_config.get("sl_level", 3.0)
            # -----------------------------------------------------------

            risk_levels = {"timeframe": timeframe, "entry_price": current_price}

            # Determine signal direction
            is_long_signal = signals.get("buy", False)
            is_short_signal = signals.get("short", False)

            # Calculate levels based on signal direction
            if is_long_signal:
                risk_levels.update({
                    "direction": "LONG",
                    "stop_loss": current_price * (1 - sl_percentage / 100),
                    "take_profit_1": current_price * (1 + tp_percentages[0] / 100),
                    "take_profit_2": current_price * (1 + tp_percentages[1] / 100),
                    "take_profit_3": current_price * (1 + tp_percentages[2] / 100),
                    "risk_reward_ratio": tp_percentages[0] / sl_percentage,
                })

            elif is_short_signal:
                risk_levels.update({
                    "direction": "SHORT",
                    "stop_loss": current_price * (1 + sl_percentage / 100),
                    "take_profit_1": current_price * (1 - tp_percentages[0] / 100),
                    "take_profit_2": current_price * (1 - tp_percentages[1] / 100),
                    "take_profit_3": current_price * (1 - tp_percentages[2] / 100),
                    "risk_reward_ratio": tp_percentages[0] / sl_percentage,
                })

            return risk_levels

        except Exception as e:
            logger.error(f"Error calculating risk levels: {e}")
            return {"error": "Failed to calculate risk levels"}

    def scan_multiple_symbols(self, symbols: List[str], timeframes: List[str] = None) -> List[Dict]:
        """Scan multiple symbols for signals across different timeframes"""
        if timeframes is None:
            timeframes = ["4h", "1d"]

        results = []

        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"🔍 Scanning {symbol} on {timeframe}")
                result = self.analyze_symbol(symbol, timeframe)
                if result:
                    results.append(result)
                time.sleep(0.2)

        return results

    def get_active_signals(self, symbols: List[str], timeframes: List[str] = None) -> List[Dict]:
        """Get only signals with active recommendations"""
        if timeframes is None:
            timeframes = ["4h", "1d"]

        all_results = self.scan_multiple_symbols(symbols, timeframes)

        # Filter only results with actual recommendations
        active_signals = []
        for result in all_results:
            if "signals" in result and result.get("recommendation"):
                signals = result["signals"]
                if signals.get("buy") or signals.get("short"):
                    active_signals.append(result)

        logger.info(f"Found {len(active_signals)} active signals out of {len(all_results)} scans")
        return active_signals

    def scan_all_symbols(self, symbols: List[str] = None, timeframes: List[str] = None) -> List[Dict]:
        """Scan all symbols and return all results"""
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        if timeframes is None:
            timeframes = ["4h", "1d"]
            
        return self.scan_multiple_symbols(symbols, timeframes)

    def validate_signal_quality(self, analysis: Dict) -> Dict:
        """Validate signal quality and reliability"""
        try:
            quality_score = 0
            quality_factors = []

            indicators = analysis.get("indicators", {})
            signals = analysis.get("signals", {})

            # Check squeeze momentum quality
            squeeze = indicators.get("squeeze", {})
            if squeeze.get("squeeze_off"):
                quality_score += 30
                quality_factors.append("Squeeze breakout confirmed")

                # Check momentum strength
                details = squeeze.get("details", {})
                momentum_value = abs(details.get("momentum_value", 0))
                if momentum_value > 0.001:
                    quality_score += 10
                    quality_factors.append("Strong momentum")

            # Check MACD quality
            macd = indicators.get("macd", {})
            if macd.get("cross_direction") != "NONE":
                quality_score += 25
                quality_factors.append("MACD cross confirmed")

                # Check if MACD is above/below zero line
                macd_details = macd.get("details", {})
                if macd_details.get("macd_above_zero") and signals.get("buy"):
                    quality_score += 10
                    quality_factors.append("MACD above zero line")
                elif not macd_details.get("macd_above_zero") and signals.get("short"):
                    quality_score += 10
                    quality_factors.append("MACD below zero line")

            # Check RSI quality
            rsi = indicators.get("rsi", {})
            rsi_value = rsi.get("value", 50)
            
            if rsi_value < 40 or rsi_value > 60:
                quality_score += 20
                level = "oversold" if rsi_value < 40 else "overbought"
                quality_factors.append(f"RSI {level} level")

                # Check RSI trend alignment
                rsi_details = rsi.get("details", {})
                rsi_trend = rsi_details.get("rsi_trend", "NEUTRAL")
                if (rsi_value < 40 and rsi_trend == "RISING") or (rsi_value > 60 and rsi_trend == "FALLING"):
                    quality_score += 5
                    quality_factors.append("RSI trend alignment")

            # Signal grade bonus
            if signals.get("buy") or signals.get("short"):
                quality_score += 15
                quality_factors.append("Strong signal grade")

            # Risk-reward assessment
            risk_levels = analysis.get("risk_levels", {})
            risk_reward = risk_levels.get("risk_reward_ratio", 0)
            if risk_reward >= 1.0:
                quality_score += 5
                quality_factors.append("Favorable risk-reward ratio")

            # Cap quality score at 100
            quality_score = min(quality_score, 100)

            return {
                "quality_score": quality_score,
                "quality_factors": quality_factors,
                "risk_reward_ratio": risk_reward,
                "signal_reliability": (
                    "HIGH" if quality_score >= 80
                    else "MEDIUM" if quality_score >= 60 
                    else "LOW"
                ),
            }

        except Exception as e:
            logger.error(f"Error validating signal quality: {e}")
            return {
                "quality_score": 0,
                "quality_factors": [],
                "risk_reward_ratio": 0,
                "signal_reliability": "UNKNOWN",
            }

    # Position Management Integration Methods
    def get_position_summary(self) -> Dict:
        """Get positions summary from PositionManager"""
        try:
            return self.position_manager.get_positions_summary()
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {"error": str(e)}
    
    def get_position_status(self, symbol: str, timeframe: str) -> Dict:
        """Get specific position status from PositionManager"""
        try:
            position = self.position_manager.get_position_status(symbol, timeframe)
            return {
                "position_found": position is not None,
                "position": position,
                "symbol": symbol,
                "timeframe": timeframe
            }
        except Exception as e:
            logger.error(f"Error getting position status: {e}")
            return {"error": str(e), "position_found": False}
    
    def force_close_position(self, symbol: str, timeframe: str, reason: str = "MANUAL") -> Dict:
        """Force close a position via PositionManager"""
        try:
            # Create position_id in the format expected by PositionManager
            position_id = f"{symbol}_{timeframe}_LONG"  # Try LONG first
            success = self.position_manager.close_position(position_id, reason)
            
            if not success:
                # Try SHORT if LONG doesn't exist
                position_id = f"{symbol}_{timeframe}_SHORT"
                success = self.position_manager.close_position(position_id, reason)
            
            if success:
                # Remove from tracking
                if symbol in self.active_positions:
                    self.active_positions.remove(symbol)
                
                return {
                    "success": True, 
                    "message": f"Closed position for {symbol} {timeframe}",
                    "reason": reason
                }
            else:
                return {
                    "success": False, 
                    "message": f"No active position found for {symbol} {timeframe}"
                }
                
        except Exception as e:
            logger.error(f"Error force closing position: {e}")
            return {"success": False, "error": str(e)}
    
    def update_all_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Update all positions with current prices via PositionManager"""
        try:
            updates = self.position_manager.update_positions()
            
            # Format results for compatibility
            results = []
            for position_id, update_info in updates.items():
                # Extract symbol from position_id (format: SYMBOL_TIMEFRAME_DIRECTION)
                parts = position_id.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    timeframe = parts[1]
                    
                    result = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "position_id": position_id,
                        "update_info": update_info
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error updating all positions: {e}")
            return []

    def get_data_storage_stats(self) -> Dict:
        """Get data storage statistics from DataManager"""
        try:
            return self.data_manager.get_cache_stats()
        except Exception as e:
            logger.error(f"Error getting data storage stats: {e}")
            return {"error": str(e)}
    
    def force_data_update(self, symbol: str, timeframe: str):
        """Force data update for symbol/timeframe via DataManager"""
        try:
            # Clear cache to force refresh
            self.data_manager.clear_cache()
            logger.info(f"Forced data update for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error forcing data update for {symbol} {timeframe}: {e}")
    
    def clear_position_history(self):
        """Clear position history for testing"""
        self.active_positions.clear()
        logger.info("Cleared position history and tracking")
    
    def shutdown(self):
        """Shutdown SignalDetector and cleanup resources"""
        try:
            logger.info("Shutting down SignalDetector CONSERVATIVE mode...")
            
            # Clear data manager cache
            if hasattr(self.data_manager, 'clear_cache'):
                self.data_manager.clear_cache()
            
            # Cleanup old positions  
            if hasattr(self.position_manager, 'cleanup_old_positions'):
                self.position_manager.cleanup_old_positions()
            
            # Clear tracking
            self.active_positions.clear()
                
            logger.info("SignalDetector shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")