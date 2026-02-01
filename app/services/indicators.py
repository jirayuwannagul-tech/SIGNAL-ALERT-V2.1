"""Technical indicators for Squeeze Bot trading system."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical analysis indicators calculator with layer organization."""

    # ================================================================
    # 🎯 LAYER 1: Squeeze Momentum Indicator (การคำนวณ Squeeze และ Momentum)
    # ================================================================

    @staticmethod
    def squeeze_momentum(
        df: pd.DataFrame, length: int = 20, mult_bb: float = 2.0, mult_kc: float = 1.5
    ) -> Tuple[bool, str, Dict]:
        """
        Calculate Squeeze Momentum Indicator.

        This indicator identifies periods of low volatility (squeeze) followed by
        directional moves when the squeeze is released.

        Args:
            df: DataFrame with OHLCV data
            length: Period for calculation (default 20)
            mult_bb: Bollinger Bands multiplier (default 2.0)
            mult_kc: Keltner Channel multiplier (default 1.5)

        Returns:
            Tuple[bool, str, Dict]: (squeeze_off, momentum_direction, details)
        """
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # Bollinger Bands calculation
            basis = close.rolling(length).mean()
            dev = mult_bb * close.rolling(length).std()
            upper_bb = basis + dev
            lower_bb = basis - dev

            # Keltner Channel calculation (using True Range)
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            rangema = true_range.rolling(length).mean()

            # Moving average for Keltner Channel
            ma = close.rolling(length).mean()
            upper_kc = ma + rangema * mult_kc
            lower_kc = ma - rangema * mult_kc

            # Squeeze detection: BB inside KC = squeeze ON, BB outside KC = squeeze OFF
            squeeze_off = (lower_bb.iloc[-1] < lower_kc.iloc[-1]) and (
                upper_bb.iloc[-1] > upper_kc.iloc[-1]
            )

            # Momentum calculation using linear regression
            highest = high.rolling(length).max()
            lowest = low.rolling(length).min()
            mid_point = (highest + lowest) / 2

            momentum_direction = "NEUTRAL"
            momentum_value = 0

            if len(close) >= length:
                # Linear regression of (close - midpoint) for momentum
                x = np.arange(length)
                y = (close - mid_point).iloc[-length:].values

                if len(y) == length and not np.isnan(y).any():
                    # Current momentum slope
                    slope = np.polyfit(x, y, 1)[0]
                    momentum_value = slope

                    # Previous momentum slope for comparison
                    if len(close) > length:
                        prev_y = (close - mid_point).iloc[-length - 1 : -1].values
                        if len(prev_y) == length and not np.isnan(prev_y).any():
                            prev_slope = np.polyfit(x, prev_y, 1)[0]

                            # Determine momentum direction
                            if slope > prev_slope:
                                momentum_direction = "UP"
                            elif slope < prev_slope:
                                momentum_direction = "DOWN"
                            else:
                                momentum_direction = "NEUTRAL"

            # Additional details for analysis
            details = {
                "bb_upper": float(upper_bb.iloc[-1]),
                "bb_lower": float(lower_bb.iloc[-1]),
                "kc_upper": float(upper_kc.iloc[-1]),
                "kc_lower": float(lower_kc.iloc[-1]),
                "momentum_value": float(momentum_value),
                "squeeze_intensity": float(upper_bb.iloc[-1] - lower_bb.iloc[-1])
                / float(upper_kc.iloc[-1] - lower_kc.iloc[-1]),
            }

            # Ensure boolean is Python native type
            squeeze_off = bool(squeeze_off)

            logger.debug(
                f"Squeeze analysis: OFF={squeeze_off}, Direction={momentum_direction}"
            )
            return squeeze_off, momentum_direction, details

        except Exception as e:
            logger.error(f"Error calculating squeeze momentum: {e}")
            return False, "NEUTRAL", {}

    # ================================================================
    # 📈 LAYER 2: MACD Uncle Cholok (MACD ลุงโฉลก 8,17,9)
    # ================================================================

    @staticmethod
    def macd_uncle_cholok(
        df: pd.DataFrame, fast: int = 8, slow: int = 17, signal: int = 9
    ) -> Tuple[float, float, str, Dict]:
        """
        MACD ลุงโฉลก (Uncle Cholok) calculation with custom periods (8,17,9).
        ✅ UPDATED: More flexible cross detection
        """
        try:
            close = df["close"]

            # Calculate MACD components
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            # Get current values
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]

            # ✅ UPDATED: Cross detection with momentum
            cross_direction = "NONE"
            if len(macd_line) > 1:
                prev_macd = macd_line.iloc[-2]
                prev_signal = signal_line.iloc[-2]

                # Bullish signal: MACD crosses above Signal OR trending up
                if current_macd > current_signal:
                    if prev_macd <= prev_signal:
                        # Traditional cross
                        cross_direction = "UP"
                        logger.debug("MACD: Traditional bullish cross detected")
                    elif current_macd > prev_macd:
                        # MACD above signal and increasing
                        cross_direction = "UP"
                        logger.debug("MACD: Bullish momentum detected (above signal + rising)")
            
                # Bearish signal: MACD crosses below Signal OR trending down
                elif current_macd < current_signal:
                    if prev_macd >= prev_signal:
                        # Traditional cross
                        cross_direction = "DOWN"
                        logger.debug("MACD: Traditional bearish cross detected")
                    elif current_macd < prev_macd:
                        # MACD below signal and decreasing
                        cross_direction = "DOWN"
                        logger.debug("MACD: Bearish momentum detected (below signal + falling)")

            # Additional details
            details = {
                "ema_fast": float(ema_fast.iloc[-1]),
                "ema_slow": float(ema_slow.iloc[-1]),
                "histogram": float(current_histogram),
                "macd_above_zero": bool(current_macd > 0),
                "signal_above_zero": bool(current_signal > 0),
                "divergence_strength": float(abs(current_macd - current_signal)),
            }

            logger.debug(
                f"MACD Uncle Cholok: {current_macd:.6f}, Signal: {current_signal:.6f}, Cross: {cross_direction}"
            )
            return float(current_macd), float(current_signal), cross_direction, details

        except Exception as e:
            logger.error(f"Error calculating MACD Uncle Cholok: {e}")
            return 0.0, 0.0, "NONE", {}

    # ================================================================
    # 📊 LAYER 3: RSI Extreme (RSI โต่ง) - ปรับ default threshold 40/60
    # ================================================================

    @staticmethod
    def rsi_extreme(
        df: pd.DataFrame,
        period: int = 14,
        low_threshold: float = 35,   # ✅ ปรับจาก 40 เป็น 35 (ต้องขายหนักจริงๆ ถึงจะน่าสะสม)
        high_threshold: float = 65,  # ✅ ปรับจาก 60 เป็น 65 (ต้องซื้อหนักจริงๆ ถึงจะน่าระวัง)
    ) -> Tuple[float, str, Dict]:
        """
        RSI โต่ง (Extreme RSI) calculation for identifying overbought/oversold conditions.
        ปรับ default threshold เป็น 40/60 เพื่อให้ได้สัญญาณมากขึ้น

        Args:
            df: DataFrame with OHLCV data
            period: RSI calculation period (default 14)
            low_threshold: Oversold threshold (default 40 - ปรับจาก 30)
            high_threshold: Overbought threshold (default 60 - ปรับจาก 70)

        Returns:
            Tuple[float, str, Dict]: (rsi_value, extreme_level, details)
        """
        try:
            close = df["close"]

            # Calculate price changes
            delta = close.diff()

            # Separate gains and losses
            gain = (
                (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=period, min_periods=1)
                .mean()
            )

            # Calculate Relative Strength and RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]

            # Determine extreme level - ใช้ threshold ใหม่
            if current_rsi < low_threshold:
                extreme_level = "LOW"
            elif current_rsi > high_threshold:
                extreme_level = "HIGH"
            else:
                extreme_level = "NORMAL"

            # Calculate RSI tren
            rsi_trend = "NEUTRAL"
            if len(rsi) >= 3:
                recent_rsi = rsi.iloc[-3:]
                if recent_rsi.iloc[-1] > recent_rsi.iloc[-2] > recent_rsi.iloc[-3]:
                    rsi_trend = "RISING"
                elif recent_rsi.iloc[-1] < recent_rsi.iloc[-2] < recent_rsi.iloc[-3]:
                    rsi_trend = "FALLING"

            # Additional details
            details = {
                "rsi_14": float(current_rsi),
                "rsi_trend": rsi_trend,
                "distance_to_oversold": float(current_rsi - low_threshold),
                "distance_to_overbought": float(high_threshold - current_rsi),
                "is_diverging": False,  # TODO: Implement divergence detection
                "threshold_low": float(low_threshold),   # เพิ่มเพื่อ debug
                "threshold_high": float(high_threshold), # เพิ่มเพื่อ debug
            }

            logger.debug(
                f"RSI Extreme: {current_rsi:.2f}, Level: {extreme_level}, Trend: {rsi_trend}, Thresholds: {low_threshold}/{high_threshold}"
            )
            return float(current_rsi), extreme_level, details

        except Exception as e:
            logger.error(f"Error calculating RSI Extreme: {e}")
            return 50.0, "NORMAL", {}

    # ================================================================
    # 🔍 LAYER 4: Comprehensive Analysis (การวิเคราะห์รวม) - แก้ไข default values
    # ================================================================

    @staticmethod
    def analyze_all_indicators(df: pd.DataFrame, config: Dict) -> Dict:
        """
        Calculate all indicators and return comprehensive analysis.
        ปรับปรุงให้ใช้ RSI threshold ใหม่ (40/60) เป็น default

        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with indicator settings

        Returns:
            Dict with all indicator results
        """
        try:
            # Get indicator settings from config
            squeeze_config = config.get("squeeze", {})
            macd_config = config.get("macd", {})
            rsi_config = config.get("rsi", {})

            # Calculate Squeeze Momentum
            squeeze_off, momentum_direction, squeeze_details = (
                TechnicalIndicators.squeeze_momentum(
                    df,
                    length=squeeze_config.get("length", 20),
                    mult_bb=squeeze_config.get("bb_mult", 2.0),
                    mult_kc=squeeze_config.get("kc_mult", 1.5),
                )
            )

            # Calculate MACD Uncle Cholok
            macd_line, signal_line, macd_cross, macd_details = (
                TechnicalIndicators.macd_uncle_cholok(
                    df,
                    fast=macd_config.get("fast", 8),
                    slow=macd_config.get("slow", 17),
                    signal=macd_config.get("signal", 9),
                )
            )

            # Calculate RSI Extreme - ⭐ แก้ไข default values เป็น 40/60
            rsi_value, rsi_extreme, rsi_details = TechnicalIndicators.rsi_extreme(
                df,
                period=rsi_config.get("period", 14),
                low_threshold=rsi_config.get("oversold", 40),   # เปลี่ยนจาก 30 เป็น 40
                high_threshold=rsi_config.get("overbought", 60), # เปลี่ยนจาก 70 เป็น 60
            )

            # Compile comprehensive analysis
            analysis = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "current_price": float(df["close"].iloc[-1]),
                "squeeze": {
                    "squeeze_off": squeeze_off,
                    "momentum_direction": momentum_direction,
                    "details": squeeze_details,
                },
                "macd": {
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "cross_direction": macd_cross,
                    "details": macd_details,
                },
                "rsi": {
                    "value": rsi_value,
                    "extreme_level": rsi_extreme,
                    "details": rsi_details,
                },
            }

            logger.info("Comprehensive indicator analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {}

    # ================================================================
    # 🛠️ LAYER 5: Helper Functions (ฟังก์ชันช่วยเหลือ)
    # ================================================================

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """
        ตรวจสอบความถูกต้องของ DataFrame
        Validate DataFrame for required columns and data quality.
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
            
        if df.empty:
            logger.error("DataFrame is empty")
            return False
            
        if df.isnull().any().any():
            logger.warning("DataFrame contains null values")
            
        return True

    @staticmethod
    def get_indicator_summary(analysis: Dict) -> Dict:
        """
        สรุปสถานะ indicators สำหรับการแสดงผล
        Get indicator summary for display purposes.
        """
        try:
            summary = {
                "timestamp": analysis.get("timestamp"),
                "price": analysis.get("current_price"),
                "squeeze_status": "OFF" if analysis.get("squeeze", {}).get("squeeze_off") else "ON",
                "momentum": analysis.get("squeeze", {}).get("momentum_direction", "NEUTRAL"),
                "macd_cross": analysis.get("macd", {}).get("cross_direction", "NONE"),
                "rsi_value": round(analysis.get("rsi", {}).get("value", 50), 2),
                "rsi_level": analysis.get("rsi", {}).get("extreme_level", "NORMAL"),
                "signals_present": {
                    "squeeze_breakout": analysis.get("squeeze", {}).get("squeeze_off", False),
                    "macd_signal": analysis.get("macd", {}).get("cross_direction") != "NONE",
                    "rsi_extreme": analysis.get("rsi", {}).get("extreme_level") != "NORMAL"
                }
            }
            return summary
        except Exception as e:
            logger.error(f"Error creating indicator summary: {e}")
            return {}

    @staticmethod
    def calculate_signal_confluence(analysis: Dict) -> Dict:
        """
        คำนวณการยืนยันสัญญาณจาก indicators หลายตัว
        Calculate signal confluence between multiple indicators.
        """
        try:
            squeeze = analysis.get("squeeze", {})
            macd = analysis.get("macd", {})
            rsi = analysis.get("rsi", {})
            
            # Count bullish signals
            bullish_signals = 0
            bearish_signals = 0
            
            # Squeeze momentum
            if squeeze.get("momentum_direction") == "UP":
                bullish_signals += 1
            elif squeeze.get("momentum_direction") == "DOWN":
                bearish_signals += 1
                
            # MACD cross
            if macd.get("cross_direction") == "UP":
                bullish_signals += 1
            elif macd.get("cross_direction") == "DOWN":
                bearish_signals += 1
                
            # RSI extreme
            if rsi.get("extreme_level") == "LOW":
                bullish_signals += 1
            elif rsi.get("extreme_level") == "HIGH":
                bearish_signals += 1
                
            # Squeeze breakout (neutral but important)
            squeeze_breakout = squeeze.get("squeeze_off", False)
            
            confluence = {
                "bullish_count": bullish_signals,
                "bearish_count": bearish_signals,
                "total_signals": bullish_signals + bearish_signals,
                "squeeze_breakout": squeeze_breakout,
                "confluence_strength": "STRONG" if (bullish_signals >= 3 or bearish_signals >= 3) else
                                     "MEDIUM" if (bullish_signals >= 2 or bearish_signals >= 2) else
                                     "WEAK",
                "direction": "BULLISH" if bullish_signals > bearish_signals else
                           "BEARISH" if bearish_signals > bullish_signals else
                           "NEUTRAL"
            }
            
            return confluence
            
        except Exception as e:
            logger.error(f"Error calculating signal confluence: {e}")
            return {}

    # ================================================================
    # 📋 LAYER 6: Debug และ Monitoring Functions
    # ================================================================

    @staticmethod
    def get_indicator_health(analysis: Dict) -> Dict:
        """
        ตรวจสอบสุขภาพของการคำนวณ indicators
        Check the health of indicator calculations.
        """
        try:
            health = {
                "overall_status": "HEALTHY",
                "issues": [],
                "warnings": []
            }
            
            # Check squeeze calculation
            squeeze = analysis.get("squeeze", {})
            if not squeeze.get("details"):
                health["issues"].append("Squeeze calculation incomplete")
                health["overall_status"] = "ERROR"
                
            # Check MACD calculation  
            macd = analysis.get("macd", {})
            if macd.get("macd_line") == 0 and macd.get("signal_line") == 0:
                health["warnings"].append("MACD values are zero - check data quality")
                
            # Check RSI calculation
            rsi = analysis.get("rsi", {})
            rsi_value = rsi.get("value", 50)
            if rsi_value < 0 or rsi_value > 100:
                health["issues"].append(f"RSI value out of range: {rsi_value}")
                health["overall_status"] = "ERROR"
                
            # Check data freshness
            timestamp = analysis.get("timestamp")
            if timestamp:
                from datetime import datetime, timedelta
                analysis_time = pd.Timestamp(timestamp)
                if pd.Timestamp.now() - analysis_time > timedelta(minutes=10):
                    health["warnings"].append("Analysis data is more than 10 minutes old")
                    
            return health
            
        except Exception as e:
            logger.error(f"Error checking indicator health: {e}")
            return {"overall_status": "ERROR", "issues": [str(e)], "warnings": []}