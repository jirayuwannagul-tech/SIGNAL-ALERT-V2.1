"""LINE Bot notification service for trading signals - REFACTORED for v2.0"""
import logging
from datetime import datetime
from typing import Dict, Optional

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import TextSendMessage

logger = logging.getLogger(__name__)


class LineNotifier:
    """
    REFACTORED LINE Bot service for v2.0
    
    Main responsibilities:
    - Send trading signal notifications
    - Send position update alerts
    - Send daily summaries and error alerts
    - Handle LINE webhook verification
    
    Uses ConfigManager for:
    - LINE channel access token
    - LINE channel secret
    - LINE user ID
    """

    def __init__(self, config: Dict):
        """
        Initialize LINE notifier with ConfigManager config
        
        Args:
            config: Configuration from ConfigManager.get_line_config()
                   Expected keys: 'access_token', 'secret', optionally 'user_id'
        """
        # Configuration from ConfigManager
        self.channel_access_token = config.get("access_token")
        self.channel_secret = config.get("secret")
        self.user_id = config.get("user_id")  # Optional, can be set later

        if not self.channel_access_token or not self.channel_secret:
            logger.warning(
                "LINE credentials not fully configured - notifications disabled"
            )
            self.line_bot_api = None
            self.handler = None
            return

        try:
            self.line_bot_api = LineBotApi(self.channel_access_token)
            self.handler = WebhookHandler(self.channel_secret)
            logger.info("LineNotifier v2.0 initialized successfully")
        except Exception as e:
            logger.error(f"LINE Bot initialization failed: {e}")
            self.line_bot_api = None
            self.handler = None

    def send_signal_alert(self, analysis: Dict) -> bool:
        """ส่งสัญญาณเทรดไป LINE และส่งต่อให้จ่าเฉย"""
        try:
            # 🚨 ส่วนที่เพิ่ม: ส่งต่อข้อมูล JSON ให้จ่าเฉย
            try:
                import requests
                jachey_url = "https://web-production-82bfc.up.railway.app/callback"
                requests.post(jachey_url, json=analysis, timeout=5)
                logger.info(f"👮‍♂️ ส่งต่อข้อมูลให้จ่าเฉยสำเร็จ: {analysis.get('symbol')}")
            except Exception as e:
                logger.error(f"❌ ส่งต่อให้จ่าเฉยพลาด: {e}")

            # --- โค้ดเดิมของพี่ ---
            if not self.line_bot_api or not self.user_id:
                logger.warning("LINE not properly configured, cannot send signal alert")
                return False

            signals = analysis.get("signals", {})
            recommendation = analysis.get("recommendation", "")

            if signals.get("buy") or signals.get("short"):
                message = self._create_entry_signal_message(analysis)
            else:
                return False

            self.line_bot_api.push_message(self.user_id, TextSendMessage(text=message))
            return True
    def send_position_update(self, update_data: Dict) -> bool:
        """
        Send position update notification to LINE
        
        Args:
            update_data: Position update data with events and position info
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            if not self.line_bot_api or not self.user_id:
                logger.warning("LINE not properly configured, cannot send position update")
                return False

            # Check if there are significant events to report
            events = update_data.get("events", [])
            if not events:
                return False  # No significant update

            message = self._create_position_update_message(update_data)
            self.line_bot_api.push_message(self.user_id, TextSendMessage(text=message))
            logger.info(f"Position update sent: {', '.join(events)}")
            return True

        except Exception as e:
            logger.error(f"Error sending position update: {e}")
            return False

    def send_daily_summary(self, summary: Dict) -> bool:
        """
        Send daily trading summary
        
        Args:
            summary: Daily summary data
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            if not self.line_bot_api or not self.user_id:
                logger.warning("LINE not properly configured, cannot send daily summary")
                return False

            message = self._create_daily_summary_message(summary)
            self.line_bot_api.push_message(self.user_id, TextSendMessage(text=message))
            logger.info("Daily summary sent")
            return True

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False

    def _create_entry_signal_message(self, analysis: Dict) -> str:
        """Create formatted message for entry signals"""
        symbol = analysis.get("symbol", "UNKNOWN")
        timeframe = analysis.get("timeframe", "4h")
        current_price = analysis.get("current_price", 0)
        signals = analysis.get("signals", {})
        risk_levels = analysis.get("risk_levels", {})
        signal_strength = analysis.get("signal_strength", 0)

        # Determine signal type and colors
        if signals.get("buy"):
            signal_type = "🟢 LONG"
            direction = "LONG"
            signal_emoji = "📈"
        elif signals.get("short"):
            signal_type = "🔴 SHORT"
            direction = "SHORT"
            signal_emoji = "📉"
        else:
            signal_type = "⚫ UNKNOWN"
            direction = "UNKNOWN"
            signal_emoji = "❓"

        # Get indicator values
        indicators = analysis.get("indicators", {})
        squeeze = indicators.get("squeeze", {})
        macd = indicators.get("macd", {})
        rsi = indicators.get("rsi", {})

        # Create formatted message
        message = f"""🤖 SQUEEZE BOT SIGNAL v2.0

{signal_emoji} {signal_type}
Symbol: {symbol}
Timeframe: {timeframe.upper()}
Price: ${current_price:.4f}
Strength: {signal_strength}%

📊 INDICATORS:
- Squeeze: {"OFF ✅" if squeeze.get('squeeze_off') else "ON ❌"}
- Momentum: {squeeze.get('momentum_direction', 'NEUTRAL')}
- MACD: {macd.get('cross_direction', 'NONE')} Cross
- RSI: {rsi.get('value', 50):.1f}

🎯 TRADE SETUP:
- Entry: ${risk_levels.get('entry_price', current_price):.4f}
- SL: ${risk_levels.get('stop_loss', 0):.4f}
- TP1: ${risk_levels.get('take_profit_1', 0):.4f}
- TP2: ${risk_levels.get('take_profit_2', 0):.4f}
- TP3: ${risk_levels.get('take_profit_3', 0):.4f}

⚖️ R:R = {risk_levels.get('risk_reward_ratio', 0):.2f}
🕐 {datetime.now().strftime('%H:%M:%S')}

#{symbol} #{timeframe.upper()} #{direction} #v2"""

        return message

    def _create_position_update_message(self, update_data: Dict) -> str:
        """Create formatted message for position updates"""
        # Extract position and update information
        position = update_data.get("position", {})
        updates = update_data.get("updates", {})
        events = update_data.get("events", [])

        symbol = position.get("symbol", "UNKNOWN")
        direction = position.get("direction", "UNKNOWN")
        current_price = position.get("current_price", 0)
        pnl_pct = position.get("pnl_pct", 0)

        # Direction emoji
        direction_emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚫"

        message = f"📊 POSITION UPDATE v2.0\n\n"
        message += f"{direction_emoji} {direction} Position\n"
        message += f"Symbol: {symbol}\n"
        message += f"Current Price: ${current_price:.4f}\n"

        # P&L with color
        pnl_emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚫"
        message += f"P&L: {pnl_emoji} {pnl_pct:+.2f}%\n\n"

        # Report events
        for event in events:
            if "SL hit" in event:
                message += f"🛑 {event}\n"
            elif "TP" in event and "hit" in event:
                message += f"🎯 {event}\n"
            elif "Position closed" in event:
                message += f"🏁 {event}\n"

        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        message += f"\n#{symbol} #{direction} #Update #v2"

        return message

    def _create_daily_summary_message(self, summary: Dict) -> str:
        """Create formatted daily summary message"""
        total_signals = summary.get("total_signals", 0)
        active_positions = summary.get("active_positions", 0)
        closed_positions = summary.get("closed_positions", 0)
        total_pnl_pct = summary.get("total_pnl_pct", 0)
        win_rate_pct = summary.get("win_rate_pct", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        version = summary.get("version", "2.0")

        # P&L with color
        pnl_emoji = "🟢" if total_pnl_pct > 0 else "🔴" if total_pnl_pct < 0 else "⚫"

        message = f"📈 DAILY SUMMARY {version}\n\n"
        message += f"🚨 Signals Today: {total_signals}\n"
        message += f"📊 Active Positions: {active_positions}\n"
        message += f"✅ Closed Positions: {closed_positions}\n"
        message += f"💰 Total P&L: {pnl_emoji} {total_pnl_pct:+.2f}%\n"
        message += f"🎯 Win Rate: {win_rate_pct:.1f}% ({wins}W/{losses}L)\n\n"

        # Best/worst performers if available
        best_performer = summary.get("best_performer", "")
        worst_performer = summary.get("worst_performer", "")

        if best_performer:
            message += f"🏆 Best: {best_performer}\n"
        if worst_performer:
            message += f"📉 Worst: {worst_performer}\n"

        message += f"\n📅 {datetime.now().strftime('%Y-%m-%d')}"
        message += f"\n#DailySummary #SqueezeBot #{version.replace('.', '')}"

        return message

    def send_test_message(self) -> bool:
        """Send test message to verify LINE integration"""
        try:
            if not self.line_bot_api or not self.user_id:
                logger.warning("LINE not properly configured for test")
                return False

            test_message = f"🤖 Squeeze Bot Test Message v2.0\n\n"
            test_message += f"✅ LINE integration is working!\n"
            test_message += f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            test_message += f"🚀 Status: Ready for LONG/SHORT signals\n"
            test_message += f"🔧 Version: 2.0-refactored"

            self.line_bot_api.push_message(
                self.user_id, TextSendMessage(text=test_message)
            )
            logger.info("Test message sent successfully")
            return True

        except Exception as e:
            logger.error(f"Error sending test message: {e}")
            return False

    def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """Send error alert to LINE"""
        try:
            if not self.line_bot_api or not self.user_id:
                logger.warning("LINE not properly configured, cannot send error alert")
                return False

            message = f"⚠️ SQUEEZE BOT ERROR v2.0\n\n"
            message += f"🚨 Error: {error_message}\n"
            if context:
                message += f"📍 Context: {context}\n"
            message += f"\n🕐 Time: {datetime.now().strftime('%H:%M:%S')}"

            self.line_bot_api.push_message(self.user_id, TextSendMessage(text=message))
            logger.info("Error alert sent to LINE")
            return True

        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
            return False

    def verify_webhook_signature(self, body: str, signature: str) -> bool:
        """Verify LINE webhook signature"""
        try:
            if not self.handler:
                return False
            self.handler.handle(body, signature)
            return True
        except InvalidSignatureError:
            logger.error("Invalid LINE webhook signature")
            return False
        except Exception as e:
            logger.error(f"Webhook signature verification error: {e}")
            return False

    def set_user_id(self, user_id: str):
        """Set LINE user ID for notifications"""
        self.user_id = user_id
        logger.info(f"LINE user ID set: {user_id}")

    def is_configured(self) -> bool:
        """Check if LINE notifier is properly configured"""
        return (
            self.line_bot_api is not None
            and self.channel_access_token is not None
            and self.channel_secret is not None
        )

    def is_ready(self) -> bool:
        """Check if LINE notifier is ready to send messages"""
        return self.is_configured() and self.user_id is not None

    def get_status(self) -> Dict:
        """Get LINE notifier status"""
        return {
            "configured": self.is_configured(),
            "ready": self.is_ready(),
            "has_user_id": self.user_id is not None,
            "version": "2.0-refactored",
        }

    def shutdown(self):
        """Shutdown LINE notifier"""
        try:
            logger.info("Shutting down LineNotifier v2.0...")
            # Clean up any resources if needed
            logger.info("LineNotifier shutdown complete")
        except Exception as e:
            logger.error(f"Error during LineNotifier shutdown: {e}")

    # Legacy compatibility methods
    def send_position_alert(self, position_data: Dict) -> bool:
        """Legacy method - redirects to send_position_update"""
        # Convert legacy format to new format
        update_data = {
            "position": position_data,
            "events": position_data.get("events", []),
            "updates": position_data.get("updates", {}),
        }
        return self.send_position_update(update_data)