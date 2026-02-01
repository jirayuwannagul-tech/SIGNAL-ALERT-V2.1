import logging
import os
import time
from threading import Thread
from flask import Flask, jsonify, request

# New refactored services
from app.services.config_manager import ConfigManager
from app.services.data_manager import DataManager
from app.services.position_manager import PositionManager

# Legacy services (will be refactored)
from app.services.signal_detector import SignalDetector
from app.services.scheduler import SignalScheduler
from app.services.sheets_logger import SheetsLogger
from app.services.line_notifier import LineNotifier
from app.services.performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_version():
    """Read and auto-increment version on startup"""
    try:
        # Read current version
        if os.path.exists('version.txt'):
            with open('version.txt', 'r') as f:
                version = int(f.read().strip())
        else:
            version = 106
        
        # Increment version
        new_version = version + 1
        
        # Save new version
        with open('version.txt', 'w') as f:
            f.write(str(new_version))
        
        logger.info(f"🔢 Version auto-incremented: 2.0.{version} → 2.0.{new_version}")
        return f"2.0.{new_version}"
    except Exception as e:
        logger.error(f"Error reading version: {e}")
        return "2.0.106"

VERSION = get_version()

# Initialize Flask app
app = Flask(__name__)

# Global services - refactored architecture
services = {
    # Core refactored services
    "config_manager": None,
    "data_manager": None, 
    "position_manager": None,
    
    # Legacy services (to be updated)
    "signal_detector": None,
    "scheduler": None,
    "line_notifier": None,
    "sheets_logger": None,
    "performance_analyzer": None,
    
    "initialized": False,
}


def initialize_services_background():
    """Initialize all services with new refactored architecture"""
    try:
        logger.info(f"🚀 Starting SIGNAL-ALERT {VERSION} service initialization...")
        
        # Step 1: Initialize ConfigManager (Singleton)
        services["config_manager"] = ConfigManager()
        logger.info("✅ ConfigManager initialized")
        
        # Step 2: Initialize DataManager (replaces PriceFetcher + DataUpdater)
        services["data_manager"] = DataManager()
        logger.info("✅ DataManager initialized (replaces PriceFetcher + DataUpdater)")
        
        # Step 3: Initialize PositionManager (replaces PositionTracker + PriceMonitor logic)
        services["position_manager"] = PositionManager(services["data_manager"])
        logger.info("✅ PositionManager initialized (replaces PositionTracker + PriceMonitor logic)")
        
        # Step 4: Initialize notification services with ConfigManager
        try:
            line_config = services["config_manager"].get_line_config()
            services["line_notifier"] = LineNotifier(line_config)
            logger.info("✅ LineNotifier initialized with ConfigManager")
        except Exception as e:
            logger.warning(f"⚠️ LineNotifier failed to initialize: {e}")
            services["line_notifier"] = None
            
        try:
            google_config = services["config_manager"].get_google_config()
            # 👇 ใส่ # ไว้หน้า 2 บรรทัดนี้
            # services["sheets_logger"] = SheetsLogger(google_config)
            # logger.info("✅ SheetsLogger initialized with ConfigManager")
            services["sheets_logger"] = None # 👈 เพิ่มบรรทัดนี้เพื่อให้ระบบรู้ว่าไม่ต้องใช้
        except Exception as e:
            logger.warning(f"⚠️ SheetsLogger failed to initialize: {e}")
            services["sheets_logger"] = None
        
        # Step 5: Initialize SignalDetector with new services
        try:
            signal_config = {
                "data_manager": services["data_manager"],
                "position_manager": services["position_manager"],
                "config_manager": services["config_manager"]
            }
            services["signal_detector"] = SignalDetector(signal_config)
            logger.info("✅ SignalDetector initialized with refactored services")
        except Exception as e:
            logger.error(f"❌ SignalDetector initialization failed: {e}")
            services["signal_detector"] = None
        
        # Step 6: Initialize Scheduler with new architecture
        try:
            scheduler_config = services["config_manager"].get_all()
            services["scheduler"] = SignalScheduler(scheduler_config)
            
            # Inject refactored services into scheduler
            services["scheduler"].set_services(
                signal_detector=services["signal_detector"],
                position_manager=services["position_manager"],
                line_notifier=services["line_notifier"],
                sheets_logger=services["sheets_logger"]
            )
            logger.info("✅ SignalScheduler initialized with refactored services")
            
            # Auto-start scheduler
            services["scheduler"].start_scheduler()
            logger.info("✅ Scheduler auto-started")
            
        except Exception as e:
            logger.error(f"❌ SignalScheduler initialization failed: {e}")
            services["scheduler"] = None
        
        # Step 7: Initialize PerformanceAnalyzer
        try:
            services["performance_analyzer"] = PerformanceAnalyzer(
                config={},
                sheets_logger=services["sheets_logger"]
            )
            logger.info("✅ PerformanceAnalyzer initialized")
        except Exception as e:
            logger.warning(f"⚠️ PerformanceAnalyzer failed to initialize: {e}")
            services["performance_analyzer"] = None
        
        # Step 8: Start automatic position monitoring
        if services["position_manager"] and services["sheets_logger"]:
            try:
                # Start background position monitoring thread
                monitor_thread = Thread(
                    target=start_position_monitoring,
                    daemon=True
                )
                monitor_thread.start()
                logger.info("✅ Background position monitoring started")
            except Exception as e:
                logger.warning(f"⚠️ Failed to start background monitoring: {e}")
        
        services["initialized"] = True
        logger.info(f"🎉 All services initialized successfully! SIGNAL-ALERT {VERSION} ready")
        
    except Exception as e:
        logger.error(f"💥 Service initialization failed: {e}")
        services["initialized"] = False


def start_position_monitoring():
    """Background thread for continuous position monitoring"""
    monitor_interval = 30  # 30 seconds
    
    while True:
        try:
            if services["initialized"] and services["position_manager"]:
                updates = services["position_manager"].update_positions()
                
                if updates:
                    logger.info(f"📊 Updated {len(updates)} positions")
                    
                    # Log to sheets if available
                    if services["sheets_logger"]:
                        try:
                            for position_id, update_info in updates.items():
                                if update_info.get('position_closed'):
                                    position = services["position_manager"].positions.get(position_id)
                                    if position:
                                        services["sheets_logger"].log_position_close(position)
                                
                                # ✅ แก้ indent ให้อยู่ใน for loop เดียวกัน
                                for tp_level in ['TP1', 'TP2', 'TP3']:
                                    tp_key = f'{tp_level}_hit'
                                    if tp_key in update_info and update_info[tp_key].get('hit'):
                                        position = services["position_manager"].positions.get(position_id)
                                        if position:
                                            services["sheets_logger"].log_tp_hit(position, update_info[tp_key])
                                            logger.info(f"Logged {tp_level} hit for {position_id}")
                                        
                        except Exception as e:
                            logger.error(f"Error logging to sheets: {e}")
                            
            time.sleep(monitor_interval)
            
        except Exception as e:
            logger.error(f"Error in position monitoring thread: {e}")
            time.sleep(monitor_interval)


# Start background initialization
Thread(target=initialize_services_background, daemon=True).start()


@app.route("/")
def root():
    """Home endpoint - system information"""
    config = services["config_manager"]
    cache_stats = services["data_manager"].get_cache_stats() if services["data_manager"] else {}
    
    return jsonify({
        "system": "SIGNAL-ALERT",
        "version": VERSION,
        "status": "running",
        "services_ready": services["initialized"],
        "architecture": "refactored",
        "services": {
            "config_manager": services["config_manager"] is not None,
            "data_manager": services["data_manager"] is not None,
            "position_manager": services["position_manager"] is not None,
            "signal_detector": services["signal_detector"] is not None,
            "scheduler": services["scheduler"] is not None
        },
        "features": [
            "Centralized Data Management",
            "Unified Position Tracking", 
            "Single Source Price Fetching",
            "Automated TP/SL Detection",
            "Google Sheets Integration",
            "Configuration Management",
            "Comprehensive Error Handling"
        ],
        "metrics": {
            "cache_stats": cache_stats,
            "debug_mode": config.is_debug_mode() if config else False
        }
    })


@app.route("/health")
def health_check():
    """System health check"""
    health_data = {
        "status": "healthy" if services["initialized"] else "initializing",
        "timestamp": time.time(),
        "services_initialized": services["initialized"],
        "version": VERSION
    }
    
    status_code = 200 if services["initialized"] else 503
    return jsonify(health_data), status_code


@app.route('/api/test/line', methods=['POST', 'GET'])
def test_line_notification():
    """Test LINE notification"""
    try:
        if not services["line_notifier"]:
            return jsonify({
                "success": False,
                "error": "LineNotifier not initialized"
            }), 500
        
        success = services["line_notifier"].send_test_message()
        
        return jsonify({
            "success": success,
            "message": "Test message sent to LINE" if success else "Failed to send",
            "line_status": services["line_notifier"].get_status()
        })
        
    except Exception as e:
        logger.error(f"Test LINE error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/line/webhook', methods=['POST'])
def line_webhook():
    """รับ webhook จาก LINE เพื่อดู Group ID"""
    try:
        # ดึง signature และ body จาก request
        signature = request.headers.get('X-Line-Signature')
        body = request.get_data(as_text=True)
        
        logger.info(f"📥 Received LINE webhook")
        
        # แปลง JSON body เป็น dict
        import json
        data = json.loads(body)
        
        # วนลูปดู events ที่ได้รับ
        for event in data.get('events', []):
            # เช็กว่ามาจากกลุ่มหรือไม่
            source = event.get('source', {})
            
            if source.get('type') == 'group':
                # 🎯 นี่คือ Group ID ที่เราต้องการ!
                group_id = source.get('groupId')
                
                # แสดง log
                logger.info(f"🎯 GROUP ID FOUND: {group_id}")
                logger.info(f"📝 Message Type: {event.get('type')}")
                logger.info(f"💬 Text: {event.get('message', {}).get('text', 'N/A')}")
                
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/startup")
def startup_probe():
    """Startup probe - always return OK for Cloud Run"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time()
    }), 200


@app.route("/keepalive")
def keepalive():
    """Keepalive endpoint for Cloud Run"""
    try:
        scheduler_status = "unknown"
        position_count = 0
        
        if services["initialized"] and services["scheduler"]:
            try:
                status_info = services["scheduler"].get_scheduler_status()
                scheduler_status = status_info.get("status", "unknown")
                
                # Auto-restart scheduler if stopped
                if scheduler_status == "stopped":
                    services["scheduler"].start_scheduler()
                    logger.info("🔄 Auto-restarted scheduler from keepalive")
                    scheduler_status = "restarted"
            except Exception as e:
                logger.warning(f"Scheduler check failed in keepalive: {e}")
                scheduler_status = "error"
        
        if services["position_manager"]:
            try:
                summary = services["position_manager"].get_positions_summary()
                position_count = summary["active_positions"]
            except Exception as e:
                logger.warning(f"Position count check failed: {e}")
        
        return jsonify({
            "status": "alive",
            "timestamp": time.time(),
            "services_initialized": services["initialized"],
            "scheduler_status": scheduler_status,
            "active_positions": position_count,
            "uptime_check": "ok",
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Keepalive endpoint error: {e}")
        return jsonify({
            "status": "alive",
            "timestamp": time.time(),
            "error": str(e),
            "version": VERSION
        }), 200


def require_services(f):
    """Decorator to check if services are ready"""
    def wrapper(*args, **kwargs):
        if not services["initialized"]:
            return jsonify({
                "error": "Services are still initializing. Please wait...",
                "retry_after": 30,
                "version": VERSION
            }), 503
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


@app.route("/api/signals")
@require_services
def get_signals():
    """Scan for trading signals with new architecture"""
    symbols = request.args.get("symbols", "BTCUSDT,ETHUSDT")
    timeframes = request.args.get("timeframes", "4h")
    
    symbols_list = [s.strip() for s in symbols.split(",")]
    timeframes_list = [t.strip() for t in timeframes.split(",")]
    
    try:
        signals_found = []
        
        for symbol in symbols_list:
            for timeframe in timeframes_list:
                signal = services["signal_detector"].analyze_symbol(symbol, timeframe)
                if signal:
                    signals_found.append(signal)
        
        return jsonify({
            "status": "success",
            "signals": signals_found,
            "signals_found": len(signals_found),
            "timestamp": time.time(),
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Error in get_signals: {e}")
        return jsonify({"error": str(e), "version": VERSION}), 500


@app.route("/api/positions")
@require_services
def get_positions():
    """Get all positions"""
    try:
        active_positions = services["position_manager"].get_active_positions()
        summary = services["position_manager"].get_positions_summary()
        
        return jsonify({
            "status": "success",
            "active_positions": active_positions,
            "summary": summary,
            "total_positions": summary["total_positions"],
            "active_count": summary["active_positions"],
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Error in get_positions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions/summary")
@require_services
def get_positions_summary():
    """Get positions summary"""
    try:
        summary = services["position_manager"].get_positions_summary()
        return jsonify({
            "status": "success",
            "summary": summary,
            "version": VERSION
        })
    except Exception as e:
        logger.error(f"Error in get_positions_summary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions/status/<symbol>/<timeframe>")
@require_services  
def get_position_status(symbol, timeframe):
    """Get specific position status"""
    try:
        position = services["position_manager"].get_position_status(symbol.upper(), timeframe)
        
        return jsonify({
            "status": "success",
            "position_found": position is not None,
            "position": position,
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Error getting position status for {symbol} {timeframe}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions/close", methods=["POST"])
@require_services
def close_position():
    """Manually close a position"""
    try:
        data = request.get_json()
        position_id = data.get("position_id")
        reason = data.get("reason", "MANUAL")
        
        if not position_id:
            return jsonify({"error": "position_id required"}), 400
        
        success = services["position_manager"].close_position(position_id, reason)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Position {position_id} closed",
                "reason": reason,
                "version": VERSION
            })
        else:
            return jsonify({
                "error": "Position not found or already closed",
                "position_id": position_id
            }), 404
            
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions/update", methods=["POST"])
@require_services
def update_positions():
    """Update all positions with current prices"""
    try:
        updates = services["position_manager"].update_positions()
        
        return jsonify({
            "status": "success",
            "positions_updated": len(updates),
            "updates": updates,
            "timestamp": time.time(),
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Error updating positions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/monitor/status")
@require_services
def get_monitor_status():
    """Get monitoring status"""
    try:
        summary = services["position_manager"].get_positions_summary()
        cache_stats = services["data_manager"].get_cache_stats()
        
        return jsonify({
            "status": "success",
            "monitoring": True,
            "active_positions_count": summary["active_positions"],
            "total_positions": summary["total_positions"],
            "cache_stats": cache_stats,
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Error getting monitor status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/monitor/force-check", methods=["POST"])
@require_services
def force_check_positions():
    """Force check all positions immediately"""
    try:
        updates = services["position_manager"].update_positions()
        
        return jsonify({
            "status": "success",
            "message": "Force check completed",
            "positions_checked": len(services["position_manager"].get_active_positions()),
            "updates": updates,
            "timestamp": time.time(),
            "version": VERSION
        })
        
    except Exception as e:
        logger.error(f"Error in force check: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/monitor/check/<symbol>")
@require_services
def get_symbol_price(symbol):
    """Get current price for specific symbol"""
    try:
        price = services["data_manager"].get_single_price(symbol.upper())
        
        if price is not None:
            return jsonify({
                "status": "success", 
                "symbol": symbol.upper(),
                "current_price": price,
                "timestamp": time.time(),
                "version": VERSION
            })
        else:
            return jsonify({
                "error": f"Failed to get price for {symbol}",
                "symbol": symbol.upper()
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scheduler/start", methods=["POST"])
@require_services
def start_scheduler():
    """Start automatic scheduler"""
    try:
        services["scheduler"].start_scheduler()
        return jsonify({
            "status": "success", 
            "message": "Scheduler started",
            "version": VERSION
        })
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scheduler/stop", methods=["POST"])
@require_services
def stop_scheduler():
    """Stop automatic scheduler"""
    try:
        services["scheduler"].stop_scheduler()
        return jsonify({
            "status": "success",
            "message": "Scheduler stopped", 
            "version": VERSION
        })
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scheduler/status")
@require_services
def get_scheduler_status():
    """Get scheduler status"""
    try:
        status = services["scheduler"].get_scheduler_status()
        return jsonify({
            "status": "success",
            "scheduler": status,
            "version": VERSION
        })
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/services")
@require_services
def debug_services():
    """Debug endpoint for service status"""
    try:
        debug_info = {
            "version": VERSION,
            "initialized": services["initialized"],
            "services": {}
        }
        
        # Check each service
        for service_name, service in services.items():
            if service_name == "initialized":
                continue
                
            if service is None:
                debug_info["services"][service_name] = "not_available"
            elif service_name == "config_manager":
                debug_info["services"][service_name] = {
                    "available": True,
                    "debug_mode": service.is_debug_mode(),
                    "version": service.get("VERSION", "unknown")
                }
            elif service_name == "data_manager":
                debug_info["services"][service_name] = {
                    "available": True,
                    "cache_stats": service.get_cache_stats()
                }
            elif service_name == "position_manager":
                summary = service.get_positions_summary()
                debug_info["services"][service_name] = {
                    "available": True,
                    "active_positions": summary["active_positions"],
                    "total_positions": summary["total_positions"],
                    "win_rate": summary["win_rate_pct"]
                }
            elif service_name == "scheduler":
                try:
                    status = service.get_scheduler_status()
                    debug_info["services"][service_name] = {
                        "available": True,
                        "status": status.get("status", "unknown")
                    }
                except Exception as e:
                    debug_info["services"][service_name] = {"error": str(e)}
            else:
                debug_info["services"][service_name] = "available"
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in debug services: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/positions")
@require_services
def debug_positions():
    """Debug positions in detail"""
    try:
        active_positions = services["position_manager"].get_active_positions()
        summary = services["position_manager"].get_positions_summary()
        
        return jsonify({
            "version": VERSION,
            "total_positions": summary["total_positions"],
            "active_positions": summary["active_positions"],
            "closed_positions": summary["closed_positions"],
            "win_rate_pct": summary["win_rate_pct"],
            "total_pnl_pct": summary["total_pnl_pct"],
            "active_positions_detail": active_positions
        })
        
    except Exception as e:
        logger.error(f"Error in debug positions: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Fix WERKZEUG error  
    if 'WERKZEUG_SERVER_FD' in os.environ:
        del os.environ['WERKZEUG_SERVER_FD']
        logger.info("Removed WERKZEUG_SERVER_FD from environment")
    
    # --- ส่วนที่แก้ไขเพื่อดัก Error $PORT ---
    raw_port = os.environ.get("PORT", "8080")
    if raw_port == "$PORT" or not raw_port.isdigit():
        port = 8080 # ถ้าเป็นตัวหนังสือ ให้บังคับใช้ 8080 ทันที
    else:
        port = int(raw_port)
    
    logger.info(f"🚀 Starting SIGNAL-ALERT {VERSION} on port {port}")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logger.error(f"💥 Failed to start Flask application: {e}")
        raise
