"""
Low-Level OBS WebSocket Client Wrapper (OBS WebSocket v5 Protocol).

Handles:
  - Connection handshake & authentication (SHA256 challenge response)
  - Sending requests (OpCode 6) & parsing responses (OpCode 7)
  - Registering event listeners for OBS events (OpCode 5)
  - Automatic error handling and connection management
"""

import json
import hashlib
import base64
import logging
import threading
import time
from typing import Dict, Any, Callable, Optional

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OBSWebSocketClient")


class OBSWebSocketClient:
    """Low-level OBS WebSocket v5 Client."""

    OPCODES = {
        "HELLO": 0,
        "IDENTIFY": 1,
        "IDENTIFIED": 2,
        "REIDENTIFY": 3,
        "EVENT": 5,
        "REQUEST": 6,
        "REQUEST_RESPONSE": 7,
        "REQUEST_BATCH": 8,
        "REQUEST_BATCH_RESPONSE": 9,
    }

    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        self.host = host
        self.port = port
        self.password = password
        self.url = f"ws://{host}:{port}"

        self.ws: Optional[Any] = None
        self.is_connected = False
        self.is_authenticated = False

        self._request_counter = 0
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        self._event_listeners: Dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _generate_auth_string(self, secret: str, challenge: str, salt: str) -> str:
        """Compute OBS WebSocket v5 SHA256 authentication string."""
        auth_concat = secret + salt
        auth_hash = hashlib.sha256(auth_concat.encode("utf-8")).digest()
        auth_base64 = base64.b64encode(auth_hash).decode("utf-8")

        secret_concat = auth_base64 + challenge
        secret_hash = hashlib.sha256(secret_concat.encode("utf-8")).digest()
        return base64.b64encode(secret_hash).decode("utf-8")

    def connect(self, timeout: int = 5) -> bool:
        """Connect to OBS WebSocket server and perform handshake."""
        if not HAS_WEBSOCKET:
            logger.error("`websocket-client` library not installed. Cannot connect.")
            return False

        try:
            logger.info(f"Connecting to OBS WebSocket at {self.url}...")
            self.ws = websocket.WebSocket()
            self.ws.settimeout(timeout)
            self.ws.connect(self.url)
            self.is_connected = True

            # Receive OpCode 0 (Hello)
            raw_hello = self.ws.recv()
            hello_data = json.loads(raw_hello)

            if hello_data.get("op") != self.OPCODES["HELLO"]:
                logger.error(f"Unexpected initial OBS opcode: {hello_data}")
                self.disconnect()
                return False

            d = hello_data.get("d", {})
            auth_info = d.get("authentication")

            identify_payload: Dict[str, Any] = {
                "op": self.OPCODES["IDENTIFY"],
                "d": {
                    "rpcVersion": d.get("rpcVersion", 1),
                    "eventSubscriptions": (1 << 16) - 1  # All events
                }
            }

            if auth_info:
                challenge = auth_info.get("challenge")
                salt = auth_info.get("salt")
                auth_resp = self._generate_auth_string(self.password, challenge, salt)
                identify_payload["d"]["authentication"] = auth_resp

            self.ws.send(json.dumps(identify_payload))

            # Receive OpCode 2 (Identified)
            raw_identified = self.ws.recv()
            identified_data = json.loads(raw_identified)

            if identified_data.get("op") == self.OPCODES["IDENTIFIED"]:
                self.is_authenticated = True
                logger.info("Successfully connected & authenticated with OBS WebSocket!")
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self._worker_thread.start()
                return True
            else:
                logger.error(f"OBS Authentication failed: {identified_data}")
                self.disconnect()
                return False

        except Exception as e:
            logger.error(f"OBS WebSocket connection error: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Close OBS WebSocket connection."""
        self._stop_event.set()
        self.is_connected = False
        self.is_authenticated = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        logger.info("OBS WebSocket client disconnected.")

    def _receive_loop(self):
        """Background thread loop to process incoming WS messages."""
        self.ws.settimeout(1.0)
        while not self._stop_event.is_set() and self.is_connected and self.ws:
            try:
                msg = self.ws.recv()
                if not msg:
                    continue
                payload = json.loads(msg)
                op = payload.get("op")
                d = payload.get("d", {})

                if op == self.OPCODES["EVENT"]:
                    event_type = d.get("eventType")
                    event_data = d.get("eventData", {})
                    self._handle_event(event_type, event_data)

                elif op == self.OPCODES["REQUEST_RESPONSE"]:
                    req_id = d.get("requestId")
                    with self._lock:
                        self._pending_requests[req_id] = d

            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning(f"Error in OBS receive loop: {e}")
                break

    def send_request(self, request_type: str, request_data: Optional[Dict[str, Any]] = None, timeout: float = 3.0) -> Dict[str, Any]:
        """Send a sync request to OBS WebSocket and wait for response."""
        if not self.is_connected or not self.ws:
            return {"requestStatus": {"result": False, "code": 0, "comment": "Not connected"}}

        with self._lock:
            self._request_counter += 1
            req_id = f"req_{self._request_counter}_{time.time()}"

        payload = {
            "op": self.OPCODES["REQUEST"],
            "d": {
                "requestType": request_type,
                "requestId": req_id,
                "requestData": request_data or {}
            }
        }

        try:
            self.ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Failed to send OBS request '{request_type}': {e}")
            return {"requestStatus": {"result": False, "code": 0, "comment": str(e)}}

        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if req_id in self._pending_requests:
                    return self._pending_requests.pop(req_id)
            time.sleep(0.01)

        return {"requestStatus": {"result": False, "code": 0, "comment": "Request timeout"}}

    def on_event(self, event_name: str, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for specific OBS event."""
        if event_name not in self._event_listeners:
            self._event_listeners[event_name] = []
        self._event_listeners[event_name].append(callback)

    def _handle_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger registered callbacks on event arrival."""
        if event_type in self._event_listeners:
            for cb in self._event_listeners[event_type]:
                try:
                    cb(event_data)
                except Exception as e:
                    logger.error(f"Error in event listener for {event_type}: {e}")
