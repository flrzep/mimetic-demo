"""
Modal WebRTC Streaming Service
Provides real-time video processing through WebRTC using Modal's infrastructure
"""

import asyncio
import base64
import io
import json
import os
import uuid
from typing import Any, Dict, List, Optional

import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel

# Modal app configuration
app = modal.App("webrtc-streaming")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi[all]",
    "aiortc",
    "opencv-python-headless", 
    "pillow",
    "numpy",
    "pydantic",
    "websockets"
])

# Models for data exchange
class PredictionResult(BaseModel):
    class_id: int
    confidence: float
    label: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None

class StreamFrame(BaseModel):
    frame_data: str  # base64 encoded frame
    width: int
    height: int
    timestamp: float

@app.function(
    image=image,
    gpu="any",  # Use GPU for inference
    scaledown_window=300,
    timeout=3600,
    concurrency_limit=10
)
@modal.web_endpoint(method="GET", label="webrtc-health")
def health():
    """Health check endpoint for WebRTC service"""
    return {"status": "ok", "service": "webrtc-streaming", "gpu": True}

@app.function(
    image=image,
    gpu="any",
    scaledown_window=300,
    timeout=3600,
    concurrency_limit=10
)
def predict_frame(frame_data: str, width: int = 640, height: int = 480) -> List[Dict]:
    """
    Process a single frame and return predictions
    This is a mock implementation - replace with your actual model
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Mock prediction - replace with your actual model inference
        predictions = [
            {
                "class_id": 0,
                "confidence": 0.85,
                "label": "detected_object",
                "bbox": {
                    "x": width * 0.2,
                    "y": height * 0.2, 
                    "width": width * 0.4,
                    "height": height * 0.3
                }
            }
        ]
        
        return predictions
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []

@app.function(
    image=image,
    gpu="any",
    scaledown_window=300,
    timeout=3600,
    concurrency_limit=5
)
class ModalWebRtcPeer:
    """Modal WebRTC Peer for real-time video processing"""
    
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.pcs: Dict[str, Any] = {}
        print(f"ModalWebRtcPeer {self.id} initialized on Modal")
        
    async def initialize(self) -> None:
        """Initialize the Modal peer"""
        print(f"ModalWebRtcPeer {self.id} initialized")
        
    async def setup_streams(self, peer_id: str) -> None:
        """Set up WebRTC streams for video processing"""
        try:
            from aiortc import MediaStreamTrack, RTCPeerConnection
            from aiortc.contrib.media import MediaPlayer, MediaRelay
            
            pc = self.pcs.get(peer_id)
            if not pc:
                return
                
            print(f"Setting up streams for peer {peer_id}")
            
            # Handle incoming tracks
            @pc.on("track")
            def on_track(track):
                print(f"Received {track.kind} track from {peer_id}")
                
                if track.kind == "video":
                    # Create processed video track
                    processed_track = self.create_processed_track(track)
                    pc.addTrack(processed_track)
                    
                @track.on("ended")
                async def on_ended():
                    print(f"Track {track.kind} ended")
                    
        except Exception as e:
            print(f"Error setting up streams: {e}")
            
    def create_processed_track(self, input_track):
        """Create a processed video track with AI predictions"""
        import asyncio

        import cv2
        import numpy as np
        from aiortc import MediaStreamTrack, VideoFrame
        
        class ProcessedVideoTrack(MediaStreamTrack):
            kind = "video"
            
            def __init__(self, track):
                super().__init__()
                self.track = track
                
            async def recv(self):
                try:
                    # Receive frame from input track
                    frame = await self.track.recv()
                    
                    # Convert to numpy array
                    img = frame.to_ndarray(format="bgr24")
                    height, width = img.shape[:2]
                    
                    # Convert to base64 for prediction
                    _, buffer = cv2.imencode('.jpg', img)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Get predictions from Modal function
                    predictions = predict_frame.remote(frame_b64, width, height)
                    
                    # Note: In a real implementation, you would draw bounding boxes here
                    # For now, we pass the frame through unchanged
                    # The frontend will handle overlay rendering
                    
                    # Create new frame
                    new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    
                    return new_frame
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    return frame
                    
        return ProcessedVideoTrack(input_track)
        
    async def generate_offer(self, peer_id: str) -> dict:
        """Generate WebRTC offer"""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            
            pc = RTCPeerConnection()
            self.pcs[peer_id] = pc
            
            await self.setup_streams(peer_id)
            
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            return {
                "type": "offer",
                "sdp": pc.localDescription.sdp,
                "peer_id": self.id
            }
            
        except Exception as e:
            print(f"Error generating offer: {e}")
            return {"error": str(e)}
            
    async def handle_offer(self, peer_id: str, offer: dict) -> dict:
        """Handle incoming WebRTC offer and generate answer"""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            
            pc = RTCPeerConnection()
            self.pcs[peer_id] = pc
            
            await self.setup_streams(peer_id)
            
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=offer["sdp"], 
                type=offer["type"]
            ))
            
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "type": "answer",
                "sdp": pc.localDescription.sdp,
                "peer_id": self.id
            }
            
        except Exception as e:
            print(f"Error handling offer: {e}")
            return {"error": str(e)}
            
    async def handle_answer(self, peer_id: str, answer: dict) -> None:
        """Handle incoming WebRTC answer"""
        try:
            from aiortc import RTCSessionDescription
            
            pc = self.pcs.get(peer_id)
            if pc:
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=answer["sdp"],
                    type=answer["type"]
                ))
                
        except Exception as e:
            print(f"Error handling answer: {e}")

@app.function(
    image=image,
    scaledown_window=300,
    timeout=3600,
    concurrency_limit=10
)
class ModalWebRtcSignalingServer:
    """Modal WebRTC Signaling Server"""
    
    def __init__(self):
        self.web_app = FastAPI()
        self.active_peers: Dict[str, Any] = {}
        self.setup_websocket_endpoint()
        
    def setup_websocket_endpoint(self):
        """Set up the WebSocket endpoint for signaling"""
        
        @self.web_app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await websocket.accept()
            
            try:
                # Create Modal peer for this client
                peer = ModalWebRtcPeer()
                
                # Store peer reference
                self.active_peers[client_id] = {
                    "websocket": websocket,
                    "peer": peer,
                    "peer_id": None
                }
                
                await peer.initialize()
                
                # Message handling loop
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        await self.handle_message(client_id, message, websocket, peer)
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        print(f"Error handling message: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))
                        
            except Exception as e:
                print(f"WebSocket connection error: {e}")
            finally:
                # Cleanup
                if client_id in self.active_peers:
                    del self.active_peers[client_id]
                try:
                    await websocket.close()
                except:
                    pass
                    
    async def handle_message(self, client_id: str, message: dict, websocket: WebSocket, peer):
        """Handle incoming WebSocket messages"""
        msg_type = message.get("type")
        
        if msg_type == "identify":
            # Client identifying itself
            peer_id = message.get("peer_id", client_id)
            self.active_peers[client_id]["peer_id"] = peer_id
            
            await websocket.send_text(json.dumps({
                "type": "identified",
                "peer_id": peer.id
            }))
            
        elif msg_type == "offer":
            # Handle WebRTC offer
            peer_id = message.get("peer_id", client_id)
            answer = await peer.handle_offer(peer_id, message)
            
            await websocket.send_text(json.dumps(answer))
            
        elif msg_type == "answer":
            # Handle WebRTC answer
            peer_id = message.get("peer_id", client_id)
            await peer.handle_answer(peer_id, message)
            
        elif msg_type == "ice-candidate":
            # Handle ICE candidates (for NAT traversal)
            # This would be implemented for production use
            pass
            
        else:
            print(f"Unknown message type: {msg_type}")

# Create signaling server instance
signaling_server = ModalWebRtcSignalingServer()

@app.function(image=image)
@modal.asgi_app()
def web():
    """Expose the signaling server as a Modal ASGI app"""
    return signaling_server.web_app

if __name__ == "__main__":
    # For local development
    print("Modal WebRTC Streaming Service")
    print("Deploy with: modal deploy webrtc_service.py")
