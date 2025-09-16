"""
Modal WebRTC infrastructure adapted from Modal examples.
Provides base classes for WebRTC peers and signaling server.
"""

import asyncio
import json
import os
import uuid
from typing import Dict, Any, Optional

import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


class ModalWebRtcPeer:
    """Base class for Modal WebRTC peers"""
    
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.pcs: Dict[str, Any] = {}
        
    async def initialize(self) -> None:
        """Override for custom initialization logic"""
        pass
        
    async def setup_streams(self, peer_id: str) -> None:
        """Override to set up media streams for the peer connection"""
        raise NotImplementedError("Subclasses must implement setup_streams")
        
    async def run_streams(self, peer_id: str) -> None:
        """Override for custom stream running logic"""
        pass
        
    async def get_turn_servers(self, peer_id: str = None, msg: dict = None) -> dict:
        """Override to provide TURN server configuration"""
        return {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
            ]
        }
        
    async def generate_offer(self, peer_id: str) -> dict:
        """Generate WebRTC offer"""
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
        
    async def handle_offer(self, peer_id: str, offer: dict) -> dict:
        """Handle incoming WebRTC offer and generate answer"""
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
        
    async def handle_answer(self, peer_id: str, answer: dict) -> None:
        """Handle incoming WebRTC answer"""
        from aiortc import RTCSessionDescription
        
        pc = self.pcs.get(peer_id)
        if pc:
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=answer["sdp"],
                type=answer["type"]
            ))


class ModalWebRtcSignalingServer:
    """Base class for Modal WebRTC signaling server"""
    
    def __init__(self):
        self.web_app = FastAPI()
        self.active_peers: Dict[str, Any] = {}
        self.setup_websocket_endpoint()
        
    def get_modal_peer_class(self):
        """Override to return the Modal peer class to spawn"""
        raise NotImplementedError("Subclasses must implement get_modal_peer_class")
        
    async def initialize(self) -> None:
        """Override for custom initialization logic"""
        pass
        
    def setup_websocket_endpoint(self):
        """Set up the WebSocket endpoint for signaling"""
        
        @self.web_app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await websocket.accept()
            
            try:
                # Spawn Modal peer for this client
                peer_class = self.get_modal_peer_class()
                peer = peer_class()
                
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
                    
    async def handle_message(self, client_id: str, message: dict, websocket: WebSocket, peer: ModalWebRtcPeer):
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

    @modal.asgi_app()
    def web(self):
        """Expose the FastAPI app as a Modal ASGI app"""
        return self.web_app
