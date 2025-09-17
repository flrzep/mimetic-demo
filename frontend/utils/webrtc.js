/**
 * WebRTC client for streaming video to Modal inference backend
 * Based on Modal's WebRTC example but adapted for our prediction pipeline
 */

export class ModalWebRtcClient extends EventTarget {
  constructor() {
    super();
    this.localStream = null;
    this.remoteStream = null;
    this.peerConnection = null;
    this.websocket = null;
    this.peerID = null;
    this.isStreaming = false;
    this.serverUrl = process.env.NODE_ENV === 'production' 
      ? window.location.origin.replace('https://', 'wss://').replace('http://', 'ws://')
      : 'ws://localhost:8000';
    this.clientId = this.generateShortUUID();
  }

  // Generate a short UUID for peer identification
  generateShortUUID() {
    return Math.random().toString(36).substr(2, 9);
  }

  // Update status and emit event
  updateStatus(message) {
    console.log(`WebRTC Status: ${message}`);
    this.dispatchEvent(new CustomEvent('status', { 
      detail: { message }
    }));
  }

  // Get webcam media stream
  async startWebcam() {
    try {
      this.localStream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }, 
        audio: false
      });
      
      this.dispatchEvent(new CustomEvent('localStream', { 
        detail: { stream: this.localStream }
      }));
      
      this.updateStatus('Webcam started successfully');
      return this.localStream;
    } catch (err) {
      console.error('Error accessing media devices:', err);
      this.updateStatus(`Camera error: ${err.message}`);
      this.dispatchEvent(new CustomEvent('error', { 
        detail: { error: err }
      }));
      throw err;
    }
  }

  // Create and set up peer connection
  async startStreaming() {
    if (this.isStreaming) {
      this.updateStatus('Already streaming');
      return;
    }

    this.peerID = this.generateShortUUID();
    this.updateStatus('Starting streaming to Modal inference backend...');
    
    try {
      await this.initializePeerConnection();
      await this.connectWebSocket();
      await this.negotiate();
      
      this.isStreaming = true;
      this.updateStatus('Streaming started successfully');
      
      this.dispatchEvent(new CustomEvent('streamingStarted'));
    } catch (error) {
      this.updateStatus(`Failed to start streaming: ${error.message}`);
      this.dispatchEvent(new CustomEvent('error', { detail: { error } }));
      throw error;
    }
  }

  // Initialize WebRTC peer connection
  async initializePeerConnection() {
    // Configure ICE servers (STUN servers for NAT traversal)
    const configuration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    };

    this.peerConnection = new RTCPeerConnection(configuration);

    // Handle connection state changes
    this.peerConnection.onconnectionstatechange = () => {
      this.updateStatus(`Connection state: ${this.peerConnection.connectionState}`);
      
      if (this.peerConnection.connectionState === 'connected') {
        this.dispatchEvent(new CustomEvent('connected'));
      } else if (this.peerConnection.connectionState === 'disconnected' || 
                 this.peerConnection.connectionState === 'failed') {
        this.handleDisconnection();
      }
    };

    // Handle incoming remote stream
    this.peerConnection.ontrack = (event) => {
      console.log('Received remote track:', event.track.kind);
      this.remoteStream = event.streams[0];
      
      this.dispatchEvent(new CustomEvent('remoteStream', { 
        detail: { stream: this.remoteStream }
      }));
    };

    // Handle ICE candidates
    this.peerConnection.onicecandidate = (event) => {
      if (event.candidate && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({
          type: 'ice-candidate',
          candidate: event.candidate,
          peer_id: this.peerID
        }));
      }
    };

    // Add local stream tracks to peer connection
    if (this.localStream) {
      this.localStream.getTracks().forEach(track => {
        this.peerConnection.addTrack(track, this.localStream);
      });
    }
  }

  // Connect to WebSocket signaling server
  async connectWebSocket() {
    return new Promise((resolve, reject) => {
      const wsUrl = `${this.serverUrl}/stream/ws/${this.clientId}`;
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        this.updateStatus('WebSocket connected');
        
        // Identify ourselves to the server
        this.websocket.send(JSON.stringify({
          type: 'identify',
          peer_id: this.peerID
        }));
        
        resolve();
      };

      this.websocket.onmessage = async (event) => {
        try {
          const message = JSON.parse(event.data);
          await this.handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error handling WebSocket message:', error);
        }
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.updateStatus('WebSocket connection error');
        reject(error);
      };

      this.websocket.onclose = () => {
        this.updateStatus('WebSocket disconnected');
        this.handleDisconnection();
      };

      // Timeout after 10 seconds
      setTimeout(() => {
        if (this.websocket.readyState !== WebSocket.OPEN) {
          reject(new Error('WebSocket connection timeout'));
        }
      }, 10000);
    });
  }

  // Handle WebSocket messages
  async handleWebSocketMessage(message) {
    switch (message.type) {
      case 'identified':
        this.updateStatus('Successfully identified with server');
        break;
        
      case 'answer':
        await this.handleAnswer(message);
        break;
        
      case 'prediction':
        // Handle real-time predictions from the server
        this.dispatchEvent(new CustomEvent('prediction', { 
          detail: { 
            predictions: message.predictions,
            timestamp: message.timestamp,
            frame_number: message.frame_number
          }
        }));
        break;
        
      case 'error':
        console.error('Server error:', message.error);
        this.updateStatus(`Server error: ${message.error}`);
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  }

  // Create and send WebRTC offer
  async negotiate() {
    try {
      const offer = await this.peerConnection.createOffer();
      await this.peerConnection.setLocalDescription(offer);

      this.websocket.send(JSON.stringify({
        type: 'offer',
        sdp: offer.sdp,
        peer_id: this.peerID
      }));

      this.updateStatus('Offer sent, waiting for answer...');
    } catch (error) {
      console.error('Error creating offer:', error);
      throw error;
    }
  }

  // Handle WebRTC answer from server
  async handleAnswer(message) {
    try {
      const answer = new RTCSessionDescription({
        type: 'answer',
        sdp: message.sdp
      });
      
      await this.peerConnection.setRemoteDescription(answer);
      this.updateStatus('Answer received, establishing connection...');
    } catch (error) {
      console.error('Error handling answer:', error);
      throw error;
    }
  }

  // Stop streaming
  async stopStreaming() {
    if (!this.isStreaming) {
      return;
    }

    this.isStreaming = false;
    this.updateStatus('Stopping streaming...');

    // Close peer connection
    if (this.peerConnection) {
      this.peerConnection.close();
      this.peerConnection = null;
    }

    // Close WebSocket
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    // Stop local media tracks
    if (this.localStream) {
      this.localStream.getTracks().forEach(track => track.stop());
      this.localStream = null;
    }

    this.remoteStream = null;
    this.peerID = null;

    this.updateStatus('Streaming stopped');
    this.dispatchEvent(new CustomEvent('streamingStopped'));
  }

  // Handle disconnection
  handleDisconnection() {
    if (this.isStreaming) {
      this.updateStatus('Connection lost, attempting to reconnect...');
      
      // Attempt to reconnect after a delay
      setTimeout(() => {
        if (this.isStreaming) {
          this.startStreaming().catch(error => {
            console.error('Reconnection failed:', error);
            this.updateStatus('Reconnection failed');
          });
        }
      }, 3000);
    }
  }

  // Get connection statistics
  async getStats() {
    if (!this.peerConnection) {
      return null;
    }

    try {
      const stats = await this.peerConnection.getStats();
      const statsObj = {};
      
      stats.forEach((report) => {
        if (report.type === 'inbound-rtp' || report.type === 'outbound-rtp') {
          statsObj[report.type] = {
            bytesReceived: report.bytesReceived,
            bytesSent: report.bytesSent,
            packetsReceived: report.packetsReceived,
            packetsSent: report.packetsSent,
            timestamp: report.timestamp
          };
        }
      });
      
      return statsObj;
    } catch (error) {
      console.error('Error getting stats:', error);
      return null;
    }
  }
}

export default ModalWebRtcClient;
