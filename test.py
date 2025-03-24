import cv2
import asyncio
import numpy as np
import os
import uuid
from pathlib import Path
from inference_sdk import InferenceHTTPClient
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Create FastAPI app
app = FastAPI()

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lPkTaCAqUye82G05U53F"
)

# Create temporary directory for frames
temp_dir = Path("temp_frames")
temp_dir.mkdir(exist_ok=True)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active peer connections
pcs = set()
relay = MediaRelay()

# Custom video track that processes frames with Roboflow
class RoboflowVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.model_id = "new-dataset-xjgtk/5"  # Roboflow model ID

    async def recv(self):
        frame = await self.track.recv()
        self.frame_count += 1

        # Process every 4th frame
        if self.frame_count % 10 == 0:
            # Convert to numpy array for OpenCV
            img = frame.to_ndarray(format="bgr24")
            
            # Save frame as temporary image
            temp_img_path = f"{temp_dir}/frame_{uuid.uuid4()}.jpg"
            cv2.imwrite(temp_img_path, img)
            
            try:
                # Send to Roboflow for inference
                result = CLIENT.infer(temp_img_path, model_id=self.model_id)
                
                # Process detection results
                if 'predictions' in result:
                    for prediction in result['predictions']:
                        # Extract bounding box coordinates
                        x = prediction['x']
                        y = prediction['y']
                        width = prediction['width']
                        height = prediction['height']
                        
                        # Convert from center coordinates to corner coordinates
                        x1 = int(x - width/2)
                        y1 = int(y - height/2)
                        x2 = int(x + width/2)
                        y2 = int(y + height/2)
                        
                        # Get class label and confidence
                        label = prediction['class']
                        confidence = prediction['confidence']
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw label with confidence
                        text = f"{label}: {confidence:.2f}"
                        cv2.putText(img, text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                
            # Create new frame from processed image
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            return new_frame
        
        return frame

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/offer")
async def offer(request: Request):
    data = await request.json()
    offer = RTCSessionDescription(sdp=data["sdp"]["sdp"], type=data["sdp"]["type"])
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
    
    @pc.on("track")
    def on_track(track):
        print(f"Track received: {track.kind}")
        if track.kind == "video":
            video_track = RoboflowVideoTrack(relay.subscribe(track))
            pc.addTrack(video_track)
    
    # Handle the offer
    await pc.setRemoteDescription(offer)
    
    # Create an answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {"sdp": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}}

@app.post("/ice")
async def ice(request: Request):
    return {"status": "ok"}

@app.post("/stop")
async def stop():
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*[coro for coro in coros if asyncio.iscoroutine(coro)])
    pcs.clear()
    return {"status": "stopped"}

@app.on_event("shutdown")
async def on_shutdown():
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*[coro for coro in coros if asyncio.iscoroutine(coro)])
    pcs.clear()
    
    # Clean up temporary files
    for file in temp_dir.glob("*.jpg"):
        try:
            os.remove(file)
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)