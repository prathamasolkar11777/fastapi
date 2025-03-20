from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential, Conv2d, BatchNorm2d, SiLU, Module, ModuleList
from torch.nn.modules.container import Sequential as SequentialContainer

# Create FastAPI app instance
app = FastAPI()

# Add necessary PyTorch classes to safe globals for PyTorch 2.6
torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Conv2d,
    BatchNorm2d,
    SiLU,
    Module,
    ModuleList,
    SequentialContainer
])

# Load YOLO model
model = YOLO('best.pt')

# Define class mapping
CLASS_MAPPING = {
    0: "Bacillus",
    1: "Xanthomonas"
}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Bacterial Colony Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Get original image dimensions
        height, width = image_np.shape[:2]

        # Run YOLO model prediction with confidence threshold
        results = model.predict(image_np, conf=0.25)  # Adjust confidence threshold as needed

        detections = []
        count = 0
        
        # Process results
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x_min, y_min, x_max, y_max = map(float, box.xyxy[0].tolist())
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    
                    # Normalize coordinates
                    x_norm = x_min / width
                    y_norm = y_min / height
                    w_norm = (x_max - x_min) / width
                    h_norm = (y_max - y_min) / height
                    
                    # Get colony type from class mapping
                    colony_type = CLASS_MAPPING.get(class_id, "Unknown")
                    
                    # Print debug information
                    print(f"Detection: Class ID = {class_id}, Type = {colony_type}, Confidence = {confidence:.2f}")
                    
                    detections.append({
                        "x": x_norm,
                        "y": y_norm,
                        "width": w_norm,
                        "height": h_norm,
                        "type": colony_type,
                        "confidence": float(confidence)
                    })
                    count += 1

        print(f"Total detections: {count}")
        
        # Return JSON response
        return JSONResponse(
            content={
                "total_colonies": count,
                "colonies": detections
            },
            headers={
                "Content-Type": "application/json"
            }
        )

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Content-Type": "application/json"
            }
        ) 