import os
import logging
from roboflow import Roboflow
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Install dependencies if not already installed
try:
    import ultralytics
    import supervision
except ImportError:
    os.system("pip install ultralytics supervision roboflow")

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # Replace with your actual API key
project = rf.workspace("colony-kx9hd").project("colony-rf2ps")
version = project.version(2)
dataset = version.download("yolov8")

# Check GPU availability
os.system("nvidia-smi")

# Define function for training and evaluation
def train_and_evaluate(epochs, imgsz, batch, lr):
    """Trains and evaluates a YOLO model with given hyperparameters."""
    model = YOLO('yolov8n.pt')  # Use YOLOv8 nano model for tuning
    
    logging.info(f"Training with epochs={epochs}, imgsz={imgsz}, batch={batch}, lr={lr}")
    results = model.train(
        data=f'{dataset.location}/data.yaml', 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch, 
        lr0=lr
    )
    
    metrics = model.val(data=f'{dataset.location}/data.yaml')
    return metrics.box.map50  # Return mAP@50 as the performance metric

# Define hyperparameter search space
epochs_range = [50, 100]
imgsz_range = [640, 1280]
batch_range = [16, 32]
lr_range = [0.01, 0.001]

# Grid Search for Best Hyperparameters
best_map = 0
best_hyperparameters = {}

for epochs in epochs_range:
    for imgsz in imgsz_range:
        for batch in batch_range:
            for lr in lr_range:
                try:
                    map50 = train_and_evaluate(epochs, imgsz, batch, lr)
                    if map50 > best_map:
                        best_map = map50
                        best_hyperparameters = {
                            'epochs': epochs, 
                            'imgsz': imgsz, 
                            'batch': batch, 
                            'lr': lr
                        }
                except Exception as e:
                    logging.error(f"Error during training with epochs={epochs}, imgsz={imgsz}, batch={batch}, lr={lr}: {e}")

# Print best hyperparameters
logging.info(f"Best mAP50: {best_map}")
logging.info(f"Best Hyperparameters: {best_hyperparameters}")

# Train final model with best hyperparameters
if best_hyperparameters:
    final_train_cmd = (
        f"yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml "
        f"epochs={best_hyperparameters['epochs']} imgsz={best_hyperparameters['imgsz']} "
        f"batch={best_hyperparameters['batch']} lr0={best_hyperparameters['lr']} "
        f"plots=True patience=10"
    )
    os.system(final_train_cmd)