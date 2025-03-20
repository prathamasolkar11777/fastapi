# Bacterial Colony Detection API

A FastAPI-based API for detecting and classifying bacterial colonies in images using YOLO (You Only Look Once) model.

## Features

- Image upload and processing
- Bacterial colony detection using YOLO model
- Classification of colonies (Bacillus and Xanthomonas)
- Normalized coordinate output for detected colonies
- Confidence scores for each detection

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your trained YOLO model file (`best.pt`) in the project root directory.

## Running the Application

To run the application:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## API Endpoints

- `GET /`: Returns a welcome message
- `POST /predict/`: Accepts an image file and returns detected colonies
  - Request: Multipart form data with key 'file' containing the image
  - Response: JSON containing total colonies count and detailed detection information

## Response Format

```json
{
    "total_colonies": <number>,
    "colonies": [
        {
            "x": <normalized_x>,
            "y": <normalized_y>,
            "width": <normalized_width>,
            "height": <normalized_height>,
            "type": "<colony_type>",
            "confidence": <confidence_score>
        },
        ...
    ]
}
```

## Note

Make sure you have the trained YOLO model file (`best.pt`) in the project directory before running the application. 