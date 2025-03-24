# GHS Hazard Label Detection Application

A real-time application for detecting and tracking Globally Harmonized System (GHS) hazard labels using computer vision.

## Features

- **Real-time Detection**: Identify GHS hazard labels in camera feeds with YOLO-based detection
- **Object Tracking**: Track detected hazard labels across video frames
- **Multi-Platform**: Works on Windows, macOS, and Linux
- **Camera Support**: Compatible with webcams, IP cameras, and video files
- **Export Capabilities**: Save screenshots and detection results to CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hazard-label-dataset.git
cd hazard-label-dataset
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

The application requires a YOLO model for detection. You can use the provided script to download a sample model:

```bash
python download_sample_model.py
```

### Start the Application

```bash
python src/app.py
```

### Using the Detection Interface

1. **Camera Controls**:
   - Select a camera from the dropdown menu
   - Choose resolution and FPS settings
   - Click "Start Camera" to begin the video feed

2. **Model Selection**:
   - Select a model from the dropdown or use "Load Model From File..."
   - The application will look for models in the `models/` directory

3. **Detection Settings**:
   - Adjust confidence threshold to control detection sensitivity
   - Modify IoU threshold for overlapping detection handling 
   - Toggle tracking on/off as needed

4. **Results and Export**:
   - View detected hazard labels in the results table
   - Take screenshots or export detection data using the File menu

## Supported GHS Hazard Label Classes

The application can detect the following GHS hazard pictograms (dependent on your trained model):

- Explosive (GHS01)
- Flammable (GHS02)
- Oxidizing (GHS03)
- Compressed Gas (GHS04)
- Corrosive (GHS05)
- Toxic (GHS06)
- Harmful/Irritant (GHS07)
- Health Hazard (GHS08)
- Environmental Hazard (GHS09)

## Troubleshooting

### Camera Issues
- If your camera doesn't appear in the list, click the refresh button
- Try different backend options in the dropdown
- For IP cameras, select "IP/URL Camera" and enter the RTSP/HTTP URL

### Detection Problems
- Ensure proper lighting for better label recognition
- Adjust the confidence threshold slider if detection is too sensitive/not sensitive enough
- Try different model files if certain hazard types aren't being detected

## Development

This project uses:
- PyQt6 for the user interface
- Ultralytics YOLOv8 for object detection
- Supervision for tracking and visualization

### Project Structure

```
hazard-label-dataset/
├── models/               # YOLO model files (.pt)
├── output/               # Screenshots and exported data
├── src/
│   ├── app/
│   │   ├── controllers/  # Application logic
│   │   ├── models/       # Data handling
│   │   ├── utils/        # Helper utilities
│   │   └── views/        # UI components
│   └── app.py            # Entry point
├── requirements.txt      # Dependencies
└── download_sample_model.py  # Utility script
```

## License

MIT License
