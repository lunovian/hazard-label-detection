# GHS Hazard Label Dataset Collection Tool

A Python tool for collecting and balancing GHS (Globally Harmonized System) hazard label image datasets.

## Features

- Automated collection of GHS hazard label images from multiple sources
- Image validation and duplicate detection
- Configurable target counts for each hazard class
- Rate limiting to respect source websites
- Progress tracking and logging

## Setup

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

4. Install Chrome WebDriver for Selenium.

## Usage

1. Configure target counts and search terms in `src/config/scraping_config.py`
2. Run the collection script:
```bash
python src/main.py
```

## Running the Detection Application

The project includes a real-time hazard label detection application using YOLO and PyQt6.

### Prerequisites

1. Install all required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare a YOLO model:
   - Place a trained YOLO model file (*.pt) in the `models/` directory
   - By default, the application looks for `models/hazard_yolov8n.pt`
   - You can also load a custom model through the application interface

### Launch the Application

#### Windows
Double-click on `run.bat` or run from the command line:
```bash
python src/app.py
```

#### macOS/Linux
Run from the terminal:
```bash
python src/app.py
```

### Using the Application

1. Click "Start Camera" to begin capturing from your webcam
2. Adjust detection parameters as needed:
   - Confidence threshold: Controls detection sensitivity
   - IoU threshold: Controls overlap handling for multiple detections
   - Enable/disable tracking as needed
3. Use the File menu to:
   - Load a custom model
   - Take screenshots
   - Export detection results

## Project Structure

```
hazard-label-dataset/
├── src/
│   ├── config/
│   │   └── scraping_config.py
│   ├── scrapers/
│   │   ├── base_scraper.py
│   │   └── google_scraper.py
│   ├── processors/
│   │   └── image_processor.py
│   └── main.py
├── downloaded_images/
├── requirements.txt
└── README.md
```

## Configuration

Modify `src/config/scraping_config.py` to:
- Adjust target counts for each class
- Set image size constraints
- Configure rate limiting
- Add search terms for each class

## License

MIT License
