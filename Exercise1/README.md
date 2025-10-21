# Exercise 1 – Computer Vision-Based Traffic Analysis

This exercise focuses on automated traffic analysis using computer vision techniques, specifically object detection and tracking for vehicle counting and trajectory analysis.

## Overview

Exercise 1 provides a complete pipeline for processing traffic video footage to extract vehicle trajectories, counts, and movement patterns. The system uses YOLOv8 for object detection and implements tracking algorithms to follow vehicles across frames. Additionally, it includes georeference capabilities to map pixel coordinates to real-world coordinates (LV95 Swiss coordinate system).

## Key Features

- **Object Detection**: YOLOv8-based vehicle detection in video frames
- **Object Tracking**: Multi-object tracking across video frames
- **Vehicle Counting**: Automated counting of vehicles passing through regions of interest
- **Georeference Mapping**: Conversion between pixel coordinates and real-world LV95 coordinates
- **ROI Selection**: Interactive tool for defining regions of interest
- **Ground Control Points**: Manual GCP definition for georeferencing
- **Pipeline Integration**: Complete end-to-end processing pipeline

## Directory Structure

```
Exercise1/
├── data/              # Input data (videos, images, reference data)
├── outputs/           # Generated outputs (counts, trajectories, visualizations)
├── runs/              # YOLO training/detection runs
├── scripts/           # Python scripts and Jupyter notebooks
│   ├── pipeline.py                      # Main processing pipeline
│   ├── object_tracking.py               # Tracking implementation
│   ├── object_counting_with_video.py    # Counting with video output
│   ├── object_counting_no_video.py      # Counting without video
│   ├── georeference.py                  # Georeferencing utilities
│   ├── pix2lv95.py                      # Pixel to LV95 conversion
│   ├── define_gcp.py                    # Ground control point definition
│   ├── roi_selector.py                  # ROI selection tool
│   ├── training.py                      # YOLO model training
│   ├── analysis.ipynb                   # Analysis notebook
│   └── ...                              # Additional analysis notebooks
└── README.md          # This file
```

## Requirements

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `ultralytics` - YOLOv8 framework
- `opencv-python` - Computer vision operations
- `torch`, `torchvision` - Deep learning framework
- `numpy` - Numerical operations
- `geopandas` - Geospatial data handling
- `shapely` - Geometric operations
- `pyproj` - Coordinate transformations
- `tqdm` - Progress bars

### Hardware Requirements

- **GPU recommended** for real-time object detection (CUDA-compatible)
- **CPU fallback** available but significantly slower
- **Memory**: At least 4GB RAM for processing typical traffic videos
- **Storage**: Depends on video size and output requirements

## Quick Start

### 1. Object Counting (No Video Output)

Count vehicles passing through a region of interest without generating video:

```bash
cd Exercise1/scripts
python object_counting_no_video.py \
    --video ../data/your_video.mp4 \
    --roi roi.json \
    --output ../outputs/counts.json
```

### 2. Object Counting (With Video Output)

Count vehicles and generate an annotated output video:

```bash
python object_counting_with_video.py \
    --video ../data/your_video.mp4 \
    --roi roi.json \
    --output ../outputs/counts.json \
    --video-output ../outputs/annotated_video.mp4
```

### 3. Object Tracking

Track vehicles across frames and extract trajectories:

```bash
python object_tracking.py \
    --video ../data/your_video.mp4 \
    --output ../outputs/trajectories.json
```

### 4. Complete Pipeline

Run the full processing pipeline including detection, tracking, and georeferencing:

```bash
python pipeline.py \
    --video ../data/your_video.mp4 \
    --roi roi.json \
    --gcp gcp.json \
    --output-dir ../outputs/
```

## Detailed Usage

### Defining Region of Interest (ROI)

Before counting vehicles, you need to define a region of interest. Use the interactive ROI selector:

```bash
python roi_selector.py \
    --image ../data/reference_frame.jpg \
    --output roi.json
```

**Instructions:**
1. Click to add polygon vertices defining the counting region
2. Press 'c' to close the polygon
3. Press 's' to save the ROI
4. Press 'q' to quit

The ROI is saved as a JSON file with polygon coordinates.

### Defining Ground Control Points (GCP)

For georeferencing, define ground control points that map pixel coordinates to real-world LV95 coordinates:

```bash
python define_gcp.py \
    --image ../data/reference_frame.jpg \
    --output gcp.json
```

**Instructions:**
1. Click on known points in the image
2. Enter the corresponding LV95 coordinates (easting, northing)
3. Add at least 4 points for accurate transformation
4. Save the GCP file

### Georeferencing and Coordinate Conversion

Convert pixel coordinates to LV95 coordinates using the defined GCPs:

```bash
python georeference.py \
    --gcp gcp.json \
    --trajectories ../outputs/trajectories.json \
    --output ../outputs/trajectories_lv95.json
```

Or use the direct conversion utility:

```python
from pix2lv95 import PixelToLV95Converter

converter = PixelToLV95Converter('gcp.json')
lv95_coords = converter.convert(pixel_x, pixel_y)
print(f"LV95: E={lv95_coords[0]:.2f}, N={lv95_coords[1]:.2f}")
```

### Training a Custom YOLO Model

If you want to train a custom model on your own data:

```bash
python training.py \
    --data your_dataset.yaml \
    --epochs 100 \
    --img 640 \
    --batch 16
```

Refer to the [Ultralytics documentation](https://docs.ultralytics.com/) for dataset preparation and training details.

## Pipeline Workflow

The complete pipeline follows these steps:

1. **Video Input**: Load traffic video footage
2. **Object Detection**: Detect vehicles in each frame using YOLOv8
3. **Object Tracking**: Associate detections across frames to form trajectories
4. **ROI Filtering**: Filter objects based on region of interest
5. **Counting**: Count vehicles entering/exiting the ROI
6. **Georeferencing** (optional): Convert trajectories to real-world coordinates
7. **Output Generation**: Export counts, trajectories, and visualizations

## Output Formats

### Counts JSON
```json
{
  "total_count": 1234,
  "counts_by_class": {
    "car": 1000,
    "truck": 150,
    "bus": 84
  },
  "timestamp_range": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"]
}
```

### Trajectories JSON
```json
{
  "trajectories": [
    {
      "id": 1,
      "class": "car",
      "frames": [0, 1, 2, ...],
      "positions": [[x0, y0], [x1, y1], [x2, y2], ...],
      "timestamps": [0.0, 0.033, 0.066, ...]
    },
    ...
  ]
}
```

### Georeferenced Trajectories
```json
{
  "trajectories": [
    {
      "id": 1,
      "class": "car",
      "lv95_positions": [[E0, N0], [E1, N1], [E2, N2], ...],
      "timestamps": [0.0, 0.033, 0.066, ...]
    },
    ...
  ]
}
```

## Analysis Notebooks

Several Jupyter notebooks are provided for exploratory analysis:

- `analysis.ipynb` - General traffic analysis and visualization
- `analysis_julian.ipynb`, `analysis_julian_2.ipynb` - Specific case studies
- `analysis_alina.ipynb`, `analysis_alisha.ipynb` - Additional analyses

To run the notebooks:

```bash
cd Exercise1/scripts
jupyter notebook
```

## Configuration Files

### ROI Configuration (roi.json)

```json
{
  "polygon": [
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
  ]
}
```

### GCP Configuration (gcp.json)

```json
{
  "gcps": [
    {
      "pixel": [px1, py1],
      "lv95": [E1, N1]
    },
    {
      "pixel": [px2, py2],
      "lv95": [E2, N2]
    },
    ...
  ]
}
```

Also supports GeoJSON format (`roi_lv95.geojson`) for ROI definition in LV95 coordinates.

## Performance Tips

### GPU Acceleration

Ensure PyTorch is using GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

If GPU is not detected:
- Install CUDA toolkit
- Reinstall PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Processing Large Videos

For large video files:
- Process in batches or segments
- Reduce frame resolution if accuracy allows
- Use video compression for output files
- Consider parallel processing for multiple videos

### Model Selection

YOLOv8 comes in different sizes with speed/accuracy tradeoffs:
- `yolov8n.pt` - Nano (fastest, lower accuracy)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (recommended balance)
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large (best accuracy, slowest)

Specify the model in your scripts or use the default.

## Coordinate Systems

### Pixel Coordinates
- Origin: Top-left corner of the image
- X-axis: Increases to the right
- Y-axis: Increases downward
- Units: Pixels

### LV95 Coordinates (Swiss National Grid)
- Origin: Defined by Swiss coordinate system
- Easting (E): Typically 2,000,000 - 3,000,000
- Northing (N): Typically 1,000,000 - 2,000,000
- Units: Meters

The georeference scripts handle transformation between these coordinate systems using affine transformations computed from ground control points.

## Troubleshooting

### Issue: YOLO model not found
**Solution**: Download the model explicitly:
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # This will download if not present
```

### Issue: GPU out of memory
**Solution**: 
- Reduce batch size
- Use a smaller YOLO model (e.g., yolov8n instead of yolov8x)
- Process video at lower resolution

### Issue: Poor tracking performance
**Solution**:
- Adjust tracking parameters (IoU threshold, maximum age, etc.)
- Improve detection quality (better model, higher resolution)
- Check for occlusions or difficult lighting conditions

### Issue: Inaccurate georeference
**Solution**:
- Add more ground control points (at least 4, more is better)
- Ensure GCPs are well-distributed across the image
- Verify LV95 coordinates are correct
- Check for lens distortion (may need camera calibration)

## Advanced Topics

### Custom Object Classes

To detect custom object classes, train a custom YOLO model:
1. Prepare a labeled dataset in YOLO format
2. Create a `data.yaml` file specifying classes
3. Run training with `training.py`
4. Update detection scripts to use the custom model

### Trajectory Analysis

Extract useful metrics from trajectories:
- **Speed estimation**: Distance / time between frames
- **Flow patterns**: Aggregate trajectories to understand traffic flow
- **Density estimation**: Count objects in zones over time
- **Origin-destination matrices**: Track where vehicles enter/exit

### Multi-Camera Setup

For multiple cameras:
- Process each camera separately
- Use georeferencing to merge trajectories in world coordinates
- Implement camera handoff logic for tracking across views

## References

### Computer Vision & Object Detection
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- OpenCV Documentation: https://docs.opencv.org/
- Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." arXiv preprint arXiv:1804.02767.

### Geospatial Processing
- Swiss Coordinate Systems: https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/reference-systems.html
- PyProj Documentation: https://pyproj4.github.io/pyproj/stable/
- GeoPandas Documentation: https://geopandas.org/

### Traffic Analysis
- Buch, N., Velastin, S. A., & Orwell, J. (2011). "A review of computer vision techniques for the analysis of urban traffic." IEEE Transactions on Intelligent Transportation Systems, 12(3), 920-939.

## Contributing

This is an educational exercise. If you find bugs or have improvements:
1. Document the issue or enhancement
2. Test your changes thoroughly
3. Follow the existing code style
4. Update documentation as needed

## License

See the main repository [LICENSE](../LICENSE) file for details.

## Support

For questions or issues:
- Check this README and inline code documentation
- Review the example notebooks
- Consult the Ultralytics YOLO documentation
- Open an issue in the repository

---

**Note**: This exercise is part of a Traffic Engineering course. The tools and scripts are designed for educational purposes and may require adaptation for production use.
