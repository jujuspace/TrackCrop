# TrackCrop

A tool for automatically cropping dashcam footage by tracking corner points.

## Demo
![DashcamCrop Demo](demo.gif)

## Features
- Automatic corner point tracking using template matching
- Real-time corner point visualization
- Adjustable template size for better tracking
- Perspective correction
- Motion smoothing with low-pass filter

## Requirements
- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jujuspace/TrackCrop.git
cd TrackCrop
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py sample.mp4
```


### Advanced Options

```bash
python main.py --input video.mp4 --output cropped.mp4 --template-size 50 --smooth-factor 0.8
```


### Parameters
- `--input`: Input video file path
- `--output`: Output video file path
- `--template-size`: Size of template for corner tracking (default: 40)
- `--smooth-factor`: Smoothing factor for motion (0-1, default: 0.7)

## How It Works
1. The program first detects corner points in the initial frame
2. These points are tracked through subsequent frames using template matching
3. A perspective transform is applied to crop and correct the video
4. Motion is smoothed using a low-pass filter

## Edge-based cropping (handles off-screen corners)

Corner tracking fails when a screen corner is **outside the frame** — you cannot
click, template, or match a vertex you cannot see. `edge_crop.py` instead tracks
points **along each edge** (top/right/bottom/left), fits a line per edge, and
derives each corner as the **intersection of two edge lines** — which is valid
even when the corner is off-screen. The perspective warp downstream is unchanged.

```bash
# headless (no GUI — works over SSH, e.g. on a remote workstation)
python edge_crop.py sample.mp4 --edges edges.json --out cropped.mp4

# interactive: click >=2 points along each edge in order top,right,bottom,left
# ('n' next edge, ESC to run); selections are saved to <name>_edges.json
python edge_crop.py sample.mp4 --pick
```

`edges.json`:

```json
{"edges": {"top": [[x,y],[x,y]], "right": [[x,y],[x,y]],
           "bottom": [[x,y],[x,y]], "left": [[x,y],[x,y]]}}
```

Outputs: an annotated video, a perspective-corrected crop, and a per-frame
`*_edge_corners.json`. The line/intersection math lives in `edge_geometry.py`
(pure NumPy) and is covered by `test_edge_geometry.py`, including an off-screen
corner case. Run the tests with `python -m pytest test_edge_geometry.py`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License (26/01/15 fixed)
CC BY-NC (Creative Commons Attribution-NonCommercial)
