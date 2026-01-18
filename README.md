# sam3d-video-segmentation

A small wrapper script that runs video visualization using the official SAM-3D-Body model from Meta.

Official SAM-3D-Body repository:
https://github.com/facebookresearch/sam-3d-body


## What this repository contains

- run_video_sam3d.py  
  A Python script that runs SAM-3D-Body on every frame of a video and writes an output video.

- environment.yml  
  A Conda environment file exported from a working setup.


## Important dependency note

This script imports code from the official SAM-3D-Body repository.

Specifically, it relies on modules such as:
- notebook.utils
- tools.vis_utils

Because of this, users must also clone the official SAM-3D-Body repository and tell Python where it is located.


## Quickstart (Linux / GCP)

### Step 1: Clone both repositories

Clone this repository:
git clone https://github.com/Smart-Lizard/sam3d-video-segmentation.git

Clone the official SAM-3D-Body repository:
git clone https://github.com/facebookresearch/sam-3d-body.git


### Step 2: Create and activate the Conda environment

Go into this repository:
cd sam3d-video-segmentation

Create the environment:
conda env create -f environment.yml

Activate it:
conda activate sam_3d_body


### Step 3: Run the script on a video

Create an output folder:
mkdir -p outputs

Run the script (note the PYTHONPATH part):
PYTHONPATH=../sam-3d-body python run_video_sam3d.py --input /path/to/input.mp4 --output outputs/out.mp4


## What PYTHONPATH means

PYTHONPATH=../sam-3d-body tells Python to also look inside the official SAM-3D-Body repository when importing modules.

This is required because run_video_sam3d.py depends on code that lives in that repository.


## Notes

- A GPU is strongly recommended. CPU execution will be very slow.
- The output video is written to the path provided with --output.
