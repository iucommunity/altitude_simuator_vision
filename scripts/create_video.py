#!/usr/bin/env python3
"""
Create a video from DroneCaptures/images folder.

Usage:
    python scripts/create_video.py --folder DroneCaptures --output output_video.mp4
    python scripts/create_video.py --folder DroneCaptures --output output_video.mp4 --fps 30
"""

import argparse
import cv2
import os
from pathlib import Path


def create_video(folder: Path, output: Path, fps: float = 30.0):
    """Create video from images in folder/images/"""
    images_folder = folder / "images"
    
    if not images_folder.exists():
        raise RuntimeError(f"Images folder not found: {images_folder}")
    
    # Get all images sorted by filename
    extensions = {'.png', '.jpg', '.PNG', '.JPG', '.jpeg', '.JPEG'}
    images = sorted([
        f for f in images_folder.iterdir()
        if f.is_file() and f.suffix in extensions
    ])
    
    if not images:
        raise RuntimeError(f"No images found in {images_folder}")
    
    print(f"Found {len(images)} images")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        raise RuntimeError(f"Could not read {images[0]}")
    
    height, width = first_img.shape[:2]
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for {output}")
    
    print(f"Writing to: {output}")
    
    # Write frames
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        out.write(img)
        
        if (i + 1) % 100 == 0 or i == len(images) - 1:
            print(f"  Progress: {i + 1}/{len(images)} frames")
    
    out.release()
    print(f"Done! Video saved to: {output}")
    print(f"Duration: {len(images) / fps:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Create video from DroneCaptures images")
    parser.add_argument("--folder", type=Path, required=True,
                        help="Path to DroneCaptures folder")
    parser.add_argument("--output", type=Path, default=Path("output_video.mp4"),
                        help="Output video file (default: output_video.mp4)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate (default: 30)")
    
    args = parser.parse_args()
    
    create_video(args.folder, args.output, args.fps)


if __name__ == "__main__":
    main()

