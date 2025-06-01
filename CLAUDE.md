# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Dot Plate Generator is a desktop application that converts images to 3D dot plates (pixel art style 3D objects). It allows users to import images, convert them to dot art style, edit them, and export as 3D STL files for 3D printing.

## Key Features

- Import images (PNG, JPG, JPEG, GIF)
- Resize and quantize images to dot art style
- Support for large images (up to 512x512)
- Multiple color reduction algorithms (simple quantization, median cut, K-means, octree, toon-style, no reduction)
- Dot editing (paint function with variable brush size, fill, eyedropper)
- 3D model generation with adjustable parameters
- Custom palette settings
- Project management (save/load projects)
- STL file export
- AI brush function using OpenAI API

## Project Structure

- `dot_plate_generator_gui.py`: Main application code with GUI implementation
- `sample.py`: Simple CLI version of the core functionality
- `design_doc.md`: Detailed design documentation
- `help.html`: Help documentation for users
- `src/`: Directory structure for future code organization
- `input/`: Directory for input images (optional)

## Development Environment

The application is built with:
- Python 3.x
- PyQt5 (GUI framework)
- PIL/Pillow (image processing)
- NumPy (numerical computation)
- scikit-image (image processing)
- trimesh (3D model generation)
- matplotlib (3D visualization)
- openai (for AI brush functionality)

## Common Commands

### Running the Application

```bash
# Run the main GUI application
python dot_plate_generator_gui.py

# Run the CLI version with input and output paths
python sample.py input_image.png output_model.stl
```

### Required Dependencies

```bash
# Install required dependencies
pip install PyQt5 pillow numpy scipy trimesh shapely scikit-image matplotlib openai
```

## Architecture

The application follows a modular architecture:
- GUI implementation using PyQt5
- Image processing functions for color quantization and dot conversion
- 3D model generation using trimesh
- Project data saved in JSON format

## Project Files

- `.dpp`: Project files (JSON format) that store:
  - Version information
  - Image path
  - Parameter values
  - Pixel data (Base64 encoded)
  - Wall color
  - Same color merge option state
  - Latest edit history

## Main Workflows

1. **Image to Dot Plate Generation**:
   - Load an image
   - Adjust parameters
   - Edit individual dots if needed
   - Export STL

2. **Creating Dot Art from Scratch**:
   - Load a reference image (optional)
   - Clear the preview
   - Manually draw dot art
   - Export STL

3. **Project Management**:
   - Save work in progress as project file
   - Load project file to continue work later