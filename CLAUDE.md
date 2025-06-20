# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pyperflut is a Python client for the Pixelflut protocol - a pixel-based display system. The main application (`pyperflut.py`) sends various content types (images, websites, Markdown, HTML, animated GIFs) to a Pixelflut server for display on a shared canvas.

## Key Architecture

- **Single-file application**: All functionality is contained in `pyperflut.py`
- **PixelflutClient class**: Main client that handles connections, pixel distribution, and content rendering
- **Protocol support**: Both text (`PX x y rrggbb`) and binary (`PB`) protocols for improved performance
- **Multi-threading**: Distributes pixel sending across multiple threads for faster transmission
- **Dynamic screen sizing**: Queries server for actual canvas dimensions rather than hardcoding

### Core Components

- **Connection management**: Socket pool with automatic reconnection and timeout handling
- **Content rendering**: Website capture via Selenium WebDriver, Markdown-to-HTML conversion
- **Image processing**: PIL-based image loading, scaling, and pixel extraction
- **Animation support**: Frame-by-frame GIF playback with custom timing controls

## Development Commands

### Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Running the Application
```bash
# Basic image display
uv run pyperflut.py <host> <port> -i image.png

# Website capture with scrolling and delta mode
uv run pyperflut.py <host> <port> -u https://example.com --scroll 5.0 --cycles 2 --delta

# Markdown rendering
uv run pyperflut.py <host> <port> -m document.md --scroll 3.0

# Animated GIF playback with delta optimization
uv run pyperflut.py <host> <port> -g animation.gif --fps 10 --loop --delta
```

### Dependencies

The project uses `uv` for dependency management with these key libraries:
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Fast array operations for pixel processing and delta comparison
- **Selenium + webdriver-manager**: Web page capture and rendering
- **Requests**: HTTP client for downloading remote content
- **Markdown**: Converting Markdown files to HTML

## Protocol Details

The client supports two Pixelflut protocols:
- **Text protocol**: `PX x y rrggbb\n` - human readable but slower
- **Binary protocol**: `PB` + 4-byte coordinates + 4-byte RGBA - faster transmission

The client automatically detects server capabilities via the `HELP` command and uses binary protocol when available.

## Content Types Supported

1. **Static images**: PNG, JPG, etc. with positioning and scaling options
2. **Websites**: Live capture with optional scrolling and refresh cycles  
3. **Markdown files**: Converted to styled HTML then rendered
4. **Local HTML files**: Direct rendering with scrolling support
5. **Animated GIFs**: Frame-by-frame playback with timing controls
6. **Screen clearing**: Solid color fills across the entire canvas

## Performance Optimizations

The client includes several performance optimizations:

- **Delta mode (`--delta`)**: Only sends pixels that have changed compared to the previous frame, dramatically reducing bandwidth for animations and live content
- **Region-based threading**: Distributes pixels in horizontal bands for better cache locality
- **Large batch sizes**: 8000 pixels per batch (binary) or 5000 (text) for reduced network overhead
- **NumPy acceleration**: Fast array operations for pixel processing and comparison
- **8MB socket buffers**: Reduced network latency for large transfers
- **Pre-allocated binary commands**: Eliminates memory allocations in hot loops