# Pyperflut - Pixelflut One-Shot Client

`pyperflut.py` is a Python client for sending content to a Pixelflut server, a pixel-based display protocol. It supports multiple content types and rendering modes, optimized for one-shot sends rather than continuous streaming. The screen size is dynamically fetched from the server's `SIZE` response (e.g., 1280x720), not hardcoded.

## Capabilities

- **Image Display**:
  - Send static image files (e.g., PNG, JPG) to the Pixelflut canvas.
  - Options: Position (`-x`, `-y`), scale (`--scale`), or fit to screen (`--fit`).
  - Example: `uv run pyperflut.py 192.168.5.51 1234 -i image.png --fit`

- **Website Capture**:
  - Capture and display a website screenshot using Selenium.
  - Static mode: Shows the top portion matching the server’s screen size.
  - Scroll mode: Scrolls through the full page with configurable interval (`--scroll`) and cycles (`--cycles`), refreshing the site each cycle for live updates.
  - Example: `uv run pyperflut.py 192.168.5.51 1234 -u https://news.ycombinator.com --scroll 5.0 --cycles 2`

- **Markdown Rendering**:
  - Convert Markdown files to HTML and display them.
  - Supports static display or scrolling for longer content.
  - Example: `uv run pyperflut.py 192.168.5.51 1234 -m document.md --scroll 5.0`

- **HTML Rendering**:
  - Display local HTML files directly.
  - Supports static or scrolling modes.
  - Example: `uv run pyperflut.py 192.168.5.51 1234 -l page.html`

- **Animated GIF Playback**:
  - Play animated GIFs fullscreen, matching the server’s screen size.
  - Options: Custom frame rate (`--fps`) to override GIF timing, loop indefinitely (`--loop`).
  - Example: `uv run pyperflut.py 192.168.5.51 1234 -g animation.gif --fps 5.0 --loop`

- **Screen Clearing**:
  - Clear the Pixelflut canvas with a solid color (default black).
  - Example: `uv run pyperflut.py 192.168.5.51 1234 -i image.png -c`

## Setup
Dependencies are managed via `uv` and defined in `pyproject.toml`. To set up the environment:

1. Ensure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Sync dependencies: `uv sync`

## Usage
Run with `uv run pyperflut.py <host> <port> [options]`. Use `-h` for full command-line help.

## Notes
- Screen size (e.g., 1280x720) is an example; the client queries the server’s actual size via the `SIZE` command.
- Uses multi-threading (`--threads`) for faster pixel sending.
- Temporary files are cleaned up automatically unless `--keep-temp` is used.
