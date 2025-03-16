#!/usr/bin/env python3
"""
Pixelflut One-Shot Client
A simple Python client for Pixelflut that sends images in one shot rather than continuous streaming.
"""

import socket
import argparse
from PIL import Image
import threading
import time
import random
import sys
import io
import tempfile
import os
import requests  # Added for GIF URL downloading
from typing import Tuple, List, Optional

# For web capture functionality
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# For Markdown conversion
import markdown


class PixelflutClient:
    """A client for sending images to a Pixelflut server in one shot."""

    def __init__(self, host: str, port: int, threads: int = 4):
        """Initialize the Pixelflut client."""
        self.host = host
        self.port = port
        self.threads = threads
        self.sockets = []
        self.screen_size = None

    def connect(self):
        """Connect to the Pixelflut server and create socket pool."""
        self.close()
        for _ in range(self.threads):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            self.sockets.append(sock)

        try:
            size_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            size_sock.connect((self.host, self.port))
            size_sock.sendall(b"SIZE\n")
            response = size_sock.recv(16).decode().strip()
            if response.startswith("SIZE "):
                width, height = map(int, response[5:].split(" "))
                self.screen_size = (width, height)
                print(f"Server screen size: {width}x{height}")
            size_sock.close()
        except Exception as e:
            print(f"Warning: Could not get screen size: {e}")

    def close(self):
        """Close all socket connections."""
        for sock in self.sockets:
            try:
                sock.close()
            except:
                pass
        self.sockets = []

    def capture_website(self, url: str, width: int = 1280, height: int = 720,
                       wait_time: int = 3, keep_temp: bool = False,
                       full_page: bool = False, capture_height: int = None,
                       max_height: int = 3000) -> str:
        """Capture a website or local HTML file as an image matching the Pixelflut canvas."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for website capture. Install it with: pip install selenium webdriver-manager")

        # Match server screen size exactly
        if self.screen_size:
            width, height = self.screen_size
            print(f"Matching capture to server screen size: {width}x{height}")

        print(f"Capturing content from: {url}")
        print("Setting up headless browser...")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(f"--window-size={width},{height}")
        chrome_options.add_argument("--force-device-scale-factor=1")  # Force 1:1 DPI

        if keep_temp:
            temp_fd, screenshot_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file.close()
            screenshot_path = temp_file.name

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            print(f"Loading page {url}...")
            driver.get(url)

            print(f"Waiting {wait_time} seconds for page to render...")
            time.sleep(wait_time)

            if full_page and capture_height is None:
                total_height = driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
                total_height = min(total_height, max_height)
                driver.set_window_size(width, total_height)
                time.sleep(1)
            elif capture_height:
                total_height = min(capture_height, max_height)
                driver.set_window_size(width, total_height)
                time.sleep(1)

            print("Taking screenshot...")
            driver.save_screenshot(screenshot_path)
            driver.quit()

            print(f"Content captured and saved to: {screenshot_path}")
            with Image.open(screenshot_path) as test_img:
                img_width, img_height = test_img.size
                print(f"Captured image dimensions: {img_width}x{img_height}")

            return screenshot_path

        except Exception as e:
            try:
                os.unlink(screenshot_path)
            except:
                pass
            raise Exception(f"Failed to capture content: {e}")

    def send_image(self, image_path: str, x: int = 0, y: int = 0, scale: float = 1.0, fit: bool = False):
        """Send an image to the Pixelflut server in one shot, always matching server size."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"Image file is empty: {image_path}")

            print(f"Loading image: {image_path} ({file_size} bytes)")
            img = Image.open(image_path).convert('RGBA')
            print(f"Image loaded successfully: {img.width}x{img.height}, mode: {img.mode}")

            # Always resize to server size first if screen_size is known
            if self.screen_size:
                screen_width, screen_height = self.screen_size
                if img.width != screen_width or img.height != screen_height:
                    img = img.resize((screen_width, screen_height), Image.LANCZOS)
                    print(f"Resized image to server size: {screen_width}x{screen_height}")

            # Apply fit or scale only after initial resize
            if fit and self.screen_size:
                screen_width, screen_height = self.screen_size
                img_width, img_height = img.size

                width_ratio = screen_width / img_width
                height_ratio = screen_height / img_height

                if img_height > 3 * img_width:  # Tall image adjustment
                    fit_scale = max(width_ratio, height_ratio * 0.3)
                    print(f"Adjusting scale for tall image: {fit_scale:.2f}")
                else:
                    fit_scale = min(width_ratio, height_ratio)

                new_width = int(img_width * fit_scale)
                new_height = int(img_height * fit_scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                x = (screen_width - new_width) // 2
                y = (screen_height - new_height) // 2
                print(f"Fitting image to screen: scale={fit_scale:.2f}, position=({x},{y})")
            elif scale != 1.0:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"Scaled image to: {new_width}x{new_height}")

            width, height = img.size
            pixels = list(img.getdata())
            pixels_by_thread = self._distribute_pixels(width, height, pixels)

            print(f"Sending image {image_path} ({width}x{height}) to {self.host}:{self.port}...")

            if not self.sockets:
                self.connect()

            threads = []
            for i in range(min(self.threads, len(pixels_by_thread))):
                t = threading.Thread(
                    target=self._send_pixels,
                    args=(i, pixels_by_thread[i], x, y, width)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            print("Image sent successfully!")

        except Exception as e:
            print(f"Error sending image: {e}")

    def send_content_with_scroll(self, url: str, scroll_interval: float, cycles: int, width: int, height: int, wait_time: int):
        """Scroll through any content (URL, HTML, or Markdown-rendered HTML) for a specified number of cycles."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for content capture. Install it with: pip install selenium webdriver-manager")

        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot scroll without server dimensions.")

        screen_width, screen_height = self.screen_size
        width = screen_width  # Match canvas exactly
        height = screen_height

        print(f"Starting scroll mode for {url} on {screen_width}x{screen_height} screen, interval={scroll_interval}s, cycles={cycles}")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(f"--window-size={width},{height}")
        chrome_options.add_argument("--force-device-scale-factor=1")  # Force 1:1 DPI

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        if not self.sockets:
            self.connect()

        try:
            cycle_count = 0
            while cycle_count < cycles:
                print(f"Cycle {cycle_count + 1}/{cycles}: Loading fresh content from {url}")
                driver.get(url)  # Refresh the page at the start of each cycle
                time.sleep(wait_time)

                total_height = driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
                print(f"Total content height: {total_height}px")

                scroll_step = screen_height  # Move by full screen height
                current_y = 0

                while current_y < total_height:
                    print(f"Cycle {cycle_count + 1}/{cycles}, scrolling to y={current_y}")
                    driver.execute_script(f"window.scrollTo(0, {current_y});")
                    time.sleep(0.5)  # Wait for scroll to settle

                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file.close()
                    screenshot_path = temp_file.name

                    driver.save_screenshot(screenshot_path)

                    # Ensure dimensions match canvas
                    with Image.open(screenshot_path) as img:
                        if img.size != (screen_width, screen_height):
                            img = img.resize((screen_width, screen_height), Image.LANCZOS)
                            img.save(screenshot_path)

                    self.send_image(screenshot_path, x=0, y=0, scale=1.0, fit=False)
                    os.unlink(screenshot_path)

                    current_y += scroll_step
                    time.sleep(scroll_interval)

                cycle_count += 1
                if cycle_count < cycles:
                    print(f"Completed cycle {cycle_count}, refreshing for next cycle")

            print(f"Completed {cycles} cycles, stopping execution")

        except Exception as e:
            print(f"Error in scrolling mode: {e}")
        finally:
            driver.quit()
            self.close()

    def send_website(self, url: str, scroll_interval: float = None, cycles: int = 1, wait_time: int = 3):
        """Render a website on the Pixelflut screen, with optional scrolling."""
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        print(f"Rendering website: {url} on {screen_width}x{screen_height} screen")

        if scroll_interval is None:
            screenshot_path = self.capture_website(
                url,
                width=screen_width,
                height=screen_height,
                wait_time=wait_time
            )
            self.send_image(screenshot_path, x=0, y=0, scale=1.0, fit=False)
            os.unlink(screenshot_path)
        else:
            self.send_content_with_scroll(
                url,
                scroll_interval=scroll_interval,
                cycles=cycles,
                width=screen_width,
                height=screen_height,
                wait_time=wait_time
            )

    def send_markdown(self, md_file: str, scroll_interval: float = None, cycles: int = 1, wait_time: int = 3):
        """Render a Markdown file on the Pixelflut screen, with optional scrolling."""
        if not os.path.exists(md_file):
            raise FileNotFoundError(f"Markdown file not found: {md_file}")

        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        print(f"Rendering Markdown file: {md_file} on {screen_width}x{screen_height} screen")

        # Convert Markdown to HTML
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html_content = markdown.markdown(md_content)

        # Create a simple HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 16px; }}
                pre, code {{ background: #f4f4f4; padding: 5px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Write to temporary HTML file
        temp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        temp_html.write(html_template.encode('utf-8'))
        temp_html.close()

        try:
            if scroll_interval is None:
                screenshot_path = self.capture_website(
                    f"file://{temp_html.name}",
                    width=screen_width,
                    height=screen_height,
                    wait_time=wait_time
                )
                self.send_image(screenshot_path, x=0, y=0, scale=1.0, fit=False)
                os.unlink(screenshot_path)
            else:
                self.send_content_with_scroll(
                    f"file://{temp_html.name}",
                    scroll_interval=scroll_interval,
                    cycles=cycles,
                    width=screen_width,
                    height=screen_height,
                    wait_time=wait_time
                )

        finally:
            os.unlink(temp_html.name)

    def send_html(self, html_file: str, scroll_interval: float = None, cycles: int = 1, wait_time: int = 3):
        """Render an HTML file on the Pixelflut screen, with optional scrolling."""
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file not found: {html_file}")

        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        print(f"Rendering HTML file: {html_file} on {screen_width}x{screen_height} screen")

        html_url = f"file://{os.path.abspath(html_file)}"

        if scroll_interval is None:
            screenshot_path = self.capture_website(
                html_url,
                width=screen_width,
                height=screen_height,
                wait_time=wait_time
            )
            self.send_image(screenshot_path, x=0, y=0, scale=1.0, fit=False)
            os.unlink(screenshot_path)
        else:
            self.send_content_with_scroll(
                html_url,
                scroll_interval=scroll_interval,
                cycles=cycles,
                width=screen_width,
                height=screen_height,
                wait_time=wait_time
            )

    def send_gif(self, gif_source: str, fps: float = None, loop: bool = False):
        """Render an animated GIF from a file or URL on the Pixelflut screen, fullscreen with optional looping."""
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        print(f"Rendering GIF from: {gif_source} on {screen_width}x{screen_height} screen, loop={loop}")

        # Handle URL or local file
        if gif_source.startswith(('http://', 'https://')):
            print(f"Downloading GIF from {gif_source}...")
            response = requests.get(gif_source, timeout=10)
            response.raise_for_status()
            gif_data = io.BytesIO(response.content)
            temp_gif = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
            temp_gif.write(gif_data.read())
            temp_gif.close()
            gif_path = temp_gif.name
        else:
            if not os.path.exists(gif_source):
                raise FileNotFoundError(f"GIF file not found: {gif_source}")
            gif_path = gif_source

        try:
            # Open the GIF
            with Image.open(gif_path) as img:
                if not img.is_animated:
                    raise ValueError(f"GIF from {gif_source} is not animated")

                # Extract frames and durations
                frames = []
                durations = []
                for frame in range(img.n_frames):
                    img.seek(frame)
                    # Convert to RGBA and resize to server size immediately
                    new_frame = Image.new("RGBA", img.size)
                    new_frame.paste(img)
                    new_frame = new_frame.resize((screen_width, screen_height), Image.LANCZOS)
                    frames.append(new_frame)
                    durations.append(img.info.get('duration', 100) / 1000.0)  # Convert ms to seconds

                if fps is not None:
                    frame_interval = 1.0 / fps
                else:
                    frame_interval = sum(durations) / len(durations)  # Average duration if no FPS

                if not self.sockets:
                    self.connect()

                try:
                    while True:
                        for i, frame in enumerate(frames):
                            # Save frame to temporary file
                            temp_frame = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_frame.close()
                            frame.save(temp_frame.name, 'PNG')

                            # Send frame, relying on send_image to enforce server size
                            print(f"Sending frame {i + 1}/{len(frames)}...")
                            self.send_image(temp_frame.name, x=0, y=0, scale=1, fit=False)
                            os.unlink(temp_frame.name)

                            # Throttle between frames
                            time.sleep(max(frame_interval, 0.1))  # Minimum 0.1s delay

                        if not loop:
                            break
                        print("Looping GIF...")
                        time.sleep(0.5)  # Delay between loops

                    print("GIF playback completed")

                except Exception as e:
                    print(f"Error in GIF playback: {e}")
                finally:
                    self.close()

        finally:
            if gif_source.startswith(('http://', 'https://')):
                os.unlink(gif_path)  # Clean up downloaded GIF

    def _send_pixels(self, thread_id: int, pixels: List[Tuple[int, int, Tuple[int, int, int]]], offset_x: int, offset_y: int, img_width: int):
        """Send a batch of pixels using a single thread and socket with throttling."""
        if not pixels:
            return

        try:
            sock = self.sockets[thread_id]
            commands = []
            for px, py, color in pixels:
                x, y = px + offset_x, py + offset_y
                r, g, b = color
                cmd = f"PX {x} {y} {r:02x}{g:02x}{b:02x}\n"
                commands.append(cmd)

            batch_size = 500  # Reduced from 100 to lower server load
            for i in range(0, len(commands), batch_size):
                try:
                    batch = commands[i:i+batch_size]
                    sock.sendall(''.join(batch).encode())
                    time.sleep(0.02)  # Increased delay to 20ms per batch
                except (ConnectionResetError, BrokenPipeError) as e:
                    print(f"Thread {thread_id} reconnecting after error: {e}")
                    try:
                        sock.close()
                    except:
                        pass
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((self.host, self.port))
                    self.sockets[thread_id] = sock
                    sock.sendall(''.join(batch).encode())
                    time.sleep(0.2)  # Increased delay after reconnect

        except Exception as e:
            print(f"Thread {thread_id} error: {e}")


    def _distribute_pixels(self, width: int, height: int, pixels: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, Tuple[int, int, int]]]]:
        """Distribute pixels among threads for efficient sending."""
        buckets = [[] for _ in range(self.threads)]

        for y in range(height):
            for x in range(width):
                pixel = pixels[y * width + x]
                if pixel[3] > 0:
                    if pixel[3] == 255:
                        rgb = pixel[:3]
                    else:
                        alpha = pixel[3] / 255.0
                        rgb = tuple(int(c * alpha) for c in pixel[:3])

                    bucket_idx = (x + y) % self.threads
                    buckets[bucket_idx].append((x, y, rgb))

        return buckets

    def clear_screen(self, color: Tuple[int, int, int] = (0, 0, 0)):
        """Clear the screen with a solid color."""
        if not self.screen_size:
            print("Warning: Screen size unknown, cannot clear screen")
            return

        width, height = self.screen_size
        r, g, b = color
        hex_color = f"{r:02x}{g:02x}{b:02x}"

        if not self.sockets:
            self.connect()

        print(f"Clearing screen ({width}x{height}) with color #{hex_color}...")

        band_height = height // self.threads
        threads = []

        for i in range(self.threads):
            start_y = i * band_height
            end_y = start_y + band_height if i < self.threads - 1 else height

            t = threading.Thread(
                target=self._clear_region,
                args=(i, 0, start_y, width, end_y, hex_color)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print("Screen cleared!")

    def _clear_region(self, thread_id: int, start_x: int, start_y: int, end_x: int, end_y: int, hex_color: str):
        """Clear a specific region of the screen with a solid color."""
        try:
            sock = self.sockets[thread_id]
            batch_size = 1000
            commands = []

            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    cmd = f"PX {x} {y} {hex_color}\n"
                    commands.append(cmd)
                    if len(commands) >= batch_size:
                        sock.sendall(''.join(commands).encode())
                        commands = []

            if commands:
                sock.sendall(''.join(commands).encode())

        except Exception as e:
            print(f"Clear region thread {thread_id} error: {e}")


def main():
    """Command line interface for the Pixelflut client."""
    parser = argparse.ArgumentParser(description='Pixelflut One-Shot Client')
    parser.add_argument('host', help='Pixelflut server hostname or IP')
    parser.add_argument('port', type=int, help='Pixelflut server port')

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('-i', '--image', help='Image file to send')
    source_group.add_argument('-u', '--url', help='Website URL to capture and send')
    source_group.add_argument('-m', '--markdown', help='Markdown file to render and send')
    source_group.add_argument('-l', '--html', help='HTML file to render and send')
    source_group.add_argument('-g', '--gif', help='Animated GIF file or URL to render')

    web_group = parser.add_argument_group('Web capture options')
    web_group.add_argument('--width', type=int, default=1280, help='Browser window width (default: 1280)')
    web_group.add_argument('--height', type=int, default=720, help='Browser window height (default: 720)')
    web_group.add_argument('--wait', type=int, default=3, help='Wait time for page loading in seconds (default: 3)')
    web_group.add_argument('--keep-temp', action='store_true', help='Keep temporary screenshot file (non-Markdown/HTML/GIF only)')
    web_group.add_argument('--full-page', action='store_true', help='Capture full page height (non-scrolling mode only)')
    web_group.add_argument('--capture-height', type=int, help='Specify custom capture height in pixels')
    web_group.add_argument('--max-height', type=int, default=3000, help='Maximum capture height in pixels (default: 3000)')

    parser.add_argument('-x', type=int, default=0, help='X position (default: 0)')
    parser.add_argument('-y', type=int, default=0, help='Y position (default: 0)')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scale factor (default: 1.0)')
    parser.add_argument('-f', '--fit', action='store_true', help='Fit image to screen (scale and center)')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads to use (default: 4)')
    parser.add_argument('-c', '--clear', action='store_true', help='Clear screen before sending')
    parser.add_argument('--scroll', type=float, help='Enable scrolling mode with interval in seconds (e.g., 5.0)')
    parser.add_argument('--cycles', type=int, default=1, help='Number of full scroll cycles (default: 1)')
    parser.add_argument('--fps', type=float, help='Frames per second for GIF playback (overrides GIF timing)')
    parser.add_argument('--loop', action='store_true', help='Loop GIF playback indefinitely')

    args = parser.parse_args()

    client = PixelflutClient(args.host, args.port, args.threads)
    temp_file = None

    try:
        client.connect()

        if args.clear:
            client.clear_screen()

        if args.markdown:
            client.send_markdown(
                args.markdown,
                scroll_interval=args.scroll,
                cycles=args.cycles,
                wait_time=args.wait
            )
        elif args.html:
            client.send_html(
                args.html,
                scroll_interval=args.scroll,
                cycles=args.cycles,
                wait_time=args.wait
            )
        elif args.url:
            client.send_website(
                args.url,
                scroll_interval=args.scroll,
                cycles=args.cycles,
                wait_time=args.wait
            )
        elif args.gif:
            client.send_gif(
                args.gif,
                fps=args.fps,
                loop=args.loop
            )
        else:
            image_path = args.image

        if not args.scroll and not args.markdown and not args.html and not args.url and not args.gif:
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)
            client.send_image(image_path, args.x, args.y, args.scale, args.fit)

    finally:
        if not args.scroll and not args.markdown and not args.html and not args.url and not args.gif:  # Scrolling/Markdown/HTML/GIF handle their own cleanup
            client.close()
            if temp_file and os.path.exists(temp_file) and not args.keep_temp:
                try:
                    os.unlink(temp_file)
                    print(f"Temporary file removed: {temp_file}")
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file: {e}")


if __name__ == '__main__':
    main()
