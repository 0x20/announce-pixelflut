#!/usr/bin/env python3
"""
Pixelflut One-Shot Client
A simple Python client for Pixelflut that sends images in one shot rather than continuous streaming.
Now with binary protocol support for significantly improved performance.
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
import requests
import struct
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

    def __init__(self, host: str, port: int, threads: int = 4, use_binary: bool = True, delta_mode: bool = False):
        """Initialize the Pixelflut client."""
        self.host = host
        self.port = port
        self.threads = threads
        self.sockets = []
        self.screen_size = None
        self.use_binary = use_binary
        self.binary_available = None
        self.lock = threading.Lock()
        self.delta_mode = delta_mode
        self.previous_image = None
        self.previous_image_hash = None

    def connect(self):
        """Connect to the Pixelflut server and create socket pool."""
        self.close()
        for i in range(self.threads):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
            sock.settimeout(10.0)  # Increased timeout
            try:
                sock.connect((self.host, self.port))
                self.sockets.append(sock)
                print(f"Thread {i} socket connected")
            except Exception as e:
                print(f"Thread {i} failed to connect: {e}")
                sock.close()

        if not self.sockets:
            raise ConnectionError("No sockets could be established")

        # Attempt to get server info with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                size_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                size_sock.settimeout(10.0)
                size_sock.connect((self.host, self.port))
                print(f"Attempt {attempt + 1}: Sending SIZE command")
                size_sock.sendall(b"SIZE\n")
                response = size_sock.recv(16).decode().strip()
                if response.startswith("SIZE "):
                    width, height = map(int, response[5:].split(" "))
                    self.screen_size = (width, height)
                    print(f"Server screen size: {width}x{height}")
                else:
                    print(f"Unexpected SIZE response: {response}")

                print(f"Attempt {attempt + 1}: Sending HELP command")
                size_sock.sendall(b"HELP\n")
                help_text = size_sock.recv(1024).decode().strip()
                self.binary_available = "PB" in help_text
                size_sock.close()

                if self.use_binary and self.binary_available:
                    print("Using binary protocol for faster pixel transfer")
                elif self.use_binary and not self.binary_available:
                    print("Warning: Binary protocol requested but not available. Falling back to text.")
                    self.use_binary = False
                else:
                    print("Using text protocol for pixel transfer")
                break  # Exit loop on success

            except socket.timeout as e:
                print(f"Attempt {attempt + 1} timed out: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                print(f"Warning: Failed to get server info after {max_retries} attempts: {e}")
                self.screen_size = (1920, 1080)  # Fallback size
                print(f"Using default screen size: {self.screen_size[0]}x{self.screen_size[1]}")
                self.binary_available = False
                self.use_binary = False
            except Exception as e:
                print(f"Warning: Could not get server info: {e}")
                self.screen_size = (1920, 1080)  # Fallback size
                print(f"Using default screen size: {self.screen_size[0]}x{self.screen_size[1]}")
                self.binary_available = False
                self.use_binary = False
                break

    def close(self):
        """Close all socket connections."""
        with self.lock:
            for sock in self.sockets:
                try:
                    sock.close()
                except:
                    pass
            self.sockets = []

    def capture_website(self, url: str, render_width: int = None, render_height: int = None,
                       wait_time: int = 3, keep_temp: bool = False, full_page: bool = False,
                       capture_height: int = None, max_height: int = 3000) -> str:
        """Capture a website or local HTML file as an image."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for website capture.")

        if self.screen_size:
            target_width, target_height = self.screen_size
            capture_width = render_width or target_width
            capture_height = render_height or target_height
            print(f"Matching final output to server screen size: {target_width}x{target_height}")
        else:
            capture_width = render_width or 1280
            capture_height = render_height or 720
            target_width, target_height = capture_width, capture_height

        print(f"Capturing content from: {url} at {capture_width}x{capture_height}")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(f"--window-size={capture_width},{capture_height}")
        chrome_options.add_argument("--force-device-scale-factor=1")

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=not keep_temp)
        screenshot_path = temp_file.name

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)
            print(f"Waiting {wait_time} seconds for page to render...")
            time.sleep(wait_time)

            if full_page and capture_height is None:
                total_height = driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
                total_height = min(total_height, max_height)
                driver.set_window_size(capture_width, total_height)
                time.sleep(1)
            elif capture_height:
                total_height = min(capture_height, max_height)
                driver.set_window_size(capture_width, total_height)
                time.sleep(1)

            driver.save_screenshot(screenshot_path)
            driver.quit()

            with Image.open(screenshot_path) as img:
                if self.screen_size and (img.width != target_width or img.height != target_height):
                    img = img.resize((target_width, target_height), Image.LANCZOS)
                    img.save(screenshot_path)
                    print(f"Resized image to server size: {target_width}x{target_height}")

            print(f"Content captured and saved to: {screenshot_path}")
            return screenshot_path

        except Exception as e:
            try:
                os.unlink(screenshot_path)
            except:
                pass
            raise Exception(f"Failed to capture content: {e}")

    def send_image(self, image_path: str, x: int = 0, y: int = 0, scale: float = 1.0, fit: bool = False):
        """Send an image to the Pixelflut server."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"Image file is empty: {image_path}")

            img = Image.open(image_path).convert('RGBA')
            print(f"Image loaded: {img.width}x{img.height}, mode: {img.mode}")

            if self.screen_size:
                screen_width, screen_height = self.screen_size
                if img.width != screen_width or img.height != screen_height:
                    img = img.resize((screen_width, screen_height), Image.LANCZOS)
                    print(f"Resized image to server size: {screen_width}x{screen_height}")

            if fit and self.screen_size:
                screen_width, screen_height = self.screen_size
                img_width, img_height = img.size
                width_ratio = screen_width / img_width
                height_ratio = screen_height / img_height
                fit_scale = min(width_ratio, height_ratio)
                new_width = int(img_width * fit_scale)
                new_height = int(img_height * fit_scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                x = (screen_width - new_width) // 2
                y = (screen_height - new_height) // 2
                print(f"Fitting image: scale={fit_scale:.2f}, position=({x},{y})")
            elif scale != 1.0:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"Scaled image to: {new_width}x{new_height}")

            width, height = img.size
            # Use numpy array for faster pixel access
            import numpy as np
            pixel_array = np.array(img)
            
            if self.delta_mode:
                pixels_by_thread = self._compute_image_delta(pixel_array, self.previous_image)
                # Store current image for next comparison
                self.previous_image = pixel_array.copy() if pixel_array is not None else None
            else:
                pixels_by_thread = self._distribute_pixels_optimized(width, height, pixel_array)

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
            raise

    def send_content_with_scroll(self, url: str, scroll_interval: float, cycles: int, width: int, height: int, wait_time: int):
        """Scroll through content for a specified number of cycles."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for content capture.")
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot scroll without server dimensions.")

        screen_width, screen_height = self.screen_size
        width = width or screen_width
        height = height or screen_height

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(f"--window-size={width},{height}")
        chrome_options.add_argument("--force-device-scale-factor=1")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        if not self.sockets:
            self.connect()

        try:
            cycle_count = 0
            while cycle_count < cycles:
                print(f"Cycle {cycle_count + 1}/{cycles}: Loading {url}")
                driver.get(url)
                time.sleep(wait_time)

                total_height = driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
                scroll_step = height
                current_y = 0

                while current_y < total_height:
                    driver.execute_script(f"window.scrollTo(0, {current_y});")
                    time.sleep(0.5)
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file.close()
                    screenshot_path = temp_file.name
                    driver.save_screenshot(screenshot_path)

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
                    print(f"Completed cycle {cycle_count}, refreshing")

            print(f"Completed {cycles} cycles")

        except Exception as e:
            print(f"Error in scrolling mode: {e}")
        finally:
            driver.quit()
            self.close()

    def send_website(self, url: str, scroll_interval: float = None, cycles: int = 1, wait_time: int = 3,
                    render_width: int = None, render_height: int = None):
        """Render a website on the Pixelflut screen."""
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        print(f"Rendering website: {url} on {screen_width}x{screen_height}")

        if scroll_interval is None:
            screenshot_path = self.capture_website(
                url,
                render_width=render_width,
                render_height=render_height,
                wait_time=wait_time
            )
            self.send_image(screenshot_path, x=0, y=0, scale=1.0, fit=False)
            os.unlink(screenshot_path)
        else:
            self.send_content_with_scroll(
                url,
                scroll_interval=scroll_interval,
                cycles=cycles,
                width=render_width or screen_width,
                height=render_height or screen_height,
                wait_time=wait_time
            )

    def send_markdown(self, md_file: str, scroll_interval: float = None, cycles: int = 1, wait_time: int = 3):
        """Render a Markdown file on the Pixelflut screen."""
        if not os.path.exists(md_file):
            raise FileNotFoundError(f"Markdown file not found: {md_file}")
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html_content = markdown.markdown(md_content)

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

        temp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        temp_html.write(html_template.encode('utf-8'))
        temp_html.close()

        try:
            if scroll_interval is None:
                screenshot_path = self.capture_website(
                    f"file://{temp_html.name}",
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
        """Render an HTML file on the Pixelflut screen."""
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file not found: {html_file}")
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        html_url = f"file://{os.path.abspath(html_file)}"

        if scroll_interval is None:
            screenshot_path = self.capture_website(
                html_url,
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

    def send_gif(self, gif_source: str, fps: float = None, loop: bool = False, gif_max_width: int = None, gif_max_height: int = None):
        """Render an animated GIF on the Pixelflut screen."""
        if not self.screen_size:
            raise ValueError("Screen size unknown; cannot render without server dimensions.")

        screen_width, screen_height = self.screen_size
        target_width = min(screen_width, gif_max_width) if gif_max_width else screen_width
        target_height = min(screen_height, gif_max_height) if gif_max_height else screen_height

        if gif_source.startswith(('http://', 'https://')):
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
            with Image.open(gif_path) as img:
                if not img.is_animated:
                    raise ValueError(f"GIF from {gif_source} is not animated")
                frames = []
                durations = []
                for frame in range(img.n_frames):
                    img.seek(frame)
                    new_frame = Image.new("RGBA", img.size)
                    new_frame.paste(img)
                    new_frame = new_frame.resize((target_width, target_height), Image.LANCZOS)
                    frames.append(new_frame)
                    durations.append(img.info.get('duration', 100) / 1000.0)

                frame_interval = 1.0 / fps if fps else sum(durations) / len(durations)
                if not self.sockets:
                    self.connect()

                try:
                    while True:
                        for i, frame in enumerate(frames):
                            temp_frame = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_frame.close()
                            frame.save(temp_frame.name, 'PNG')
                            self.send_image(temp_frame.name, x=0, y=0, scale=1.0, fit=False)
                            os.unlink(temp_frame.name)
                            time.sleep(max(frame_interval, 0.1))
                        if not loop:
                            break
                        time.sleep(0.5)
                finally:
                    self.close()

        finally:
            if gif_source.startswith(('http://', 'https://')):
                os.unlink(gif_path)

    def _distribute_pixels_optimized(self, width: int, height: int, pixel_array) -> List[List[Tuple[int, int, Tuple[int, int, int, int]]]]:
        """Distribute pixels among threads using horizontal bands for better cache locality."""
        import numpy as np
        buckets = [[] for _ in range(self.threads)]
        
        # Split image into horizontal bands
        band_height = height // self.threads
        remainder = height % self.threads
        
        current_y = 0
        for thread_idx in range(self.threads):
            # Add extra row to first 'remainder' threads
            thread_band_height = band_height + (1 if thread_idx < remainder else 0)
            end_y = min(current_y + thread_band_height, height)
            
            # Process this thread's band
            for y in range(current_y, end_y):
                for x in range(width):
                    if len(pixel_array.shape) == 3:  # RGB/RGBA
                        if pixel_array.shape[2] == 4:  # RGBA
                            pixel = tuple(pixel_array[y, x])
                            if pixel[3] > 0:  # Non-transparent
                                buckets[thread_idx].append((x, y, pixel))
                        else:  # RGB - add alpha
                            r, g, b = pixel_array[y, x]
                            pixel = (r, g, b, 255)
                            buckets[thread_idx].append((x, y, pixel))
                    else:  # Grayscale
                        val = pixel_array[y, x]
                        pixel = (val, val, val, 255)
                        buckets[thread_idx].append((x, y, pixel))
            
            current_y = end_y
        
        return buckets

    def clear_previous_image(self):
        """Clear the stored previous image to force a full send on next image."""
        self.previous_image = None
        print("Previous image cleared - next image will be sent in full")

    def _compute_image_delta(self, current_image, previous_image) -> List[List[Tuple[int, int, Tuple[int, int, int, int]]]]:
        """Compare current image with previous and return only changed pixels distributed across threads."""
        import numpy as np
        
        if previous_image is None or current_image.shape != previous_image.shape:
            # No previous image or different size - return all pixels
            return self._distribute_pixels_optimized(current_image.shape[1], current_image.shape[0], current_image)
        
        # Efficient delta calculation using numpy array operations
        height, width = current_image.shape[:2]
        buckets = [[] for _ in range(self.threads)]
        
        # Calculate difference and find non-zero (changed) pixels
        if len(current_image.shape) == 3:  # RGB/RGBA
            # Subtract arrays and find pixels where any channel changed
            diff = current_image != previous_image
            diff_mask = np.any(diff, axis=2)
        else:  # Grayscale
            diff_mask = current_image != previous_image
        
        # Get coordinates of all changed pixels at once
        y_coords, x_coords = np.where(diff_mask)
        
        # Extract pixel values for changed coordinates
        if len(current_image.shape) == 3:  # RGB/RGBA
            if current_image.shape[2] == 4:  # RGBA
                # Get RGBA values for changed pixels
                changed_pixels_rgba = current_image[y_coords, x_coords]
                # Filter out transparent pixels and create coordinate tuples
                alpha_mask = changed_pixels_rgba[:, 3] > 0
                valid_coords = np.where(alpha_mask)
                y_valid = y_coords[valid_coords]
                x_valid = x_coords[valid_coords]
                rgba_valid = changed_pixels_rgba[valid_coords]
                changed_pixels = [(int(x), int(y), tuple(rgba)) for x, y, rgba in zip(x_valid, y_valid, rgba_valid)]
            else:  # RGB - add alpha channel
                changed_pixels_rgb = current_image[y_coords, x_coords]
                changed_pixels = [(int(x), int(y), (*rgb, 255)) for x, y, rgb in zip(x_coords, y_coords, changed_pixels_rgb)]
        else:  # Grayscale
            changed_pixels_gray = current_image[y_coords, x_coords]
            changed_pixels = [(int(x), int(y), (int(val), int(val), int(val), 255)) for x, y, val in zip(x_coords, y_coords, changed_pixels_gray)]
        
        # Distribute changed pixels across threads
        for i, pixel_data in enumerate(changed_pixels):
            thread_idx = i % self.threads
            buckets[thread_idx].append(pixel_data)
        
        total_changed = len(changed_pixels)
        total_pixels = width * height
        print(f"Delta mode: {total_changed}/{total_pixels} pixels changed ({total_changed/total_pixels*100:.1f}%)")
        
        return buckets

    def _send_pixels(self, thread_id: int, pixels: List[Tuple[int, int, Tuple[int, int, int, int]]], offset_x: int, offset_y: int, img_width: int):
        """Send a batch of pixels using a single thread."""
        if not pixels:
            return
        with self.lock:
            if thread_id >= len(self.sockets):
                print(f"Thread {thread_id} has no socket assigned")
                return
            sock = self.sockets[thread_id]
        try:
            if self.use_binary and self.binary_available:
                self._send_pixels_binary(sock, pixels, offset_x, offset_y, thread_id)
            else:
                self._send_pixels_text(sock, pixels, offset_x, offset_y, thread_id)
        except Exception as e:
            print(f"Thread {thread_id} error: {e}")
            with self.lock:
                sock.close()
                new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)
                new_sock.settimeout(10.0)
                new_sock.connect((self.host, self.port))
                self.sockets[thread_id] = new_sock
            if self.use_binary and self.binary_available:
                self._send_pixels_binary(new_sock, pixels, offset_x, offset_y, thread_id)
            else:
                self._send_pixels_text(new_sock, pixels, offset_x, offset_y, thread_id)

    def _send_pixels_binary(self, sock, pixels, offset_x, offset_y, thread_id):
        """Send pixels using the binary protocol."""
        batch_size = 8000
        batches = [pixels[i:i + batch_size] for i in range(0, len(pixels), batch_size)]
        total_pixels = len(pixels)
        sent_pixels = 0
        for batch in batches:
            try:
                # Pre-allocate bytearray for entire batch (10 bytes per pixel: 'PB' + x + y + RGBA)
                commands = bytearray(len(batch) * 10)
                offset = 0
                for px, py, color in batch:
                    x, y = px + offset_x, py + offset_y
                    r, g, b, a = color
                    # Pack directly into pre-allocated buffer
                    commands[offset:offset+2] = b'PB'
                    commands[offset+2:offset+6] = struct.pack("<HH", x, y)
                    commands[offset+6:offset+10] = bytes([r, g, b, a])
                    offset += 10
                sock.sendall(commands)
                sent_pixels += len(batch)
                print(f"Thread {thread_id}: Sent {sent_pixels}/{total_pixels} pixels")
            except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
                print(f"Thread {thread_id} reconnecting after error: {e}")
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
                sock.settimeout(10.0)
                sock.connect((self.host, self.port))
                sock.sendall(commands)
                with self.lock:
                    self.sockets[thread_id] = sock

    def _send_pixels_text(self, sock, pixels, offset_x, offset_y, thread_id):
        """Send pixels using the text protocol."""
        commands = []
        for px, py, color in pixels:
            x, y = px + offset_x, py + offset_y
            r, g, b, a = color
            if a < 255:
                alpha = a / 255.0
                r = int(r * alpha)
                g = int(g * alpha)
                b = int(b * alpha)
            cmd = f"PX {x} {y} {r:02x}{g:02x}{b:02x}\n"
            commands.append(cmd)

        batch_size = 5000
        total_pixels = len(pixels)
        sent_pixels = 0
        for i in range(0, len(commands), batch_size):
            try:
                batch = commands[i:i + batch_size]
                sock.sendall(''.join(batch).encode())
                sent_pixels += len(batch)
                print(f"Thread {thread_id}: Sent {sent_pixels}/{total_pixels} pixels")
            except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
                print(f"Thread {thread_id} reconnecting after error: {e}")
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
                sock.settimeout(10.0)
                sock.connect((self.host, self.port))
                sock.sendall(''.join(batch).encode())
                with self.lock:
                    self.sockets[thread_id] = sock

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
        """Clear a region with a solid color."""
        with self.lock:
            if thread_id >= len(self.sockets):
                print(f"Thread {thread_id} has no socket assigned")
                return
            sock = self.sockets[thread_id]
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            if self.use_binary and self.binary_available:
                commands = bytearray()
                batch_size = 8000
                pixel_count = 0
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        cmd = bytearray(b'PB')
                        cmd.extend(struct.pack("<HH", x, y))
                        cmd.extend([r, g, b, 255])
                        commands.extend(cmd)
                        pixel_count += 1
                        if pixel_count >= batch_size:
                            sock.sendall(commands)
                            commands = bytearray()
                            pixel_count = 0
                if commands:
                    sock.sendall(commands)
            else:
                batch_size = 8000
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
    web_group.add_argument('--render-width', type=int, help='Website render width (before upscaling)')
    web_group.add_argument('--render-height', type=int, help='Website render height (before upscaling)')
    web_group.add_argument('--wait', type=int, default=3, help='Wait time for page loading in seconds (default: 3)')
    web_group.add_argument('--keep-temp', action='store_true', help='Keep temporary screenshot file')
    web_group.add_argument('--full-page', action='store_true', help='Capture full page height')
    web_group.add_argument('--capture-height', type=int, help='Specify custom capture height in pixels')
    web_group.add_argument('--max-height', type=int, default=3000, help='Maximum capture height in pixels (default: 3000)')

    parser.add_argument('-x', type=int, default=0, help='X position (default: 0)')
    parser.add_argument('-y', type=int, default=0, help='Y position (default: 0)')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scale factor (default: 1.0)')
    parser.add_argument('-f', '--fit', action='store_true', help='Fit image to screen')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads (default: 4)')
    parser.add_argument('-c', '--clear', action='store_true', help='Clear screen before sending')
    parser.add_argument('--scroll', type=float, help='Enable scrolling mode with interval in seconds')
    parser.add_argument('--cycles', type=int, default=1, help='Number of scroll cycles (default: 1)')  # Fixed typo from --cycle
    parser.add_argument('--fps', type=float, help='Frames per second for GIF playback')
    parser.add_argument('--loop', action='store_true', help='Loop GIF playback')
    parser.add_argument('--gif-max-width', type=int, help='Maximum width for GIF frames')
    parser.add_argument('--gif-max-height', type=int, help='Maximum height for GIF frames')
    parser.add_argument('--no-binary', action='store_true', help='Disable binary protocol')
    parser.add_argument('--delta', action='store_true', help='Enable delta mode - only send changed pixels')

    args = parser.parse_args()

    client = PixelflutClient(args.host, args.port, args.threads, use_binary=not args.no_binary, delta_mode=args.delta)
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
                wait_time=args.wait,
                render_width=args.render_width,
                render_height=args.render_height
            )
        elif args.gif:
            client.send_gif(
                args.gif,
                fps=args.fps,
                loop=args.loop,
                gif_max_width=args.gif_max_width,
                gif_max_height=args.gif_max_height
            )
        elif args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            client.send_image(args.image, args.x, args.y, args.scale, args.fit)

    except Exception as e:
        print(f"Main execution error: {e}")
    finally:
        if not args.scroll and not args.markdown and not args.html and not args.url and not args.gif:
            client.close()
            if temp_file and os.path.exists(temp_file) and not args.keep_temp:
                try:
                    os.unlink(temp_file)
                    print(f"Temporary file removed: {temp_file}")
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file: {e}")


if __name__ == "__main__":
    main()
