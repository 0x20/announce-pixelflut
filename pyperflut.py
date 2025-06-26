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

    def __init__(self, host: str, port: int, threads: int = 4, use_binary: bool = True, delta_mode: bool = False, binary_fast: bool = False, fps_target: int = None, max_width: int = None, max_height: int = None):
        """Initialize the Pixelflut client."""
        self.host = host
        self.port = port
        self.threads = threads
        self.sockets = []
        self.screen_size = None
        self.use_binary = use_binary
        self.binary_fast = binary_fast
        self.binary_available = None
        self.fast_binary_available = None
        self.lock = threading.Lock()
        self.delta_mode = delta_mode
        self.delta_mode_enabled = delta_mode  # Track if delta mode is currently active
        self.previous_image = None
        self.previous_image_hash = None
        self._shutting_down = False
        
        # FPS optimization settings
        self.fps_target = fps_target
        self.max_width = max_width
        self.max_height = max_height
        self.target_frame_time = 1.0 / fps_target if fps_target else None
        self.performance_history = []
        
        # Memory pool for batch buffers to eliminate GC pressure
        self._buffer_pools = {
            'binary': {},      # thread_id -> list of bytearrays
            'binary_fast': {}, # thread_id -> list of bytearrays  
            'text': {}         # thread_id -> list of strings
        }
        self._max_pool_size = 8  # Keep up to 8 buffers per thread per protocol
        
        # Lock-free socket management using thread-local storage
        self._thread_local = threading.local()
        self._socket_valid = [False] * threads  # Atomic flags for socket validity

    def connect(self):
        """Connect to the Pixelflut server and create socket pool."""
        # Close existing connections without setting shutdown flag
        with self.lock:
            for sock in self.sockets:
                try:
                    sock.close()
                except:
                    pass
            self.sockets = []
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
                self.fast_binary_available = "PFST" in help_text
                size_sock.close()

                if self.binary_fast and self.fast_binary_available:
                    print("Using fast binary protocol (PFST batches) for maximum performance")
                elif self.binary_fast and not self.fast_binary_available:
                    print("Warning: Fast binary protocol requested but not supported. Falling back to standard binary.")
                    self.binary_fast = False
                    print("Using optimized standard binary protocol (PB commands)")
                elif self.use_binary and self.binary_available:
                    print("Using optimized standard binary protocol (PB commands)")
                elif self.use_binary and not self.binary_available:
                    print("Warning: Binary protocol requested but not available. Falling back to ASCII.")
                    self.use_binary = False
                else:
                    print("Using ASCII protocol for pixel transfer")
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
        self._shutting_down = True
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
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")  # Skip images for faster loading
        chrome_options.add_argument("--disable-javascript")  # Skip JS for stability
        chrome_options.add_argument(f"--window-size={capture_width},{capture_height}")
        chrome_options.add_argument("--force-device-scale-factor=1")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Add additional stability options
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=not keep_temp)
        screenshot_path = temp_file.name

        try:
            # Try to use ChromeDriverManager first
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception as e:
                print(f"ChromeDriverManager failed: {e}")
                print("Trying system chromedriver...")
                # Fallback to system chromedriver
                driver = webdriver.Chrome(options=chrome_options)
            
            print(f"Loading URL: {url}")
            driver.set_page_load_timeout(30)  # 30 second timeout
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

    def send_array_data(self, pixel_array, width: int, height: int, x: int = 0, y: int = 0):
        """Send numpy array data directly for maximum performance."""
        try:
            import time
            start_time = time.time()
            
            # Detailed timing for bottleneck analysis
            pixel_prep_start = time.time()
            
            if self.delta_mode_enabled:
                delta_start = time.time()
                pixels_by_thread = self._compute_image_delta(pixel_array, self.previous_image)
                # Store current image for next comparison (avoid expensive copy)
                self.previous_image = pixel_array  # Direct reference for next comparison
                delta_time = time.time() - delta_start
                total_pixels = sum(len(p) for p in pixels_by_thread)
                full_frame_size = width * height
                change_percentage = (total_pixels / full_frame_size) * 100
                print(f"Delta computation: {delta_time:.3f}s, {total_pixels} pixels to send ({change_percentage:.1f}% changed)")
                
                # Smart delta mode: temporarily disable if not beneficial (>80% change rate)
                if change_percentage > 80:
                    print("🔄 Temporarily disabling delta mode - too many changes")
                    self.delta_mode_enabled = False
                    self.previous_image = None  # Reset for next frame
            else:
                pixels_by_thread = self._distribute_pixels_optimized(width, height, pixel_array)
                total_pixels = sum(len(p) for p in pixels_by_thread)
                print(f"Full frame: {total_pixels} pixels to send")
                
                # Re-enable delta mode for next frame if originally requested
                if self.delta_mode and not self.delta_mode_enabled:
                    self.delta_mode_enabled = True
                    self.previous_image = pixel_array  # Set baseline for next comparison
            
            pixel_prep_time = time.time() - pixel_prep_start
            
            if not self.sockets:
                self.connect()

            thread_prep_start = time.time()
            network_start = time.time()
            threads = []
            try:
                for i in range(min(self.threads, len(pixels_by_thread))):
                    t = threading.Thread(
                        target=self._send_pixels,
                        args=(i, pixels_by_thread[i], x, y, width)
                    )
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()
                network_time = time.time() - network_start
                
                total_time = time.time() - start_time
                processing_time = total_time - network_time
                if total_pixels > 0:
                    print(f"Frame sent: {total_time:.3f}s total ({pixel_prep_time:.3f}s pixel prep + {network_time:.3f}s network), {total_pixels/total_time:.0f} pps")
                    
                    # Performance analysis and adaptive suggestions
                    if self.fps_target:
                        target_time = self.target_frame_time
                        current_fps = 1.0 / total_time
                        print(f"{self.fps_target} FPS target: {target_time*1000:.1f}ms per frame, current: {total_time*1000:.1f}ms ({current_fps:.1f} FPS)")
                        
                        if total_time <= target_time:
                            print(f"✅ Target FPS achieved! ({current_fps:.1f} FPS)")
                        else:
                            speedup_needed = total_time / target_time
                            print(f"⚠️  Need {speedup_needed:.1f}x speedup for {self.fps_target} FPS")
                    else:
                        # Show what FPS levels are achievable
                        current_fps = 1.0 / total_time
                        print(f"Current performance: {current_fps:.1f} FPS")
                        
                        # Calculate achievable resolutions for common FPS targets
                        for target_fps in [30, 60]:
                            target_time = 1.0 / target_fps
                            if total_time > target_time:
                                scale_needed = (target_time / total_time) ** 0.5
                                suggested_width = int(width * scale_needed)
                                suggested_height = int(height * scale_needed)
                                print(f"For {target_fps} FPS: try --fps-target {target_fps} (≈{suggested_width}x{suggested_height})")
                    
            except KeyboardInterrupt:
                print("\nInterrupted, shutting down threads...")
                self._shutting_down = True
                # Wait for threads to finish gracefully with timeout
                for t in threads:
                    t.join(timeout=1.0)
                raise

        except KeyboardInterrupt:
            self._shutting_down = True
            raise
        except Exception as e:
            print(f"Error sending array data: {e}")
            raise

    def send_image_data(self, img: Image.Image, x: int = 0, y: int = 0, scale: float = 1.0, fit: bool = False, skip_resize: bool = False):
        """Send PIL Image object directly without saving to disk."""
        try:
            if not skip_resize:
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
            
            print(f"Sending image data ({width}x{height}) to {self.host}:{self.port}...")
            if not self.sockets:
                self.connect()

            threads = []
            try:
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
            except KeyboardInterrupt:
                print("\nInterrupted, shutting down threads...")
                self._shutting_down = True
                # Wait for threads to finish gracefully with timeout
                for t in threads:
                    t.join(timeout=1.0)
                raise

        except KeyboardInterrupt:
            print("\nImage sending interrupted")
            self._shutting_down = True
            raise
        except Exception as e:
            print(f"Error sending image: {e}")
            import traceback
            traceback.print_exc()
            raise

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
            try:
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
            except KeyboardInterrupt:
                print("\nInterrupted, shutting down threads...")
                self._shutting_down = True
                # Wait for threads to finish gracefully with timeout
                for t in threads:
                    t.join(timeout=1.0)
                raise

        except KeyboardInterrupt:
            print("\nImage sending interrupted")
            self._shutting_down = True
            raise
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
        
        # Apply FPS-based resolution scaling
        if self.fps_target:
            target_width, target_height = self._calculate_fps_optimized_resolution(screen_width, screen_height)
            print(f"FPS optimization: targeting {self.fps_target} FPS, using {target_width}x{target_height} resolution")
        else:
            target_width = min(screen_width, gif_max_width) if gif_max_width else screen_width
            target_height = min(screen_height, gif_max_height) if gif_max_height else screen_height
            
        # Apply manual resolution limits
        if self.max_width:
            target_width = min(target_width, self.max_width)
        if self.max_height:
            target_height = min(target_height, self.max_height)

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
                print(f"Processing {img.n_frames} frames, resizing to {target_width}x{target_height}...")
                
                # Pre-process all frames for maximum performance
                import numpy as np
                frame_arrays = []
                
                for frame in range(img.n_frames):
                    img.seek(frame)
                    # Convert to RGBA and resize immediately for optimal memory usage
                    new_frame = Image.new("RGBA", img.size)
                    new_frame.paste(img)
                    # Pre-resize all frames to target size to avoid doing it repeatedly
                    new_frame = new_frame.resize((target_width, target_height), Image.LANCZOS)
                    
                    # Convert to numpy array once for faster processing during animation
                    frame_array = np.array(new_frame)
                    frame_arrays.append(frame_array)
                    
                    frames.append(new_frame)  # Keep PIL image for compatibility
                    durations.append(img.info.get('duration', 100) / 1000.0)
                
                print(f"All {len(frames)} frames pre-processed and cached in memory")
                
                # Store arrays for fast delta computation
                self._frame_arrays = frame_arrays
                frame_interval = 1.0 / fps if fps else sum(durations) / len(durations)
                print(f"Frames processed, starting animation with {frame_interval:.2f}s interval...")
                if not self.sockets:
                    self.connect()

                try:
                    cycle = 0
                    while True:
                        cycle += 1
                        for i, frame in enumerate(frames):
                            if self._shutting_down:
                                break
                            print(f"Sending frame {i+1}/{len(frames)}")
                            # Use pre-computed numpy arrays for maximum performance
                            frame_array = self._frame_arrays[i]
                            self.send_array_data(frame_array, target_width, target_height, x=0, y=0)
                            time.sleep(max(frame_interval, 0.1))
                        if not loop or self._shutting_down:
                            break
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    print("\nAnimation interrupted, cleaning up...")
                    self._shutting_down = True
                except Exception as e:
                    print(f"Animation error: {e}")
                    raise
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
    
    def _get_buffer(self, protocol: str, thread_id: int, required_size: int):
        """Get a buffer from the pool or create a new one."""
        if protocol not in self._buffer_pools:
            return bytearray(required_size) if protocol != 'text' else []
        
        if thread_id not in self._buffer_pools[protocol]:
            self._buffer_pools[protocol][thread_id] = []
        
        pool = self._buffer_pools[protocol][thread_id]
        
        # Look for a buffer that's large enough
        for i, buffer in enumerate(pool):
            if len(buffer) >= required_size:
                # Remove from pool and return
                return pool.pop(i)
        
        # No suitable buffer found, create new one
        if protocol == 'text':
            return []
        else:
            return bytearray(required_size)
    
    def _return_buffer(self, protocol: str, thread_id: int, buffer):
        """Return a buffer to the pool for reuse."""
        if protocol not in self._buffer_pools:
            return
        
        if thread_id not in self._buffer_pools[protocol]:
            self._buffer_pools[protocol][thread_id] = []
        
        pool = self._buffer_pools[protocol][thread_id]
        
        # Only keep buffer if pool isn't full
        if len(pool) < self._max_pool_size:
            # Clear buffer content for reuse
            if protocol == 'text':
                buffer.clear()
            else:
                # For bytearrays, we'll reuse as-is since we'll overwrite the content
                pass
            pool.append(buffer)

    def _compute_image_delta(self, current_image, previous_image) -> List[List[Tuple[int, int, Tuple[int, int, int, int]]]]:
        """Compare current image with previous and return only changed pixels distributed across threads."""
        import numpy as np
        
        if previous_image is None or current_image.shape != previous_image.shape:
            # No previous image or different size - return all pixels
            return self._distribute_pixels_optimized(current_image.shape[1], current_image.shape[0], current_image)
        
        # Efficient delta calculation using numpy array operations
        height, width = current_image.shape[:2]
        buckets = [[] for _ in range(self.threads)]
        
        # Calculate difference and find non-zero (changed) pixels using optimized operations
        if len(current_image.shape) == 3:  # RGB/RGBA
            # Use more efficient comparison for multi-channel images
            # This avoids creating the intermediate diff array
            if current_image.shape[2] == 4:  # RGBA - consider alpha channel in comparison
                # Use bitwise operations for faster comparison
                diff_mask = np.any(current_image != previous_image, axis=2)
            else:  # RGB
                diff_mask = np.any(current_image != previous_image, axis=2)
        else:  # Grayscale
            diff_mask = current_image != previous_image
        
        # Use nonzero() instead of where() for slightly better performance
        changed_indices = np.nonzero(diff_mask)
        y_coords, x_coords = changed_indices
        
        # Early exit if no changes
        if len(y_coords) == 0:
            print("Delta mode: 0/0 pixels changed (0.0% changed)")
            return buckets
        
        # Extract pixel values for changed coordinates using optimized vectorized operations
        if len(current_image.shape) == 3:  # RGB/RGBA
            if current_image.shape[2] == 4:  # RGBA
                # Get RGBA values for changed pixels in one operation
                changed_pixels_rgba = current_image[y_coords, x_coords]
                # Vectorized alpha filtering - only process non-transparent pixels
                alpha_nonzero = changed_pixels_rgba[:, 3] > 0
                
                if np.any(alpha_nonzero):
                    # Apply mask in a single vectorized operation
                    valid_coords = np.column_stack((x_coords[alpha_nonzero], y_coords[alpha_nonzero]))
                    valid_colors = changed_pixels_rgba[alpha_nonzero]
                    
                    # Combine coordinates and colors efficiently
                    # Use numpy's structured arrays for better performance
                    changed_pixels = [(int(x), int(y), (int(r), int(g), int(b), int(a))) 
                                    for (x, y), (r, g, b, a) in zip(valid_coords, valid_colors)]
                else:
                    changed_pixels = []
            else:  # RGB - add alpha channel efficiently
                changed_pixels_rgb = current_image[y_coords, x_coords]
                # Create coordinates-colors pairs directly without intermediate stacking
                changed_pixels = [(int(x), int(y), (int(r), int(g), int(b), 255)) 
                                for (x, y), (r, g, b) in zip(zip(x_coords, y_coords), changed_pixels_rgb)]
        else:  # Grayscale - optimize conversion
            changed_pixels_gray = current_image[y_coords, x_coords]
            # Direct conversion without intermediate arrays
            changed_pixels = [(int(x), int(y), (int(val), int(val), int(val), 255)) 
                            for (x, y), val in zip(zip(x_coords, y_coords), changed_pixels_gray)]
        
        # Distribute changed pixels across threads using modulo for even distribution
        for i, pixel_data in enumerate(changed_pixels):
            thread_idx = i % self.threads
            buckets[thread_idx].append(pixel_data)
        
        total_changed = len(changed_pixels)
        total_pixels = width * height
        print(f"Delta mode: {total_changed}/{total_pixels} pixels changed ({total_changed/total_pixels*100:.1f}%)")
        
        return buckets

    def _calculate_fps_optimized_resolution(self, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Calculate optimal resolution to hit target FPS."""
        if not self.fps_target:
            return screen_width, screen_height
            
        # Base performance model: ~1.16s for 921,600 pixels = 1.26ms per 1000 pixels
        # For 30 FPS we need 33.3ms total, so ~26,000 pixels max (accounting for overhead)
        target_frame_time = 1.0 / self.fps_target
        
        # Performance model based on recent measurements (more conservative)
        ms_per_1k_pixels = 0.9  # Based on ~60ms for 49k pixels = 1.2ms per 1k pixels
        overhead_ms = 20  # Lower overhead with fast protocol
        
        available_time_ms = (target_frame_time * 1000) - overhead_ms
        max_pixels = int(available_time_ms / ms_per_1k_pixels * 1000)
        
        # Add safety margin for target FPS
        max_pixels = int(max_pixels * 0.6)  # Use only 60% for safety margin
        
        # Ensure we have at least some reasonable minimum
        max_pixels = max(max_pixels, 20000)  # At least ~141x141
        
        # Calculate scaling factor
        current_pixels = screen_width * screen_height
        if current_pixels <= max_pixels:
            return screen_width, screen_height
            
        scale_factor = (max_pixels / current_pixels) ** 0.5
        
        # Round to reasonable values
        target_width = int(screen_width * scale_factor)
        target_height = int(screen_height * scale_factor)
        
        # Ensure even dimensions for better compatibility
        target_width = target_width - (target_width % 2)
        target_height = target_height - (target_height % 2)
        
        print(f"Resolution scaling: {screen_width}x{screen_height} ({current_pixels:,} pixels) → {target_width}x{target_height} ({target_width*target_height:,} pixels)")
        print(f"Estimated frame time: {(target_width*target_height/1000*ms_per_1k_pixels + overhead_ms):.1f}ms for {self.fps_target} FPS target")
        
        return target_width, target_height

    def _send_pixels(self, thread_id: int, pixels: List[Tuple[int, int, Tuple[int, int, int, int]]], offset_x: int, offset_y: int, img_width: int):
        """Send a batch of pixels using a single thread."""
        if not pixels or self._shutting_down:
            return
        
        # Safe socket access with bounds checking
        with self.lock:
            if self._shutting_down or thread_id >= len(self.sockets):
                return
            sock = self.sockets[thread_id]
        
        try:
            if self.binary_fast:
                self._send_pixels_binary_fast(sock, pixels, offset_x, offset_y, thread_id)
            elif self.use_binary and self.binary_available:
                self._send_pixels_binary(sock, pixels, offset_x, offset_y, thread_id)
            else:
                self._send_pixels_text(sock, pixels, offset_x, offset_y, thread_id)
        except Exception as e:
            if self._shutting_down:
                return
            print(f"Thread {thread_id} error: {e}")
            # Try to reconnect if not shutting down
            try:
                with self.lock:
                    if self._shutting_down or thread_id >= len(self.sockets):
                        return
                    
                    sock.close()
                    new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
                    new_sock.settimeout(10.0)
                    new_sock.connect((self.host, self.port))
                    self.sockets[thread_id] = new_sock
                    
                    # Retry sending
                    if self.use_binary and self.binary_available:
                        self._send_pixels_binary(new_sock, pixels, offset_x, offset_y, thread_id)
                    else:
                        self._send_pixels_text(new_sock, pixels, offset_x, offset_y, thread_id)
            except Exception as reconnect_error:
                if not self._shutting_down:
                    print(f"Thread {thread_id} reconnection failed: {reconnect_error}")

    def _send_pixels_binary(self, sock, pixels, offset_x, offset_y, thread_id):
        """Send pixels using the binary protocol."""
        # Increased batch size for better throughput while maintaining full Pixelflut compatibility
        batch_size = 32000  # 4x larger batches for better performance
        batches = [pixels[i:i + batch_size] for i in range(0, len(pixels), batch_size)]
        
        for batch in batches:
            try:
                # Get buffer from pool or create new one
                required_size = len(batch) * 10
                commands = self._get_buffer('binary', thread_id, required_size)
                if len(commands) < required_size:
                    commands.extend(b'\x00' * (required_size - len(commands)))
                
                offset = 0
                
                # Optimized batch preparation using direct memory operations
                for px, py, color in batch:
                    x, y = px + offset_x, py + offset_y
                    r, g, b, a = color
                    # Pack directly into pre-allocated buffer for maximum speed
                    commands[offset:offset+2] = b'PB'  # Standard Pixelflut binary command
                    commands[offset+2:offset+6] = struct.pack("<HH", x, y)  # Little-endian coordinates
                    commands[offset+6:offset+10] = bytes([r, g, b, a])  # RGBA values
                    offset += 10
                
                # Send entire batch at once for better network efficiency
                sock.sendall(commands[:required_size])
                
                # Return buffer to pool for reuse
                self._return_buffer('binary', thread_id, commands)
                
            except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
                if self._shutting_down:
                    return
                print(f"Thread {thread_id} reconnecting after error: {e}")
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
                sock.settimeout(10.0)
                sock.connect((self.host, self.port))
                with self.lock:
                    if thread_id < len(self.sockets):
                        self.sockets[thread_id] = sock
                # Retry the batch
                sock.sendall(commands)
            except Exception as e:
                if not self._shutting_down:
                    print(f"Thread {thread_id} send error: {e}")
                raise

    def _send_pixels_binary_fast(self, sock, pixels, offset_x, offset_y, thread_id):
        """Send pixels using experimental fast binary protocol with optimized format."""
        # Even larger batches for the experimental fast protocol
        batch_size = 64000  # 8x larger batches for maximum throughput
        batches = [pixels[i:i + batch_size] for i in range(0, len(pixels), batch_size)]
        
        for batch in batches:
            try:
                # Experimental format: 4-byte header + batch of 8-byte pixels (x,y,rgba)
                # Header: "PFST" (4 bytes) + count (4 bytes little-endian)
                count = len(batch)
                required_size = 8 + count * 8  # Header + 8 bytes per pixel
                
                # Get buffer from pool or create new one
                commands = self._get_buffer('binary_fast', thread_id, required_size)
                if len(commands) < required_size:
                    commands.extend(b'\x00' * (required_size - len(commands)))
                
                # Write header
                commands[0:4] = b'PFST'  # Pixelflut Fast protocol identifier
                commands[4:8] = struct.pack("<I", count)  # Pixel count
                
                # Write pixel data (8 bytes per pixel: x,y,rgba)
                offset = 8
                for px, py, color in batch:
                    x, y = px + offset_x, py + offset_y
                    r, g, b, a = color
                    # Pack as: x(2) + y(2) + r(1) + g(1) + b(1) + a(1)
                    commands[offset:offset+8] = struct.pack("<HHBBBB", x, y, r, g, b, a)
                    offset += 8
                
                # Send entire batch with header
                sock.sendall(commands[:required_size])
                
                # Return buffer to pool for reuse
                self._return_buffer('binary_fast', thread_id, commands)
                
            except Exception as e:
                if not self._shutting_down:
                    print(f"Thread {thread_id} fast binary send error: {e}")
                raise

    def _send_pixels_text(self, sock, pixels, offset_x, offset_y, thread_id):
        """Send pixels using the text protocol."""
        # Get buffer from pool for command accumulation
        commands = self._get_buffer('text', thread_id, len(pixels))
        
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

        # Increased batch size for better performance while maintaining full compatibility  
        batch_size = 20000  # 4x larger batches for ASCII protocol
        for i in range(0, len(commands), batch_size):
            try:
                batch = commands[i:i + batch_size]
                # Send entire batch as one network operation for efficiency
                sock.sendall(''.join(batch).encode())
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
        
        # Return buffer to pool for reuse
        self._return_buffer('text', thread_id, commands)
    
    def _get_thread_socket(self, thread_id: int):
        """Get socket for current thread using lock-free access."""
        # Check if we have a thread-local socket
        if hasattr(self._thread_local, 'socket') and self._socket_valid[thread_id]:
            return self._thread_local.socket
        
        # Need to get/create socket - this is the only place we need a lock
        with self.lock:
            if self._shutting_down or thread_id >= len(self.sockets):
                return None
            
            # Store socket in thread-local storage for lock-free future access
            self._thread_local.socket = self.sockets[thread_id]
            self._socket_valid[thread_id] = True
            return self._thread_local.socket
    
    def _invalidate_thread_socket(self, thread_id: int):
        """Mark thread socket as invalid (called after errors)."""
        self._socket_valid[thread_id] = False
        if hasattr(self._thread_local, 'socket'):
            delattr(self._thread_local, 'socket')
    
    def _replace_thread_socket(self, thread_id: int, new_socket):
        """Replace socket for a specific thread."""
        with self.lock:
            if thread_id < len(self.sockets):
                self.sockets[thread_id] = new_socket
        
        # Update thread-local storage
        self._thread_local.socket = new_socket
        self._socket_valid[thread_id] = True

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
    parser.add_argument('--no-binary', action='store_true', help='Disable binary protocol (use ASCII)')
    parser.add_argument('--binary-fast', action='store_true', help='Use optimized fast binary protocol (experimental)')
    parser.add_argument('--delta', action='store_true', help='Enable delta mode - only send changed pixels')
    parser.add_argument('--fps-target', type=int, help='Target FPS - automatically reduces resolution if needed')
    parser.add_argument('--fps-max-width', type=int, help='Maximum width for FPS optimization')
    parser.add_argument('--fps-max-height', type=int, help='Maximum height for FPS optimization')

    args = parser.parse_args()

    client = PixelflutClient(
        args.host, 
        args.port, 
        args.threads, 
        use_binary=not args.no_binary, 
        delta_mode=args.delta, 
        binary_fast=args.binary_fast,
        fps_target=args.fps_target,
        max_width=args.fps_max_width,
        max_height=args.fps_max_height
    )
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
