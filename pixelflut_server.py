#!/usr/bin/env python3
"""
High-Performance Dummy Pixelflut Server
A testing server that implements both ASCII and binary Pixelflut protocols.
"""

import socket
import threading
import argparse
import time
import struct
import sys
from collections import defaultdict
from typing import Dict, Tuple, Optional
import numpy as np

# Optional pygame for visual display
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class PixelflutServer:
    """High-performance dummy Pixelflut server for testing clients."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 1234, 
                 width: int = 1280, height: int = 720, max_connections: int = 100,
                 show_display: bool = False, scale_display: float = 1.0):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.max_connections = max_connections
        self.show_display = show_display
        self.scale_display = scale_display
        
        # Performance tracking
        self.stats = {
            'connections': 0,
            'total_pixels': 0,
            'ascii_pixels': 0,
            'binary_pixels': 0,
            'invalid_commands': 0,
            'bytes_received': 0,
            'start_time': time.time(),
            'regions_updated': 0,
            'full_updates': 0,
            'batch_operations': 0
        }
        
        # Virtual canvas (always enabled when display is shown)
        self.canvas_enabled = True
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Display setup
        self.display_width = int(width * scale_display)
        self.display_height = int(height * scale_display)
        self.screen = None
        self.clock = None
        self.display_dirty = False
        self.display_lock = threading.Lock()
        
        # Performance optimizations for sparse updates
        self.dirty_regions = set()  # Track changed regions
        self.pixel_batch = []  # Batch pixels for efficient processing
        self.batch_lock = threading.Lock()
        self.region_size = 32  # 32x32 pixel regions for batching
        
        self.running = False
        self.server_socket = None
        self.stats_lock = threading.Lock()
        
    def start(self):
        """Start the server."""
        # Initialize display if requested
        if self.show_display:
            if not PYGAME_AVAILABLE:
                print("Warning: pygame not available, running without display")
                self.show_display = False
            else:
                self._init_display()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Large receive buffer for performance
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)
            self.running = True
            
            print(f"Pixelflut server started on {self.host}:{self.port}")
            print(f"Canvas size: {self.width}x{self.height}")
            if self.show_display:
                print(f"Display: {self.display_width}x{self.display_height} (scale: {self.scale_display:.2f})")
            print(f"Canvas storage: {'enabled' if self.canvas_enabled else 'disabled (performance mode)'}")
            print("Supported commands: SIZE, HELP, PX (ASCII), PB (binary)")
            print("Press Ctrl+C to stop")
            
            # Start stats reporting thread
            stats_thread = threading.Thread(target=self._stats_reporter, daemon=True)
            stats_thread.start()
            
            # Use different main loops depending on display mode
            if self.show_display:
                self._run_with_display()
            else:
                self._run_headless()
                    
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.stop()
    
    def _run_headless(self):
        """Run server without display (original behavior)."""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                # Set large buffers for client connections too
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
                
                with self.stats_lock:
                    self.stats['connections'] += 1
                
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
    
    def _run_with_display(self):
        """Run server with display updates in main thread."""
        # Set socket to non-blocking for display integration
        self.server_socket.settimeout(0.1)
        
        last_display_update = time.time()
        display_interval = 1.0 / 60  # 60 FPS for smoother animation
        
        while self.running:
            current_time = time.time()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
            
            # Accept new connections (non-blocking)
            try:
                client_socket, address = self.server_socket.accept()
                # Set large buffers for client connections too
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
                
                with self.stats_lock:
                    self.stats['connections'] += 1
                
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                pass  # No new connections, continue
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
            
            # Update display if needed
            if self.display_dirty and (current_time - last_display_update) >= display_interval:
                # Flush any pending pixel batches first
                self._flush_pixel_batch()
                
                with self.display_lock:
                    try:
                        # Optimized region-based display updates
                        if self.dirty_regions and len(self.dirty_regions) < 50:  # If few regions changed
                            self._update_dirty_regions()
                            with self.stats_lock:
                                self.stats['regions_updated'] += len(self.dirty_regions)
                        else:
                            # Full display update for major changes
                            self._update_full_display()
                            with self.stats_lock:
                                self.stats['full_updates'] += 1
                        
                        pygame.display.flip()
                        self.display_dirty = False
                        self.dirty_regions.clear()
                        last_display_update = current_time
                        
                    except Exception as e:
                        print(f"Display update error: {e}")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.show_display and PYGAME_AVAILABLE:
            pygame.quit()
        self._print_final_stats()
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle a client connection."""
        print(f"Client connected: {address}")
        
        try:
            buffer = bytearray()
            
            while self.running:
                try:
                    # Receive data in large chunks for performance
                    data = client_socket.recv(65536)
                    if not data:
                        break
                        
                    with self.stats_lock:
                        self.stats['bytes_received'] += len(data)
                    
                    buffer.extend(data)
                    
                    # Process commands from buffer
                    self._process_buffer(buffer, client_socket)
                    
                except socket.timeout:
                    continue
                except socket.error:
                    break
                    
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Client disconnected: {address}")
    
    def _process_buffer(self, buffer: bytearray, client_socket: socket.socket):
        """Process commands from the buffer."""
        i = 0
        while i < len(buffer):
            # Check for fast binary protocol (PFST)
            if i + 3 < len(buffer) and buffer[i:i+4] == b'PFST':
                if i + 8 <= len(buffer):  # Header available
                    # Parse PFST header: "PFST" + pixel count (4 bytes little-endian)
                    count = struct.unpack("<I", buffer[i+4:i+8])[0]
                    total_size = 8 + count * 8  # Header + pixels
                    
                    if i + total_size <= len(buffer):  # Full batch available
                        # Process entire batch of pixels
                        pixel_offset = i + 8
                        for pixel_idx in range(count):
                            pixel_start = pixel_offset + pixel_idx * 8
                            # Parse pixel: x(2) + y(2) + r(1) + g(1) + b(1) + a(1)
                            x, y, r, g, b, a = struct.unpack("<HHBBBB", buffer[pixel_start:pixel_start+8])
                            self._set_pixel_binary(x, y, r, g, b, a)
                        
                        i += total_size
                    else:
                        # Incomplete batch, wait for more data
                        break
                else:
                    # Incomplete header, wait for more data
                    break
            
            # Check for standard binary command (PB)
            elif i + 1 < len(buffer) and buffer[i:i+2] == b'PB':
                if i + 10 <= len(buffer):  # Full binary command available
                    # Parse binary command: PB + 2 bytes x + 2 bytes y + 4 bytes RGBA
                    x, y = struct.unpack("<HH", buffer[i+2:i+6])
                    r, g, b, a = buffer[i+6:i+10]
                    
                    self._set_pixel_binary(x, y, r, g, b, a)
                    i += 10
                else:
                    # Incomplete binary command, wait for more data
                    break
            
            # Check for ASCII commands
            elif i < len(buffer):
                # Look for newline to complete command
                newline_pos = buffer.find(b'\n', i)
                if newline_pos == -1:
                    # No complete command yet, wait for more data
                    break
                
                # Extract command
                command = buffer[i:newline_pos].decode('ascii', errors='ignore').strip()
                self._process_ascii_command(command, client_socket)
                i = newline_pos + 1
            else:
                break
        
        # Remove processed data from buffer
        if i > 0:
            del buffer[:i]
    
    def _process_ascii_command(self, command: str, client_socket: socket.socket):
        """Process an ASCII command."""
        try:
            parts = command.split()
            if not parts:
                return
            
            cmd = parts[0].upper()
            
            if cmd == "SIZE":
                response = f"SIZE {self.width} {self.height}\n"
                client_socket.send(response.encode())
                
            elif cmd == "HELP":
                help_text = (
                    "Available commands:\n"
                    "SIZE - Get canvas dimensions\n"
                    "HELP - Show this help\n"
                    "PX x y rrggbb - Set pixel (ASCII)\n"
                    "PB - Set pixel (binary: PB + 2 bytes x + 2 bytes y + 4 bytes RGBA)\n"
                    "PFST - Fast batch protocol (PFST + count + batch of 8-byte pixels)\n"
                )
                client_socket.send(help_text.encode())
                
            elif cmd == "PX" and len(parts) >= 4:
                # Parse PX x y rrggbb[aa]
                x = int(parts[1])
                y = int(parts[2])
                color_hex = parts[3]
                
                # Parse hex color
                if len(color_hex) == 6:  # RGB
                    r = int(color_hex[0:2], 16)
                    g = int(color_hex[2:4], 16)
                    b = int(color_hex[4:6], 16)
                    a = 255
                elif len(color_hex) == 8:  # RGBA
                    r = int(color_hex[0:2], 16)
                    g = int(color_hex[2:4], 16)
                    b = int(color_hex[4:6], 16)
                    a = int(color_hex[6:8], 16)
                else:
                    with self.stats_lock:
                        self.stats['invalid_commands'] += 1
                    return
                
                self._set_pixel_ascii(x, y, r, g, b, a)
                
            else:
                with self.stats_lock:
                    self.stats['invalid_commands'] += 1
                    
        except (ValueError, IndexError):
            with self.stats_lock:
                self.stats['invalid_commands'] += 1
    
    def _set_pixel_ascii(self, x: int, y: int, r: int, g: int, b: int, a: int):
        """Set a pixel from ASCII command."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.canvas_enabled:
                # Batch pixel for efficient processing
                with self.batch_lock:
                    self.pixel_batch.append((x, y, r, g, b, a))
                    
                    # Process batch when it gets large enough
                    if len(self.pixel_batch) >= 1000:
                        self._process_pixel_batch()
            
            with self.stats_lock:
                self.stats['total_pixels'] += 1
                self.stats['ascii_pixels'] += 1
    
    def _set_pixel_binary(self, x: int, y: int, r: int, g: int, b: int, a: int):
        """Set a pixel from binary command."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.canvas_enabled:
                # Batch pixel for efficient processing
                with self.batch_lock:
                    self.pixel_batch.append((x, y, r, g, b, a))
                    
                    # Process batch when it gets large enough
                    if len(self.pixel_batch) >= 1000:
                        self._process_pixel_batch()
            
            with self.stats_lock:
                self.stats['total_pixels'] += 1
                self.stats['binary_pixels'] += 1
    
    def _process_pixel_batch(self):
        """Process accumulated pixels efficiently using vectorized operations."""
        if not self.pixel_batch:
            return
            
        batch = self.pixel_batch.copy()
        self.pixel_batch.clear()
        
        if not batch:
            return
            
        # Convert to numpy arrays for vectorized processing
        coords = np.array([(x, y) for x, y, r, g, b, a in batch])
        colors = np.array([(r, g, b, a) for x, y, r, g, b, a in batch])
        
        # Separate coordinates
        xs, ys = coords[:, 0], coords[:, 1]
        rs, gs, bs, alphas = colors[:, 0], colors[:, 1], colors[:, 2], colors[:, 3]
        
        # Apply alpha blending vectorized
        alpha_mask = alphas < 255
        if np.any(alpha_mask):
            # Get current pixel values for blending
            current_pixels = self.canvas[ys[alpha_mask], xs[alpha_mask]]
            alpha_values = alphas[alpha_mask] / 255.0
            
            # Vectorized alpha blending
            blended_r = (current_pixels[:, 0] * (1 - alpha_values) + rs[alpha_mask] * alpha_values).astype(np.uint8)
            blended_g = (current_pixels[:, 1] * (1 - alpha_values) + gs[alpha_mask] * alpha_values).astype(np.uint8)
            blended_b = (current_pixels[:, 2] * (1 - alpha_values) + bs[alpha_mask] * alpha_values).astype(np.uint8)
            
            self.canvas[ys[alpha_mask], xs[alpha_mask], 0] = blended_r
            self.canvas[ys[alpha_mask], xs[alpha_mask], 1] = blended_g
            self.canvas[ys[alpha_mask], xs[alpha_mask], 2] = blended_b
        
        # Set opaque pixels directly
        opaque_mask = alphas == 255
        if np.any(opaque_mask):
            self.canvas[ys[opaque_mask], xs[opaque_mask], 0] = rs[opaque_mask]
            self.canvas[ys[opaque_mask], xs[opaque_mask], 1] = gs[opaque_mask]
            self.canvas[ys[opaque_mask], xs[opaque_mask], 2] = bs[opaque_mask]
        
        # Track dirty regions for optimized display updates
        if self.show_display:
            self._mark_dirty_regions(xs, ys)
        
        # Update batch operation stats
        with self.stats_lock:
            self.stats['batch_operations'] += 1
    
    def _mark_dirty_regions(self, xs, ys):
        """Mark regions as dirty for optimized display updates."""
        # Group pixels into regions to minimize display updates
        for x, y in zip(xs, ys):
            region_x = x // self.region_size
            region_y = y // self.region_size
            self.dirty_regions.add((region_x, region_y))
        
        self.display_dirty = True
    
    def _update_dirty_regions(self):
        """Update only the dirty regions of the display for better performance."""
        for region_x, region_y in self.dirty_regions:
            # Calculate region bounds
            start_x = region_x * self.region_size
            start_y = region_y * self.region_size
            end_x = min(start_x + self.region_size, self.width)
            end_y = min(start_y + self.region_size, self.height)
            
            # Extract region from canvas
            region_canvas = self.canvas[start_y:end_y, start_x:end_x]
            
            # Convert to pygame surface
            if region_canvas.size > 0:
                region_surface = pygame.surfarray.make_surface(
                    np.transpose(region_canvas, (1, 0, 2))
                )
                
                # Scale if necessary
                if self.scale_display != 1.0:
                    scaled_width = int((end_x - start_x) * self.scale_display)
                    scaled_height = int((end_y - start_y) * self.scale_display)
                    region_surface = pygame.transform.scale(
                        region_surface, (scaled_width, scaled_height)
                    )
                    display_x = int(start_x * self.scale_display)
                    display_y = int(start_y * self.scale_display)
                else:
                    display_x, display_y = start_x, start_y
                
                # Blit region to screen
                self.screen.blit(region_surface, (display_x, display_y))
    
    def _update_full_display(self):
        """Update the entire display (fallback for major changes)."""
        if self.scale_display == 1.0:
            # Direct conversion for 1:1 scale
            canvas_surface = pygame.surfarray.make_surface(
                np.transpose(self.canvas, (1, 0, 2))
            )
            self.screen.blit(canvas_surface, (0, 0))
        else:
            # Scale the canvas for display
            canvas_surface = pygame.surfarray.make_surface(
                np.transpose(self.canvas, (1, 0, 2))
            )
            scaled_surface = pygame.transform.scale(
                canvas_surface, (self.display_width, self.display_height)
            )
            self.screen.blit(scaled_surface, (0, 0))
    
    def _flush_pixel_batch(self):
        """Force process any remaining pixels in batch."""
        with self.batch_lock:
            if self.pixel_batch:
                self._process_pixel_batch()
    
    def _stats_reporter(self):
        """Periodically report performance statistics."""
        last_pixels = 0
        last_time = time.time()
        
        while self.running:
            time.sleep(5)  # Report every 5 seconds
            
            current_time = time.time()
            with self.stats_lock:
                current_pixels = self.stats['total_pixels']
                elapsed = current_time - self.stats['start_time']
                recent_pixels = current_pixels - last_pixels
                recent_time = current_time - last_time
                
                if recent_time > 0:
                    recent_pps = recent_pixels / recent_time
                    total_pps = current_pixels / elapsed if elapsed > 0 else 0
                    
                    print(f"Stats: {current_pixels:,} pixels total "
                          f"({self.stats['ascii_pixels']:,} ASCII, {self.stats['binary_pixels']:,} binary) | "
                          f"{recent_pps:.0f} pps recent, {total_pps:.0f} pps average | "
                          f"{self.stats['connections']} connections | "
                          f"{self.stats['bytes_received'] / 1024 / 1024:.1f} MB received | "
                          f"Batches: {self.stats['batch_operations']} | "
                          f"Regions: {self.stats['regions_updated']} | "
                          f"Full updates: {self.stats['full_updates']}")
            
            last_pixels = current_pixels
            last_time = current_time
    
    def _print_final_stats(self):
        """Print final statistics when server stops."""
        elapsed = time.time() - self.stats['start_time']
        if elapsed > 0:
            print("\n=== Final Statistics ===")
            print(f"Runtime: {elapsed:.1f} seconds")
            print(f"Total pixels: {self.stats['total_pixels']:,}")
            print(f"  - ASCII pixels: {self.stats['ascii_pixels']:,}")
            print(f"  - Binary pixels: {self.stats['binary_pixels']:,}")
            print(f"Average pixels/second: {self.stats['total_pixels'] / elapsed:.0f}")
            print(f"Total connections: {self.stats['connections']}")
            print(f"Data received: {self.stats['bytes_received'] / 1024 / 1024:.1f} MB")
            print(f"Invalid commands: {self.stats['invalid_commands']}")
            
            if self.stats['binary_pixels'] > 0 and self.stats['ascii_pixels'] > 0:
                binary_ratio = self.stats['binary_pixels'] / self.stats['total_pixels']
                print(f"Binary protocol usage: {binary_ratio * 100:.1f}%")

    def _init_display(self):
        """Initialize pygame display."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption(f"Pixelflut Server - {self.width}x{self.height}")
        self.clock = pygame.time.Clock()
        
        # Fill with black initially
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
        print("Visual display initialized")



def main():
    """Command line interface for the Pixelflut server."""
    parser = argparse.ArgumentParser(description='High-Performance Dummy Pixelflut Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=1234, help='Port to bind to (default: 1234)')
    parser.add_argument('--width', type=int, default=1280, help='Canvas width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Canvas height (default: 720)')
    parser.add_argument('--max-connections', type=int, default=100, help='Maximum connections (default: 100)')
    parser.add_argument('--no-canvas', action='store_true', help='Disable canvas storage for maximum performance')
    parser.add_argument('--show-display', action='store_true', help='Show visual display window (requires pygame)')
    parser.add_argument('--scale-display', type=float, default=1.0, help='Scale factor for display window (default: 1.0)')
    
    args = parser.parse_args()
    
    # Check for pygame if display is requested
    if args.show_display and not PYGAME_AVAILABLE:
        print("Error: --show-display requires pygame. Install with: pip install pygame")
        sys.exit(1)
    
    # Auto-scale large displays
    if args.show_display and args.scale_display == 1.0:
        if args.width > 1920 or args.height > 1080:
            args.scale_display = min(1920 / args.width, 1080 / args.height)
            print(f"Auto-scaling large display: {args.scale_display:.2f}")
    
    server = PixelflutServer(
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        max_connections=args.max_connections,
        show_display=args.show_display,
        scale_display=args.scale_display
    )
    
    if args.no_canvas:
        server.canvas_enabled = False
        if args.show_display:
            print("Warning: --no-canvas conflicts with --show-display. Canvas will be enabled.")
            server.canvas_enabled = True
        else:
            print("Canvas storage disabled - running in pure performance mode")
    
    try:
        server.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()