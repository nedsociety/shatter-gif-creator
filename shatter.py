# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pillow>=10.0.0",
#     "numpy>=1.21.0",
# ]
# ///

"""
Generate an explosive shattering effect GIF from a static input image.

High level:
- The image is partitioned into polygonal shards that fully cover the frame
  via a jittered grid subdivision.
- Each shard extracts its own RGBA subimage masked to its polygon.
- Shards are assigned outward velocities from the image center, plus gravity
  and air resistance, and an angular velocity.
- Frames are rendered by rotating and compositing shards, producing a GIF.

CLI usage:
  python shatter.py input.png output.gif --duration 1200 --shards 5 --fps 24
"""

import numpy as np
from PIL import Image, ImageDraw, ImageChops
import random
import math
from typing import List, Tuple
import argparse

class ImageShard:
    """Represents a single polygonal shard extracted from the source image.

    The shard maintains its own RGBA image, centroid, linear velocity, and
    angular velocity for animation.

    Args:
        vertices: Polygon vertices in source-image coordinates.
        original_image: The full source `PIL.Image` in RGBA.
        image_center: Center of the full image, used to derive explosion vector.
    """
    def __init__(self, vertices: List[Tuple[int, int]], original_image: Image.Image, image_center: Tuple[float, float]):
        self.original_vertices = vertices.copy()
        self.vertices = vertices.copy()
        self.original_image = original_image
        self.shard_image = self._extract_shard_image()
        self.center = self._calculate_center()
        self.original_center = self.center
        
        # Calculate explosion velocity based on distance and direction from image center
        dx = self.center[0] - image_center[0]
        dy = self.center[1] - image_center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Normalize direction vector
        if distance > 0:
            direction_x = dx / distance
            direction_y = dy / distance
        else:
            direction_x = random.uniform(-1, 1)
            direction_y = random.uniform(-1, 1)
        
        # Explosion force magnitude scales with distance from image center
        explosion_force = distance * 0.1
        
        # Add some randomness to the explosion
        force_variation = random.uniform(0.7, 1.3)
        explosion_force *= force_variation
        
        # Set initial velocity (explosion + small random component)
        self.velocity_x = direction_x * explosion_force + random.uniform(-1, 1)
        self.velocity_y = direction_y * explosion_force + random.uniform(-1, 1)
        
        # Rotation properties
        self.rotation = 0.0
        self.rotation_speed = random.uniform(-8, 8)  # Increased rotation speed
        
        # Gravity and air resistance (tuned for a snappy explosion feel)
        self.gravity = original_image.height / 50
        self.air_resistance = 0.98  # Gradually slow down explosion velocity
        
    def _calculate_center(self) -> Tuple[float, float]:
        x_coords = [v[0] for v in self.vertices]
        y_coords = [v[1] for v in self.vertices]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    def _extract_shard_image(self) -> Image.Image:
        """Extract and alpha-mask the shard's RGBA image within its bounding box."""
        # Get bounding box with proper int conversion
        x_coords = [int(v[0]) for v in self.original_vertices]
        y_coords = [int(v[1]) for v in self.original_vertices]
        
        min_x = max(0, min(x_coords))
        max_x = min(self.original_image.width, max(x_coords))
        min_y = max(0, min(y_coords))
        max_y = min(self.original_image.height, max(y_coords))
        
        if min_x >= max_x or min_y >= max_y:
            return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        
        # Create mask with integer dimensions
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        
        if width <= 0 or height <= 0:
            return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Adjust vertices relative to bounding box with int conversion
        adjusted_vertices = [(int(x - min_x), int(y - min_y)) for x, y in self.original_vertices]
        
        if len(adjusted_vertices) >= 3:
            mask_draw.polygon(adjusted_vertices, fill=255)
        
        # Extract region and apply mask
        region = self.original_image.crop((min_x, min_y, max_x, max_y))
        if region.mode != 'RGBA':
            region = region.convert('RGBA')
        
        # Apply transparency mask derived from polygon rasterization
        old_alpha = region.getchannel('A')
        new_alpha = ImageChops.multiply(old_alpha, mask)
        region.putalpha(new_alpha)

        return region
    
    def update(self):
        """Advance shard physics by one tick: drag, gravity, rotation, and position."""
        # Apply air resistance to explosion velocity (gradually slows down)
        self.velocity_x *= self.air_resistance
        self.velocity_y *= self.air_resistance
        
        # Apply gravity (always pulls down)
        self.velocity_y += self.gravity
        
        # Update rotation
        self.rotation += self.rotation_speed
        
        # Update center position
        self.center = (
            self.center[0] + self.velocity_x,
            self.center[1] + self.velocity_y
        )
    
    def get_rotated_image(self) -> Tuple[Image.Image, Tuple[int, int]]:
        """Return the rotated shard image and paste position (top-left coordinate)."""
        if self.shard_image.size[0] <= 1 or self.shard_image.size[1] <= 1:
            return self.shard_image, (int(self.center[0]), int(self.center[1]))
        
        # Rotate the shard image with float rotation
        rotated = self.shard_image.rotate(
            self.rotation, 
            expand=True, 
            fillcolor=(0, 0, 0, 0),
            resample=Image.BILINEAR
        )
        
        # Calculate position (top-left corner for pasting) with int conversion
        paste_x = int(self.center[0] - rotated.width // 2)
        paste_y = int(self.center[1] - rotated.height // 2)
        
        return rotated, (paste_x, paste_y)

def create_complete_shards(width: int, height: int, num_shards: int = 30) -> List[List[Tuple[int, int]]]:
    """Create polygon shards that completely cover the image without gaps.

    Strategy: build a rows×cols grid sized to the desired shard count, then
    split each cell into four triangles around a slightly jittered cell center.
    Jitter is clamped to prevent holes along cell boundaries.
    """
    shards = []
    
    # Calculate grid dimensions
    aspect_ratio = width / height
    rows = max(1, int(math.sqrt(num_shards / aspect_ratio)))
    cols = max(1, int(num_shards / rows))
    
    # Ensure we have enough cells
    while rows * cols < num_shards:
        if cols * aspect_ratio < rows:
            cols += 1
        else:
            rows += 1
    
    # Create exact cell dimensions (no gaps)
    cell_width = width / cols
    cell_height = height / rows
    
    print(f"Creating {rows}x{cols} grid ({rows*cols} cells) for {num_shards} shards")
    
    for row in range(rows):
        for col in range(cols):
            # Calculate exact cell boundaries
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            
            # Ensure we don't exceed image bounds
            x2 = min(x2, width)
            y2 = min(y2, height)
            
            # Skip cells that are too small
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            
            # Add some randomness for crack effect, but keep coverage
            jitter = min(int(cell_width * 0.2), int(cell_height * 0.2), 5)
            
            # Create center point with slight randomness
            center_x = (x1 + x2) // 2 + random.randint(-jitter, jitter)
            center_y = (y1 + y2) // 2 + random.randint(-jitter, jitter)
            
            # Clamp center to be within cell bounds
            center_x = max(x1 + 1, min(x2 - 1, center_x))
            center_y = max(y1 + 1, min(y2 - 1, center_y))
            
            # Create 4 triangles that exactly cover the cell rectangle
            triangles = [
                # Top triangle
                [(x1, y1), (x2, y1), (center_x, center_y)],
                # Right triangle  
                [(x2, y1), (x2, y2), (center_x, center_y)],
                # Bottom triangle
                [(x2, y2), (x1, y2), (center_x, center_y)],
                # Left triangle
                [(x1, y2), (x1, y1), (center_x, center_y)]
            ]
            
            # Add slight randomness to edges while maintaining coverage
            for triangle in triangles:
                randomized_triangle = []
                for i, (x, y) in enumerate(triangle):
                    if i < 2:  # Only randomize the first two points, keep center fixed
                        # Add small jitter but ensure we don't create gaps
                        edge_jitter = min(jitter // 2, 2)
                        new_x = x + random.randint(-edge_jitter, edge_jitter)
                        new_y = y + random.randint(-edge_jitter, edge_jitter)
                        
                        # Clamp to image bounds
                        new_x = max(0, min(width - 1, new_x))
                        new_y = max(0, min(height - 1, new_y))
                        
                        randomized_triangle.append((new_x, new_y))
                    else:
                        randomized_triangle.append((x, y))
                
                shards.append(randomized_triangle)
    
    print(f"Created {len(shards)} shards total")
    return shards

def create_shatter_gif(input_path: str, output_path: str, duration_ms: int = 2000, 
                      num_shards: int = 20, fps: int = 24):
    """Create a shattering glass effect GIF from a static image.

    Args:
        input_path: Path to input image. Will be converted to RGBA.
        output_path: Path to write the output GIF.
        duration_ms: Total animation duration in milliseconds.
        num_shards: Approximate number of base shards; actual count may be higher
            due to 4-triangle subdivision per grid cell.
        fps: Frames per second of the output GIF.
    """
    
    # Load image
    original_image = Image.open(input_path)
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    
    width, height = original_image.size
    image_center = (width / 2, height / 2)
    print(f"Processing image: {width}x{height}, center: {image_center}")
    
    # Create shards with complete coverage
    print("Creating shards...")
    shard_polygons = create_complete_shards(width, height, num_shards)
    
    shards = []
    for i, vertices in enumerate(shard_polygons):
        if len(vertices) >= 3:  # Valid polygon
            try:
                # Pass image center for explosion calculation
                shard = ImageShard(vertices, original_image, image_center)
                shards.append(shard)
            except Exception as e:
                print(f"Warning: Skipping shard {i}: {e}")
                continue
    
    print(f"Successfully created {len(shards)} explosive shards")
    
    # Animation parameters - ensure integers
    num_frames = int((duration_ms * fps) // 1000)
    pause_frames = int(num_frames * 0.15)  # ~15% pause at start
    
    frames = []
    
    print("Generating explosive frames...")
    for frame_num in range(num_frames):
        # Create new frame with transparent background
        frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        if frame_num < pause_frames:
            # Show original image during pause
            frame = original_image.copy()
        else:
            # Draw all shards at their current positions
            for shard in shards:
                try:
                    rotated_shard, position = shard.get_rotated_image()
                    
                    # Only paste if the shard is still near the viewport (with margin)
                    if (position[0] + rotated_shard.width > -50 and 
                        position[1] + rotated_shard.height > -50 and
                        position[0] < width + 50 and position[1] < height + 50):
                        
                        # Paste with alpha compositing
                        frame.paste(rotated_shard, position, rotated_shard)
                    
                    # Update shard for next frame (only after first breaking frame)
                    if frame_num > pause_frames:
                        shard.update()
                        
                except Exception as e:
                    # Skip problematic shards
                    continue
        
        frames.append(frame)
        
        if frame_num % 10 == 0:
            print(f"Frame {frame_num + 1}/{num_frames}")
    
    # Save GIF
    print("Saving explosive GIF...")
    frame_duration = int(1000 // fps)  # Ensure integer
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
        disposal=2  # Clear frame before drawing next one
    )
    
    print(f"💥 Explosive shatter GIF saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create an explosive shattering glass effect GIF')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output GIF path')
    parser.add_argument('--duration', type=int, default=1200, help='Duration in ms (default: 1200)')
    parser.add_argument('--shards', type=int, default=5, help='Number of base shards (default: 5)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    
    args = parser.parse_args()
    
    try:
        create_shatter_gif(
            args.input,
            args.output,
            duration_ms=args.duration,
            num_shards=args.shards,
            fps=args.fps
        )
        print("💥🎉 Explosive success!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
