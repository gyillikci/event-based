"""
Vibration Pattern Generator
Generate geometric shapes that vibrate at different frequencies for testing event cameras.

This GUI displays multiple shapes that oscillate at configurable frequencies,
designed to work with the Metavision vibration estimation sample.
"""

import cv2
import numpy as np
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class VibrationShape:
    """Configuration for a vibrating shape."""
    shape_type: str  # 'circle', 'rectangle', 'triangle', 'line'
    center_x: int
    center_y: int
    size: int
    frequency: float  # Hz
    amplitude: int
    color: Tuple[int, int, int]
    
    def get_offset(self, time_ms: float) -> Tuple[int, int]:
        """Calculate current position offset based on time and frequency."""
        # Oscillate vertically
        offset_y = int(self.amplitude * np.sin(2 * np.pi * self.frequency * time_ms / 1000.0))
        return 0, offset_y


def draw_shape(frame: np.ndarray, shape: VibrationShape, offset: Tuple[int, int]):
    """Draw a shape with given offset."""
    x = shape.center_x + offset[0]
    y = shape.center_y + offset[1]
    
    if shape.shape_type == 'circle':
        cv2.circle(frame, (x, y), shape.size, shape.color, -1)
        cv2.circle(frame, (x, y), shape.size, (255, 255, 255), 2)
        
    elif shape.shape_type == 'rectangle':
        half_size = shape.size // 2
        cv2.rectangle(frame, 
                     (x - half_size, y - half_size),
                     (x + half_size, y + half_size),
                     shape.color, -1)
        cv2.rectangle(frame, 
                     (x - half_size, y - half_size),
                     (x + half_size, y + half_size),
                     (255, 255, 255), 2)
        
    elif shape.shape_type == 'triangle':
        points = np.array([
            [x, y - shape.size],
            [x - shape.size, y + shape.size // 2],
            [x + shape.size, y + shape.size // 2]
        ])
        cv2.fillPoly(frame, [points], shape.color)
        cv2.polylines(frame, [points], True, (255, 255, 255), 2)
        
    elif shape.shape_type == 'line':
        cv2.line(frame, 
                (x - shape.size, y), 
                (x + shape.size, y),
                shape.color, shape.size // 3)


def add_text_overlay(frame: np.ndarray, shapes: List[VibrationShape]):
    """Add frequency information overlay."""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(frame, "Vibration Pattern Generator", (10, y_offset), 
                font, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    # Instructions
    cv2.putText(frame, "Point event camera at this screen", (10, y_offset), 
                font, 0.5, (200, 200, 200), 1)
    y_offset += 25
    
    cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", (10, y_offset), 
                font, 0.5, (200, 200, 200), 1)
    y_offset += 35
    
    # Shape frequencies
    for i, shape in enumerate(shapes):
        text = f"{shape.shape_type.capitalize()}: {shape.frequency:.1f} Hz"
        cv2.putText(frame, text, (10, y_offset), 
                   font, 0.5, shape.color, 2)
        y_offset += 25


def create_default_shapes(width: int, height: int) -> List[VibrationShape]:
    """Create default set of vibrating shapes."""
    spacing = width // 5
    y_center = height // 2
    
    shapes = [
        VibrationShape('circle', spacing, y_center, 40, 15.0, 30, (0, 255, 0)),
        VibrationShape('rectangle', spacing * 2, y_center, 50, 30.0, 35, (255, 0, 0)),
        VibrationShape('triangle', spacing * 3, y_center, 45, 50.0, 40, (0, 0, 255)),
        VibrationShape('circle', spacing * 4, y_center, 35, 75.0, 25, (255, 255, 0)),
        VibrationShape('rectangle', spacing // 2, y_center - 150, 40, 100.0, 30, (255, 0, 255)),
        VibrationShape('triangle', spacing * 2, y_center - 150, 40, 120.0, 35, (0, 255, 255)),
    ]
    
    return shapes


def main():
    parser = argparse.ArgumentParser(
        description='Generate vibrating geometric patterns for event camera testing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--width', type=int, default=1280,
                       help='Window width in pixels')
    parser.add_argument('--height', type=int, default=720,
                       help='Window height in pixels')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Run in fullscreen mode')
    parser.add_argument('--fps', type=int, default=60,
                       help='Target frame rate')
    parser.add_argument('--background', type=int, default=0,
                       help='Background brightness (0-255)')
    
    args = parser.parse_args()
    
    # Create window
    window_name = 'Vibration Pattern Generator'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, args.width, args.height)
    
    # Create shapes
    shapes = create_default_shapes(args.width, args.height)
    
    # Timing
    frame_time = 1.0 / args.fps
    start_time = time.time()
    frame_count = 0
    
    print(f"Vibration Pattern Generator")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Target FPS: {args.fps}")
    print(f"\nGenerating {len(shapes)} vibrating shapes:")
    for shape in shapes:
        print(f"  - {shape.shape_type.capitalize()}: {shape.frequency:.1f} Hz")
    print(f"\nPoint your event camera at this window.")
    print(f"Press 'q' to quit, 's' to save screenshot\n")
    
    try:
        while True:
            loop_start = time.time()
            
            # Create frame
            frame = np.ones((args.height, args.width, 3), dtype=np.uint8) * args.background
            
            # Calculate elapsed time
            elapsed_time_ms = (time.time() - start_time) * 1000
            
            # Draw all shapes
            for shape in shapes:
                offset = shape.get_offset(elapsed_time_ms)
                draw_shape(frame, shape, offset)
            
            # Add text overlay
            add_text_overlay(frame, shapes)
            
            # Add FPS counter
            if frame_count > 0:
                actual_fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {actual_fps:.1f}", 
                           (args.width - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                filename = f"vibration_pattern_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            
            # Frame rate control
            frame_count += 1
            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cv2.destroyAllWindows()
        actual_fps = frame_count / (time.time() - start_time)
        print(f"\nAverage FPS: {actual_fps:.2f}")
        print(f"Total frames: {frame_count}")


if __name__ == "__main__":
    main()
