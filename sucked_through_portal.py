"""
Sucked Through the Portal - Editable Video Editor
=================================================

This script uses the supplied MP4 as its visual source and applies editable
video-processing effects. It does not recreate the original artwork from
scratch; the portal, logo, and landscape are already baked into the source.

Requirements:
    pip install opencv-python numpy

Run:
    python sucked_through_portal_editor.py

Put "Sucked Through the portal.mp4" in the same folder as this script,
or change INPUT_VIDEO below.

Edit the SETTINGS section to customize the result.
"""

from pathlib import Path
import cv2
import numpy as np


# ============================================================
# SETTINGS - CHANGE THESE VALUES TO EDIT YOUR VIDEO
# ============================================================

INPUT_VIDEO = "Sucked Through the portal.mp4"
OUTPUT_VIDEO = "Sucked_Through_the_Portal_EDITED.mp4"

# Time range from the original video, in seconds
START_TIME = 0.0
END_TIME = None                 # None = use the end of the source video

# Playback speed:
# 1.0 = original speed
# 0.5 = half speed
# 2.0 = double speed
PLAYBACK_SPEED = 1.0

# Output dimensions. None keeps the original dimensions.
OUTPUT_WIDTH = None
OUTPUT_HEIGHT = None

# Overall image adjustments
BRIGHTNESS = 0                  # -100 to +100
CONTRAST = 1.0                   # 1.0 = original
SATURATION = 1.0                 # 1.0 = original
GAMMA = 1.0                      # 1.0 = original

# Colour adjustment
HUE_SHIFT = 0                    # OpenCV hue units: approximately -90 to +90

# Glow effect
GLOW_ENABLED = False
GLOW_STRENGTH = 0.35             # 0.0 to 1.0
GLOW_BLUR_SIZE = 31              # Odd number recommended: 15, 31, 51, etc.
GLOW_THRESHOLD = 170             # 0 to 255; higher = only brightest areas glow

# Sharpening
SHARPEN_ENABLED = False
SHARPEN_STRENGTH = 0.5           # 0.0 to 2.0

# Optional vignette (darkens image edges)
VIGNETTE_ENABLED = False
VIGNETTE_STRENGTH = 0.25         # 0.0 to 1.0

# Optional zoom/crop
ZOOM = 1.0                       # 1.0 = original; 1.1 = 10% zoom in
PAN_X = 0.0                      # -1.0 left, 0 centre, +1 right
PAN_Y = 0.0                      # -1.0 up, 0 centre, +1 down

# Export settings
OUTPUT_FPS = None                # None = use source FPS
CODEC = "mp4v"


# ============================================================
# VIDEO EFFECTS
# ============================================================

def apply_colour_adjustments(frame):
    """Apply brightness, contrast, saturation, hue, and gamma."""
    result = frame.astype(np.float32)

    # Contrast and brightness
    result = (result - 127.5) * CONTRAST + 127.5 + BRIGHTNESS
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Saturation and hue
    if SATURATION != 1.0 or HUE_SHIFT != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= SATURATION
        hsv[:, :, 0] = (hsv[:, :, 0] + HUE_SHIFT) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gamma correction
    if GAMMA > 0 and GAMMA != 1.0:
        lookup = np.array([
            min(255, max(0, int(((i / 255.0) ** (1.0 / GAMMA)) * 255)))
            for i in range(256)
        ], dtype=np.uint8)
        result = cv2.LUT(result, lookup)

    return result


def apply_glow(frame):
    """Add a soft glow around the brightest parts of the image."""
    if not GLOW_ENABLED:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, GLOW_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    bright_areas = cv2.bitwise_and(frame, frame, mask=mask)

    blur_size = max(3, int(GLOW_BLUR_SIZE))
    if blur_size % 2 == 0:
        blur_size += 1

    glow = cv2.GaussianBlur(
        bright_areas, (blur_size, blur_size), 0
    )

    return cv2.addWeighted(
        frame, 1.0, glow, GLOW_STRENGTH, 0
    )


def apply_sharpening(frame):
    """Increase edge definition."""
    if not SHARPEN_ENABLED:
        return frame

    blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
    return cv2.addWeighted(
        frame, 1.0 + SHARPEN_STRENGTH,
        blurred, -SHARPEN_STRENGTH, 0
    )


def apply_vignette(frame):
    """Darken the edges while leaving the centre brighter."""
    if not VIGNETTE_ENABLED:
        return frame

    height, width = frame.shape[:2]
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    distance = xx**2 + yy**2
    mask = 1.0 - VIGNETTE_STRENGTH * np.clip(distance / 2.0, 0, 1)
    mask = np.clip(mask, 0, 1)

    return np.clip(
        frame.astype(np.float32) * mask[:, :, None], 0, 255
    ).astype(np.uint8)


def apply_zoom_and_pan(frame):
    """Zoom into the frame and shift the visible crop."""
    if ZOOM <= 1.0:
        return frame

    height, width = frame.shape[:2]
    crop_width = max(1, int(width / ZOOM))
    crop_height = max(1, int(height / ZOOM))

    max_x = width - crop_width
    max_y = height - crop_height

    x_start = int(max_x * (0.5 + 0.5 * PAN_X))
    y_start = int(max_y * (0.5 + 2.5 * PAN_Y))

    x_start = max(0, min(max_x, x_start))
    y_start = max(0, min(max_y, y_start))

    cropped = frame[
        y_start:y_start + crop_height,
        x_start:x_start + crop_width
    ]

    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)


def process_frame(frame):
    """Apply all enabled effects in order."""
    frame = apply_zoom_and_pan(frame)
    frame = apply_colour_adjustments(frame)
    frame = apply_glow(frame)
    frame = apply_sharpening(frame)
    frame = apply_vignette(frame)

    if OUTPUT_WIDTH and OUTPUT_HEIGHT:
        frame = cv2.resize(
            frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=cv2.INTER_LANCZOS4
        )

    return frame


# ============================================================
# MAIN EXPORT
# ============================================================

def main():
    script_folder = Path(__file__).resolve().parent
    input_path = script_folder / INPUT_VIDEO
    output_path = script_folder / OUTPUT_VIDEO

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find input video: {input_path}\n"
            "Place the MP4 beside this Python script or update INPUT_VIDEO."
        )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the input video.")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / source_fps

    fps = OUTPUT_FPS if OUTPUT_FPS else source_fps
    end_time = duration if END_TIME is None else min(END_TIME, duration)

    if START_TIME < 0 or START_TIME >= end_time:
        raise ValueError("START_TIME must be before END_TIME.")

    if PLAYBACK_SPEED <= 0:
        raise ValueError("PLAYBACK_SPEED must be greater than zero.")

    output_width = OUTPUT_WIDTH or source_width
    output_height = OUTPUT_HEIGHT or source_height

    first_frame = max(0, int(START_TIME * source_fps))
    last_frame = min(total_frames, int(end_time * source_fps))
    selected_frames = max(0, last_frame - first_frame)
    output_frames = max(1, int(selected_frames / PLAYBACK_SPEED))
    output_duration = output_frames / fps

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps,
        (output_width, output_height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output MP4.")

    print("Processing video...")
    print(f"Source: {source_width} x {source_height} at {source_fps:.2f} FPS")
    print(f"Selected source time: {START_TIME:.2f} to {end_time:.2f} seconds")
    print(f"Output: {output_width} x {output_height} at {fps:.2f} FPS")
    print(f"Approximate output duration: {output_duration:.2f} seconds")

    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frames = []

    while len(frames) < selected_frames:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    capture.release()

    if not frames:
        writer.release()
        raise RuntimeError("No frames were read from the selected time range.")

    # Select source frames according to playback speed.
    for output_index in range(output_frames):
        source_index = min(
            len(frames) - 1,
            int(output_index * PLAYBACK_SPEED)
        )

        edited = process_frame(frames[source_index])
        writer.write(edited)

        if output_index % max(1, int(fps)) == 0:
            percent = 100 * (output_index + 1) / output_frames
            print(f"Progress: {percent:.1f}%")

    writer.release()
    print(f"\nFinished! Saved video to:\n{output_path}")


if __name__ == "__main__":
    main()
