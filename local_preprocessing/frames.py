import cv2
from pathlib import Path

def extract_frames_every_n_seconds(
    video_file_path: Path,
    output_dir: Path,
    seconds_interval: float = 2.0
) -> None:
    """
    Extract one frame every N seconds from a video file.

    Parameters
    ----------
    video_file_path : Path
        Path to the input video file.
    output_dir : Path
        Directory where extracted frames will be saved.
    seconds_interval : float
        Time interval (in seconds) between saved frames.
        For example, 2.0 means one frame every 2 seconds.
    """

    # Create the output directory if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_file_path))
    if not cap.isOpened():
        print("Failed to open video:", video_file_path.resolve())
        return

    # Read FPS from video metadata
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    if fps_raw <= 0:
        print("Invalid FPS value:", fps_raw)
        cap.release()
        return

    # Convert time interval (seconds) to frame step
    step = int(round(fps_raw * seconds_interval))

    print("fps_raw:", fps_raw)
    print("seconds_interval:", seconds_interval)
    print("step_frames:", step)

    frame_count = 0  # Total frames read
    saved = 0        # Frames saved

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Save one frame every 'step' frames
        if frame_count % step == 0:
            frame_filename = output_dir / f"frame_{saved+1:03d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved += 1
        frame_count += 1

    cap.release()

    print("saved_frames:", saved)
    print("output_dir:", output_dir.resolve())


if __name__ == "__main__":
    # Resolve paths relative to this script
    here = Path(__file__).resolve().parent

    video_file = here / "video.mp4"
    output_dir = here / "frames"   # <- fixed folder name

    extract_frames_every_n_seconds(
        video_file_path=video_file,
        output_dir=output_dir,
        seconds_interval=2.0
    )