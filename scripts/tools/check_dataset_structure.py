import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import sys

repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_uniform_frames(dataset: LeRobotDataset, episode_index: int, num_frames: int = 5):
    episode = dataset.meta.episodes[episode_index]
    start_idx = int(episode["dataset_from_index"])
    end_idx = int(episode["dataset_to_index"])

    total_frames = end_idx - start_idx

    if total_frames < num_frames:
        print(f"Warning: Episode {episode_index} has only {total_frames} frames, using all available frames")
        frame_indices = list(range(start_idx, end_idx))
    else:
        step = (total_frames - 1) / (num_frames - 1)
        frame_indices = [int(start_idx + i * step) for i in range(num_frames)]

    return frame_indices


def get_camera_key(dataset: LeRobotDataset):
    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError("No camera keys found in dataset")

    for key in camera_keys:
        if "wrist" in key or "hand" in key:
            return key

    return camera_keys[0]


def extract_frames_as_images(dataset: LeRobotDataset, frame_indices: list, camera_key: str):
    images = []
    for idx in frame_indices:
        frame_data = dataset[idx]
        if camera_key not in frame_data:
            raise KeyError(f"Camera key '{camera_key}' not found in frame {idx}")

        img_array = frame_data[camera_key]

        if isinstance(img_array, np.ndarray):
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img = Image.fromarray(img_array, mode='RGB')
            else:
                img = Image.fromarray(img_array)
        else:
            img = img_array

        images.append(img)

    return images


def concatenate_images_horizontally(images: list, task_name: str = ""):
    if not images:
        raise ValueError("No images to concatenate")

    widths = [img.width for img in images]
    heights = [img.height for img in images]

    total_width = sum(widths)
    max_height = max(heights)

    concatenated = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    x_offset = 0
    for img in images:
        y_offset = (max_height - img.height) // 2
        concatenated.paste(img, (x_offset, y_offset))
        x_offset += img.width

    return concatenated


def process_dataset(dataset_path: Path, num_frames: int = 5, episode_index: int = 0):
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_path.name}")
    print(f"{'='*60}")

    # For local datasets, construct repo_id from the relative path under lerobot directory
    # e.g., /path/to/lerobot/franka_dual_arm_cleaned/12ring_stick_E200_merged_cleaned
    # becomes repo_id = "franka_dual_arm_cleaned/12ring_stick_E200_merged_cleaned"
    # and root = /path/to/lerobot

    # Find the lerobot directory in the path
    path_parts = dataset_path.parts
    if "lerobot" in path_parts:
        lerobot_idx = path_parts.index("lerobot")
        # repo_id is everything after "lerobot" in the path
        repo_id = "/".join(path_parts[lerobot_idx + 1:])
        # root is the lerobot directory
        root = Path("/".join(path_parts[:lerobot_idx + 1]))
    else:
        # Fallback: use dataset name as repo_id and parent as root
        repo_id = dataset_path.name
        root = dataset_path.parent

    print(f"Loading dataset from: {dataset_path}")
    print(f"  repo_id: {repo_id}")
    print(f"  root: {root}")
    dataset = LeRobotDataset(repo_id, root=root, download_videos=True)

    print(f"Dataset info:")
    print(f"  - Total episodes: {dataset.meta.total_episodes}")
    print(f"  - Total frames: {len(dataset)}")
    print(f"  - FPS: {dataset.fps}")
    print(f"  - Camera keys: {dataset.meta.camera_keys}")

    camera_key = get_camera_key(dataset)
    print(f"  - Using camera: {camera_key}")

    frame_indices = extract_uniform_frames(dataset, episode_index, num_frames)
    print(f"\nExtracting {num_frames} frames from episode {episode_index}:")
    print(f"  - Frame indices: {frame_indices}")

    images = extract_frames_as_images(dataset, frame_indices, camera_key)
    print(f"  - Extracted {len(images)} images")

    concatenated = concatenate_images_horizontally(images, dataset_path.name)

    return concatenated, frame_indices


def main():
    parser = argparse.ArgumentParser(
        description="Extract time slices from LeRobot datasets and create concatenated images"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to the dataset directories"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames to extract from each episode (default: 5)"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to extract from (default: 0, first episode)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the concatenated images (default: current directory)"
    )

    args = parser.parse_args()

    output_dir = args.output_dir or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_path in args.datasets:
        if not dataset_path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            continue

        try:
            concatenated_img, frame_indices = process_dataset(
                dataset_path,
                args.num_frames,
                args.episode_index
            )

            output_filename = f"{dataset_path.name}_timeslice.png"
            output_path = output_dir / output_filename
            concatenated_img.save(output_path, quality=95)

            print(f"\n✓ Saved concatenated image to: {output_path}")
            print(f"  - Image size: {concatenated_img.width}x{concatenated_img.height}")

        except Exception as e:
            print(f"\n✗ Error processing dataset {dataset_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()