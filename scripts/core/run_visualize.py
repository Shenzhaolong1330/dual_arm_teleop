import argparse
import gc
import logging
import time
from difflib import get_close_matches
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.constants import HF_LEROBOT_HOME


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_record_cfg_path() -> Path:
    return _default_scripts_dir() / "config" / "record_cfg.yaml"


def _load_record_cfg_yaml(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "visualize" not in cfg:
        raise ValueError(f"Visualize config must contain a top-level `visualize` mapping: {cfg_path}")
    return cfg


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # LeRobotDataset(episodes=[...]) filters the table and reindexes its rows.
        # Metadata's global dataset_from/to_index are not indices in that table.
        episode_ids = np.asarray(dataset.hf_dataset["episode_index"])
        self.frame_ids = np.flatnonzero(episode_ids == episode_index).tolist()
        if not self.frame_ids:
            raise ValueError(f"Episode {episode_index} has no frames in the loaded dataset")

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def resolve_dataset_root(repo_id: str, root: Path | None, *, allow_hub_download: bool = False) -> Path:
    path = Path(root).expanduser() if root is not None else Path(HF_LEROBOT_HOME) / repo_id
    if not (path / "meta/info.json").is_file() and not allow_hub_download:
        candidates = []
        if path.parent.is_dir():
            names = []
            try:
                for sibling in path.parent.iterdir():
                    try:
                        if (sibling / "meta/info.json").is_file():
                            names.append(sibling.name)
                    except OSError:
                        continue  # An inaccessible sibling is only a missing hint.
            except OSError:
                pass
            candidates = get_close_matches(path.name, names, n=3, cutoff=0.5)
        hint = ""
        if candidates:
            hint = "\nSimilar local datasets (exact names, including spaces): " + ", ".join(repr(n) for n in candidates)
        raise FileNotFoundError(
            f"Local dataset metadata not found: {path / 'meta/info.json'}{hint}\n"
            "Check visualize.dataset_name or --root. Hub download requires --allow-hub-download."
        )
    return path


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id
    features = dataset.meta.features
    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())
        
            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display task description
            if 'task' in batch:
                rr.log("task", rr.TextLog(batch['task'][i]))
                
            # display each dimension of action space (e.g. actuators command)
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{features['action']['names'][dim_idx]}", rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"observation.state/{features['observation.state']['names'][dim_idx]}", rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")

def main(argv: list[str] | None = None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=_default_record_cfg_path(),
        help="Path to record_cfg.yaml.",
    )
    pre_args, _ = pre_parser.parse_known_args(argv)
    cfg = _load_record_cfg_yaml(pre_args.config_path)

    parser = argparse.ArgumentParser(
        description="Visualize a LeRobot dataset episode with Rerun.",
        parents=[pre_parser],
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=cfg["visualize"]["dataset_name"],
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=cfg["visualize"]["episode_index"],
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Exact local dataset directory containing meta/info.json. Defaults to HF_LEROBOT_HOME/repo_id.",
    )
    parser.add_argument("--allow-hub-download", action="store_true",
                        help="Allow downloading a missing dataset from Hugging Face.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args(argv)
    kwargs = vars(args)
    kwargs.pop("config_path")
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    allow_hub_download = kwargs.pop("allow_hub_download")
    root = resolve_dataset_root(repo_id, root, allow_hub_download=allow_hub_download)

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
