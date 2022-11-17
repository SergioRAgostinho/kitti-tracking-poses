from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import os
from os.path import expanduser
from typing import Final

import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import line_aa
from tqdm import tqdm

from kitti import KITTITracking


_BBOX_3D: Final = 0.5 * np.array(
    [
        [-1, 0, -1],
        [-1, 0, 1],
        [-1, -2, 1],
        [-1, -2, -1],
        [1, -2, -1],
        [1, 0, -1],
        [1, 0, 1],
        [1, -2, 1],
    ]
)

_BBOX_EDGES: Final = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 4],
    ]
)


def rot_y(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_1d(theta)
    n = len(theta)
    return np.stack(
        (
            np.stack([np.cos(theta), np.zeros(n), np.sin(theta)], axis=-1),
            np.stack([np.zeros(n), np.ones(n), np.zeros(n)], axis=-1),
            np.stack([-np.sin(theta), np.zeros(n), np.cos(theta)], axis=-1),
        ),
        axis=-2,
    )


def plot_3d_bbox(
    rgb: np.array, labels: pd.DataFrame, calibration: dict[str, np.ndarray]
) -> Image.Image:

    sizes = labels[["l", "w", "h"]].values
    obj_3d_bbox = sizes[:, None] * _BBOX_3D

    # move them in camera frame of reference frame
    R = rot_y(labels.yaw.values)
    t = labels[["X", "Y", "Z"]].values
    pts_3d = obj_3d_bbox @ R.transpose(0, 2, 1) + t[:, None]

    # transform to the image frame of reference
    pts_2d_c2 = (
        np.concatenate([pts_3d, np.ones((len(pts_3d), 8, 1))], axis=-1)
        @ calibration["P2"].T
    )

    # don't render cuboids behind the camera
    valid_cuboids = np.all(pts_2d_c2[..., -1] > 0, axis=-1)
    pts_2d_c2 = (pts_2d_c2 / pts_2d_c2[..., -1, None])[valid_cuboids, ..., :2]

    # Draw bboxes onto the image
    h, w = rgb.shape[:2]
    for c0, r0, c1, r1 in np.hstack(
        [
            pts_2d_c2[:, _BBOX_EDGES[:, 0]].astype(int).reshape(-1, 2),
            pts_2d_c2[:, _BBOX_EDGES[:, 1]].astype(int).reshape(-1, 2),
        ]
    ):
        rr, cc, val = line_aa(r0, c0, r1, c1)
        mask = np.logical_and.reduce([rr >= 0, rr < h, cc >= 0, cc < w])
        rgb[rr[mask], cc[mask]] = val[mask, None] * np.array((0, 255, 0))

    return Image.fromarray(rgb)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "prefix",
        type=expanduser,
        help="Path to the tracking dataset. Something along the lines of 'path_to_kitti/tracking'.",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        default=10,
        help="Maximum number of frames rendered for each sequence.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=expanduser,
        default=".",
        help="Output folder for the gifs with the rendered 3D bounding boxes.",
    )
    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    ds = KITTITracking(
        args.prefix, fields=["labels", "calibration", "rgb", "lidar", "pose"]
    )
    for seq in ds.train:
        images = []
        for i, data in enumerate(
            tqdm(
                seq,
                desc=f"Rendering for sequence '{seq.name}'",
                total=min(args.frames, len(seq)),
            )
        ):
            # for data in tqdm(seq, desc=f"Rendering for sequence '{seq.name}'"):
            if i == args.frames:
                break

            image = plot_3d_bbox(data["rgb"], data["labels"], seq.calibration)
            images.append(
                image.resize(
                    (image.width // 2, image.height // 2), Image.Resampling.LANCZOS
                )
            )

        output_path = os.path.join(args.output_prefix, f"train-{seq.name}.gif")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000 // 10,
            loop=0,
            optimize=True,
        )


if __name__ == "__main__":
    main()
