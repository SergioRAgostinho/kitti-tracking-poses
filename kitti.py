import logging
import os
from os import PathLike
from os.path import join as pjoin
from typing import Callable, Iterable, NamedTuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_LOGGER = logging.getLogger(__name__)


def oxts_to_pose(lat, lon, alt, roll, pitch, yaw):
    """This implementation is a python reimplementation of the convertOxtsToPose
    MATLAB function in the original development toolkit for raw data
    """
    n = len(lat)

    # converts lat/lon coordinates to mercator coordinates using mercator scale
    #        mercator scale             * earth radius
    scale = np.cos(lat[0] * np.pi / 180.0) * 6378137

    position = np.stack(
        [
            scale * lon * np.pi / 180.0,
            scale * np.log(np.tan((90.0 + lat) * np.pi / 360.0)),
            alt,
        ],
        axis=-1,
    )

    R = Rotation.from_euler("zyx", np.stack([yaw, pitch, roll], axis=-1)).as_matrix()

    # extract relative transformation with respect to the first frame
    T0_inv = np.block([[R[0].T, -R[0].T @ position[0].reshape(3, 1)], [0, 0, 0, 1]])
    T = T0_inv @ np.block(
        [[R, position[:, :, None]], [np.zeros((n, 1, 3)), np.ones((n, 1, 1))]]
    )
    return T


def load_png(path: str) -> np.ndarray:
    im = Image.open(path)
    return np.array(im)


def load_lidar_data(path):
    scan = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    return scan


def _get_frame_files(path):
    return sorted([int(os.path.splitext(file.name)[0]) for file in os.scandir(path)])


def _calibration_setup_cb(path):
    data = {}
    with open(path + ".txt") as f:
        data["P0"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
        data["P1"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
        data["P2"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
        data["P3"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

        line = f.readline()
        data["R_rect"] = np.fromstring(line[line.index(" ") :], sep=" ").reshape((3, 3))

        line = f.readline()
        data["Tr_velo_cam"] = np.fromstring(line[line.index(" ") :], sep=" ").reshape(
            (3, 4)
        )

        line = f.readline()
        data["Tr_imu_velo"] = np.fromstring(line[line.index(" ") :], sep=" ").reshape(
            (3, 4)
        )
    return data, None


def _labels_frame_cb(seq, key):
    df = seq.labels
    return df[df.frame == key]


def _labels_setup_cb(path):
    cols = (
        "frame",
        "track_id",
        "obj_type",
        "truncation",
        "occlusion",
        "obs_angle",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "l",
        "X",
        "Y",
        "Z",
        "yaw",
    )
    df = pd.read_csv(path + ".txt", sep=" ", names=cols)

    # filter out DontCare obj_types
    df = df[df.obj_type != "DontCare"]
    return df, df.frame.unique().tolist()


def _pose_frame_cb(seq, key):
    return seq.pose[key]


def _pose_setup_cb(path):
    # see readme of raw data dev kit for explanation of these fields
    cols = (
        "lat",
        "lon",
        "alt",
        "roll",
        "pitch",
        "yaw",
        "vn",
        "ve",
        "vf",
        "vl",
        "vu",
        "ax",
        "ay",
        "az",
        "af",
        "al",
        "au",
        "wx",
        "wy",
        "wz",
        "wf",
        "wl",
        "wu",
        "posacc",
        "velacc",
        "navstat",
        "numsats",
        "posmode",
        "velmode",
        "orimode",
    )
    df = pd.read_csv(path + ".txt", sep=" ", names=cols, index_col=False)

    # extract a relative pose with respect to the original frame
    poses = oxts_to_pose(*df[["lat", "lon", "alt", "roll", "pitch", "yaw"]].values.T)
    return poses, df.index.tolist()


def _rgb_frame_cb(seq, key):
    path = pjoin(seq.prefix, seq.fields["rgb"].folder, seq.name, f"{key:06d}.png")
    return load_png(path)


def _rgb_setup_cb(path):
    return None, _get_frame_files(path)


def _lidar_frame_cb(seq, key):
    try:
        scan = load_lidar_data(
            os.path.join(
                seq.prefix, seq.fields["lidar"].folder, seq.name, f"{key:06d}.bin"
            )
        )
    except FileNotFoundError:
        # training sequence 1 has some frames without scans
        scan = np.empty((0, 4), dtype=np.float32)
    return scan


def _lidar_setup_cb(path):
    return None, _get_frame_files(path)


class Field(NamedTuple):
    id: str
    folder: str
    setup_cb: Callable
    frame_cb: Optional[Callable]


class _Sequence:
    """Represents a sequence within a partition."""

    def __init__(self, prefix, name, fields):
        self.prefix = prefix
        self.name = name

        # process sequence wide fields
        self.frames = []
        self.frame_cb = {}
        self.fields = fields

        for field in fields.values():

            if field.setup_cb:
                data, frames = field.setup_cb(pjoin(prefix, field.folder, name))

                # update frame information if possible
                if frames:
                    if len(frames) > len(self.frames):
                        self.frames = frames
                    elif len(frames) == len(self.frames):
                        assert self.frames == frames

                # if there's more info to store
                if data is not None:
                    setattr(self, field.id, data)

            if field.frame_cb:
                self.frame_cb[field.id] = field.frame_cb

        # parse frames to process
        _LOGGER.debug("Initialized sequence %s", name)

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, key):
        return {field: callback(self, key) for field, callback in self.frame_cb.items()}

    def __next__(self):
        if self._i >= len(self):
            raise StopIteration

        out = self[self._i]
        self._i += 1
        return out


class _Partition:
    """Represent a dataset partition, e.g., `train` or `test`."""

    _all_fields = {
        "calibration": Field("calibration", "calib", _calibration_setup_cb, None),
        "labels": Field("labels", "label_02", _labels_setup_cb, _labels_frame_cb),
        "lidar": Field("lidar", "velodyne", _lidar_setup_cb, _lidar_frame_cb),
        "pose": Field("pose", "oxts", _pose_setup_cb, _pose_frame_cb),
        "rgb": Field("rgb", "image_02", _rgb_setup_cb, _rgb_frame_cb),
    }

    def __init__(
        self, name: str, prefix: PathLike, fields: Optional[Iterable[str]] = None
    ):
        self.name = name
        self.prefix = prefix
        self.fields = {}

        # check if all fields are available
        for field in fields:
            if not os.path.isdir(
                pjoin(self.prefix, self.name, self._all_fields[field].folder)
            ):
                raise RuntimeError(f"No data for field: {field}")

            self.fields[field] = self._all_fields[field]

        # if all are present, register callback
        a_key = next(iter(self.fields))

        # TODO this will for fields which use a single file to store info about the
        # the whole sequence
        seq_names = sorted(
            [
                os.path.splitext(d.name)[0]
                for d in os.scandir(
                    pjoin(self.prefix, self.name, self.fields[a_key].folder)
                )
            ]
        )
        self.sequences = [
            _Sequence(os.path.join(self.prefix, self.name), name, self.fields)
            for name in tqdm(seq_names, desc=f"Initializing `{name}` sequences")
        ]

    def __getitem__(self, key):
        return self.sequences[key]

    def __iter__(self):
        return iter(self.sequences)

    def __len__(self):
        return len(self.sequences)


class KITTITracking:
    """Parsing KITTI tracking data."""

    def __init__(self, prefix: PathLike, fields: Optional[Iterable[str]] = None):
        self.prefix = prefix

        self.train = _Partition("training", self.prefix, fields=fields)
        # self.test = _Partition("testing", self.prefix, fields=fields)
