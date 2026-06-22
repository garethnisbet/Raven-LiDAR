"""Read 3DMakerpro Raven ROS1 bags (PointCloud2 scans, IMU, CompressedImage).

The bags are a ROS1 v2.0 container but carry ROS2-style type names
(``sensor_msgs/msg/...``). Their embedded message definitions are genuine ROS1
(``std_msgs/Header`` has a ``seq`` field), so deserialization must use the
**ROS1** typestore -- the ROS2 typestore misaligns on the missing ``seq`` and
raises ``UnicodeDecodeError``.

The ``/vanjee_722z`` PointCloud2 has a 26-byte point step:
    x  f32 @0   y  f32 @4   z  f32 @8
    intensity f32 @12   ring u16 @16   timestamp f64 @18
Scans live in the moving ``vanjee_lidar`` sensor frame and contain NaNs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

TOPIC_CLOUD = "/vanjee_722z"
TOPIC_IMU = "/vanjee_imu_packets"
TOPIC_IMAGE = "/camera_front/image_raw"

# Structured dtype matching the 26-byte vanjee point step.
POINT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "ring", "timestamp"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
        "offsets": [0, 4, 8, 12, 16, 18],
        "itemsize": 26,
    }
)

_TYPESTORE = get_typestore(Stores.ROS1_NOETIC)


def _stamp_ns(header) -> int:
    """ROS Header stamp -> integer nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


@dataclasses.dataclass
class Scan:
    stamp_ns: int
    points: np.ndarray  # structured array of POINT_DTYPE (NaN rows removed)

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.points["x"], self.points["y"], self.points["z"]], axis=1)


def _open(bag: str | Path) -> Reader:
    return Reader(Path(bag))


def topic_summary(bag: str | Path) -> list[tuple[str, str, int]]:
    """Return (topic, msgtype, count) for every connection in the bag."""
    with _open(bag) as r:
        return [(c.topic, c.msgtype, c.msgcount) for c in r.connections]


def iter_pointclouds(bag: str | Path, drop_nan: bool = True) -> Iterator[Scan]:
    """Yield :class:`Scan` for each ``/vanjee_722z`` PointCloud2 message."""
    with _open(bag) as r:
        conns = [c for c in r.connections if c.topic == TOPIC_CLOUD]
        for conn, _t, raw in r.messages(connections=conns):
            m = _TYPESTORE.deserialize_ros1(raw, conn.msgtype)
            n = m.width * m.height
            buf = np.frombuffer(bytes(m.data), dtype=POINT_DTYPE, count=n)
            if drop_nan:
                good = np.isfinite(buf["x"]) & np.isfinite(buf["y"]) & np.isfinite(buf["z"])
                buf = buf[good]
            yield Scan(stamp_ns=_stamp_ns(m.header), points=buf.copy())


def iter_images(bag: str | Path) -> Iterator[tuple[int, str, bytes]]:
    """Yield ``(stamp_ns, format, jpeg_bytes)`` for each CompressedImage."""
    with _open(bag) as r:
        conns = [c for c in r.connections if c.topic == TOPIC_IMAGE]
        for conn, _t, raw in r.messages(connections=conns):
            m = _TYPESTORE.deserialize_ros1(raw, conn.msgtype)
            yield _stamp_ns(m.header), str(m.format), bytes(m.data)


def iter_imu(bag: str | Path) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(stamp_ns, angular_velocity[3], linear_acceleration[3])``."""
    with _open(bag) as r:
        conns = [c for c in r.connections if c.topic == TOPIC_IMU]
        for conn, _t, raw in r.messages(connections=conns):
            m = _TYPESTORE.deserialize_ros1(raw, conn.msgtype)
            av = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            la = np.array(
                [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
            )
            yield _stamp_ns(m.header), av, la


def count_images(bag: str | Path) -> int:
    for topic, _ty, count in topic_summary(bag):
        if topic == TOPIC_IMAGE:
            return count
    return 0


if __name__ == "__main__":
    import sys

    for bag in sys.argv[1:]:
        print(f"== {bag} ==")
        for topic, ty, count in topic_summary(bag):
            print(f"  {topic:28s} {ty:34s} {count}")
