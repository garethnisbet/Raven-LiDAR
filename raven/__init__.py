"""raven: tools to edit 3DMakerpro Raven (JMK7) LiDAR captures and build Gaussian splats.

Pipeline:
    extract -> (cloud_editor) -> colmap_pipeline -> align -> train_splat

See README.md for the full command sequence.
"""

__version__ = "0.1.0"
