"""Tests of the post-processing movies."""
import os
import pickle
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from scatter import post_proc


def write_results(folder: str) -> str:
    """Write a pickle with the vertical velocity of a 1 m x 1 m plane at y = 0, with 0.5 m elements, at 6 steps."""
    coordinates = np.array([0., 0.5, 1.])
    position = np.array([[x, 0., z] for z in coordinates for x in coordinates])
    time = np.linspace(0, 0.1, 6)
    velocity = {str(i + 1): {"y": 0.05 * np.sin(2 * np.pi * (10 * time - xyz[0]))} for i, xyz in enumerate(position)}
    file = os.path.join(folder, "data.pickle")
    with open(file, "wb") as f:
        pickle.dump({"time": time, "position": position, "velocity": velocity}, f)
    return file


def make_movie(results: str, gif: str, temp_folder: str) -> None:
    """Make the movie of the vertical velocity on the y = 0 plane of the results of write_results."""
    post_proc.make_movie(results, 0., 0.5, 1., 1., "velocity", output_file=gif, temp_folder=temp_folder)


def read_frames(gif: str) -> List[npt.NDArray[np.uint8]]:
    """Read the frames of a gif as RGB arrays."""
    frames = []
    with Image.open(gif) as image:
        for i in range(image.n_frames):
            image.seek(i)
            frames.append(np.array(image.convert("RGB")))
    return frames


class TestMakeMovie:
    """Movie of a result on a horizontal plane."""

    @pytest.fixture(autouse=True)
    def non_interactive_backend(self) -> None:
        """Render the frames without a display."""
        plt.switch_backend("Agg")

    def test_frames_and_duration(self, tmp_path: Any):
        """The movie has one frame per time step before the last one, at 30 frames per second."""
        gif = os.path.join(tmp_path, "movie.gif")
        temp_folder = os.path.join(tmp_path, "frames")
        make_movie(write_results(str(tmp_path)), gif, temp_folder)

        with Image.open(gif) as image:
            assert image.n_frames == 5
            durations = []
            for i in range(image.n_frames):
                image.seek(i)
                durations.append(image.info["duration"])
        # gif stores durations in steps of 10 ms, so 1000 / 30 ms is written as 30 ms
        np.testing.assert_allclose(durations, 30.)
        assert not os.path.exists(temp_folder)

    def test_frames_in_time_order(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
        """The frames are in time order whatever order the file system lists them in."""
        results = write_results(str(tmp_path))
        listdir = os.listdir
        movies = []
        for reverse in [False, True]:
            monkeypatch.setattr(post_proc.os, "listdir", lambda path: sorted(listdir(path), reverse=reverse))
            gif = os.path.join(tmp_path, f"movie_{reverse}.gif")
            make_movie(results, gif, os.path.join(tmp_path, f"frames_{reverse}"))
            movies.append(read_frames(gif))

        # the frames differ, so a different order would show
        assert not np.array_equal(movies[0][0], movies[0][-1])
        assert len(movies[0]) == len(movies[1])
        for frame, expected in zip(movies[1], movies[0]):
            np.testing.assert_array_equal(frame, expected)
