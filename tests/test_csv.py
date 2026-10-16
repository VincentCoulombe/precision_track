import copy
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from utils import temp_csv_file

from precision_track.outputs.csv import (
    CsvActions,
    CsvBoundingBoxes,
    CsvCorrections,
    CsvEmbeddingOutput,
    CsvKeypoints,
    CsvSearchAreas,
    CsvTailtagValidations,
    CsvVelocities,
)

dummy_one_data_sample = {
    "img_id": 2,
    "pred_instances": dict(
        labels=[5],
        bboxes=[[30, 30, 40, 40]],
        scores=[0.5],
        keypoints=[np.array([[35, 35], [45, 45]])],
        keypoint_scores=[np.array([0.8, 0.4])],
        velocities=[[-2, 3]],
    ),
    "pred_track_instances": dict(
        instances_id=[0],
        labels=[5],
        bboxes=[[30, 30, 40, 40]],
        scores=[0.5],
        keypoints=[np.array([[35, 35], [45, 45]])],
        keypoint_scores=[np.array([0.8, 0.4])],
        velocities=[[-2, 3]],
        erraticity=[[0.0]],
        features=[[0.5, -0.25, 0.75, 1.5]],
        action_embeddings=[[-1.5, 0.125, 0.0, 2.25]],
        actions=["Walking"],
        action_scores=[0.5],
    ),
    "correction_instances": dict(instances_id=[0], tags_id=[5], corrected_id=[9]),
    "search_areas": dict(
        instances_id=[0],
        labels=[5],
        bboxes=[[30, 30, 40, 40]],
        scores=[0.5],
    ),
    "validation_instances": dict(instances_id=[0], tags_id=[5], bboxes=[[30, 30, 40, 40]], tags_precision=[0.9]),
}

dummy_empty_data_sample = {
    "img_id": 1,
    "pred_instances": dict(
        labels=[],
        bboxes=[],
        scores=[],
        keypoints=[],
        keypoint_scores=[],
    ),
    "pred_track_instances": dict(
        instances_id=[],
        labels=[],
        bboxes=[],
        scores=[],
        keypoints=[],
        keypoint_scores=[],
        velocities=[],
        erraticity=[],
        features=[],
        action_embeddings=[],
        actions=[],
        action_scores=[],
    ),
    "correction_instances": dict(instances_id=[], tags_id=[], corrected_id=[]),
    "search_areas": dict(
        instances_id=[],
        labels=[],
        bboxes=[],
        scores=[],
    ),
    "validation_instances": dict(instances_id=[], tags_id=[], bboxes=[], tags_precision=[]),
}

dummy_data_sample = {
    "img_id": 0,
    "pred_instances": dict(
        labels=[5, 6],
        bboxes=[[30, 30, 40, 40], [10, 10, 20, 20]],
        scores=[0.5, 0.9],
        keypoints=[np.array([[35, 35], [45, 45]]), np.array([[30, 30], [40, 40]])],
        keypoint_scores=[np.array([0.8, 0.4]), np.array([0.7, 0.5])],
    ),
    "pred_track_instances": dict(
        instances_id=[0, 1],
        labels=[5, 6],
        bboxes=[[30, 30, 40, 40], [10, 10, 20, 20]],
        scores=[0.5, 0.4],
        keypoints=[np.array([[35, 35], [45, 45]]), np.array([[30, 30], [40, 40]])],
        keypoint_scores=[np.array([0.8, 0.4]), np.array([0.7, 0.5])],
        velocities=[[-2, 3], [2, -1]],
        erraticity=[[0.0], [0.5]],
        features=[[0.5, -0.25, 0.75, 1.5], [-0.5, 0.25, -0.75, -1.5]],
        action_embeddings=[[-1.5, 0.125, 0.0, 2.25], [1.5, -0.125, 0.5, -2.25]],
        actions=["Walking", "Rearing"],
        action_scores=[0.5, 0.25],
    ),
    "correction_instances": dict(instances_id=[0, 1], tags_id=[5, 155], corrected_id=[9, 4]),
    "search_areas": dict(
        instances_id=[0, 1],
        labels=[5, 6],
        bboxes=[[30, 30, 40, 40], [10, 10, 20, 20]],
        scores=[0.5, 0.4],
    ),
    "validation_instances": dict(
        instances_id=[0, 1],
        tags_id=[5, 155],
        bboxes=[[30, 30, 40, 40], [10, 10, 20, 20]],
        tags_precision=[0.9, 0.8],
    ),
}


@pytest.fixture
def dummy_data_samples():
    return [dummy_data_sample, dummy_empty_data_sample, dummy_one_data_sample]


@pytest.mark.parametrize(
    "CsvClass, path, subtype",
    [
        (CsvBoundingBoxes, "test_bboxes.csv", "tracked_bboxes"),
        (CsvKeypoints, "test_kpts.csv", None),
        (CsvTailtagValidations, "test_vals.csv", None),
        (CsvSearchAreas, "test_searchs.csv", None),
        (CsvCorrections, "test_corrections.csv", None),
    ],
)
def test_init(CsvClass, path, subtype):
    with temp_csv_file(path):
        with pytest.raises(ValueError):
            for precision in [1, 16, -5, 0.5, 1000, 34, 65]:
                CsvClass(path, precision=precision, subtype=subtype)

        with pytest.raises(AssertionError):
            for inst_data in ["labels", "classes", "tracks", "ids"]:
                CsvClass(path, instance_data=inst_data, subtype=subtype)

        if CsvClass is CsvBoundingBoxes:
            with pytest.raises(AssertionError):
                for unsup_frmt in ["xyxy", "cxcyah", "xywh", "bounding boxes"]:
                    CsvClass(path, bbox_format=unsup_frmt, subtype=subtype)
                CsvClass(path, bbox_format="unsupported_format")
        if CsvClass in [CsvBoundingBoxes, CsvKeypoints]:
            with pytest.raises(AssertionError):
                for unsup_inst_data in [
                    "validation_instances",
                    "correction_instances",
                    "search_areas",
                ]:
                    CsvClass(path, instance_data=unsup_inst_data, subtype=subtype)
            for sup_inst_data in ["pred_instances", "pred_track_instances"]:
                CsvClass(path, instance_data=sup_inst_data, subtype=subtype)
        else:
            with pytest.raises(AssertionError):
                for unsup_inst_data in ["pred_instances", "pred_track_instances"]:
                    CsvClass(path, instance_data=unsup_inst_data)
        if CsvClass is CsvCorrections:
            CsvClass(path, instance_data="correction_instances")
        if CsvClass is CsvSearchAreas:
            CsvClass(path, instance_data="search_areas")
        if CsvClass is CsvTailtagValidations:
            CsvClass(path, instance_data="validation_instances")

        obj = CsvClass(path, subtype=subtype)
        assert obj.path.endswith(path)
        assert obj.precision == 32
        if CsvClass is CsvBoundingBoxes:
            assert obj.confidence_threshold == 0.1
        else:
            assert obj.confidence_threshold == 0.5
        assert os.path.splitext(path)[1] == obj.EXTENSION
        dir_name = os.path.abspath(os.path.dirname(path))
        assert os.path.exists(dir_name)
        assert dir_name == os.path.dirname(obj.path)
        assert os.path.splitext(os.path.basename(path))[0] == os.path.splitext(os.path.basename(obj.path))[0]
        assert obj._results == []
        assert obj.curr_frame_idx == 0


@pytest.mark.parametrize(
    "CsvClass, path, subtype",
    [
        (CsvBoundingBoxes, "test_bboxes.csv", "tracked_bboxes"),
        (CsvKeypoints, "test_kpts.csv", None),
        (CsvTailtagValidations, "test_vals.csv", None),
        (CsvSearchAreas, "test_searchs.csv", None),
        (CsvCorrections, "test_corrections.csv", None),
        (CsvVelocities, "test_vels.csv", None),
        (CsvEmbeddingOutput, "test_embeddings.csv", None),
    ],
)
def test_call(CsvClass, path, subtype, dummy_data_samples):
    obj = CsvClass(path, subtype=subtype)
    data_sample = dummy_data_samples[0]
    obj(data_sample)
    assert len(obj.results) == 2
    assert obj.frame_id_mapping == OrderedDict([(0, (0, 2))])
    assert len(obj) == 1
    data_sample = dummy_data_samples[1]
    obj(data_sample)
    assert obj.frame_id_mapping == OrderedDict([(0, (0, 2)), (1, (2, 2))])
    assert len(obj) == 2
    data_sample = dummy_data_samples[2]
    obj(data_sample)
    assert len(obj.results) == 3
    assert len(obj) == 3
    assert obj.frame_id_mapping == OrderedDict([(0, (0, 2)), (1, (2, 2)), (2, (2, 3))])


@pytest.mark.parametrize(
    "CsvClass, path, subtype",
    [
        (CsvBoundingBoxes, "test_bboxes.csv", "tracked_bboxes"),
        (CsvKeypoints, "test_kpts.csv", None),
        (CsvTailtagValidations, "test_vals.csv", None),
        (CsvSearchAreas, "test_searchs.csv", None),
        (CsvCorrections, "test_corrections.csv", None),
        (CsvVelocities, "test_vels.csv", None),
        (CsvEmbeddingOutput, "test_embeddings.csv", None),
    ],
)
def test_save_and_read(CsvClass, path, subtype, dummy_data_samples):
    with temp_csv_file(path):
        obj = CsvClass(path, subtype=subtype)
        for data_sample in dummy_data_samples:
            obj(data_sample)
        obj.save()
        path = os.path.abspath(path)
        assert os.path.exists(path)
        assert os.path.exists(obj.path)

        obj_read = CsvClass(path, subtype=subtype)
        obj_read.read()
        assert len(obj_read.results) == len(obj.results)
        assert obj.frame_id_mapping == obj_read.frame_id_mapping or obj_read.frame_id_mapping == OrderedDict([(0, (0, 2)), (2, (2, 3))])
        assert len(obj) == len(obj_read)

        for original, loaded in zip(obj.results, obj_read.results):
            assert np.allclose(original, loaded)


@pytest.mark.parametrize(
    "CsvClass, path, subtype",
    [
        (CsvBoundingBoxes, "test_bboxes.csv", "tracked_bboxes"),
        (CsvKeypoints, "test_kpts.csv", None),
        (CsvTailtagValidations, "test_vals.csv", None),
        (CsvSearchAreas, "test_searchs.csv", None),
        (CsvCorrections, "test_corrections.csv", None),
        (CsvVelocities, "test_vels.csv", None),
        (CsvEmbeddingOutput, "test_embeddings.csv", None),
    ],
)
def test_getitem(CsvClass, path, subtype, dummy_data_samples):
    with temp_csv_file(path):
        obj = CsvClass(path, subtype=subtype)
        for data_sample in dummy_data_samples:
            obj(data_sample)

        if isinstance(obj, CsvBoundingBoxes):
            assert obj[0] == [
                [0, 5, -1, 30, 30, 40, 40, 0.5],
                [0, 6, -1, 10, 10, 20, 20, 0.9],
            ]
            assert obj[1] == []
            assert obj[2] == [[2, 5, -1, 30, 30, 40, 40, 0.5]]
        elif isinstance(obj, CsvKeypoints):
            assert obj[0] == [[0, 5, -1, 35, 35, 0.8, 45, 45, 0.4], [0, 6, -1, 30, 30, 0.7, 40, 40, 0.5]]
            assert obj[1] == []
            assert obj[2] == [[2, 5, -1, 35, 35, 0.8, 45, 45, 0.4]]
        elif isinstance(obj, CsvVelocities):
            assert obj[0] == [[0, 5, 0, -2, 3, 0.0], [0, 6, 1, 2, -1, 0.5]]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, -2, 3, 0.0]]
        elif isinstance(obj, CsvTailtagValidations):
            assert obj[0] == [
                [0, 5, 0, 30, 30, 40, 40, 0.9],
                [0, 155, 1, 10, 10, 20, 20, 0.8],
            ]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, 30, 30, 40, 40, 0.9]]
        elif isinstance(obj, CsvCorrections):
            assert obj[0] == [[0, 5, 0, 9], [0, 155, 1, 4]]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, 9]]
        elif isinstance(obj, CsvEmbeddingOutput):
            assert obj.columns == ["emb_0", "emb_1", "emb_2", "emb_3"]
            assert obj[0] == [
                [0, 5, 0, 0.5, -0.25, 0.75, 1.5],
                [0, 6, 1, -0.5, 0.25, -0.75, -1.5],
            ]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, 0.5, -0.25, 0.75, 1.5]]
        elif isinstance(obj, CsvSearchAreas):
            assert obj[0] == [
                [0, 5, 0, 30, 30, 40, 40],
                [0, 6, 1, 10, 10, 20, 20],
            ]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, 30, 30, 40, 40]]

        assert obj[3] == [[]]


def test_embedding_init():
    path = "test_embeddings.csv"
    with temp_csv_file(path):
        for precision in [1, 32, 64, -5, 0.5, 1000]:
            obj = CsvEmbeddingOutput(path, precision=precision)
            assert obj.precision == 16, "CsvEmbeddingOutput always stores float16, whatever is configured."

        for unsup_inst_data in ["validation_instances", "correction_instances", "search_areas", "pred_instances"]:
            with pytest.raises(AssertionError):
                CsvEmbeddingOutput(path, instance_data=unsup_inst_data)

        obj = CsvEmbeddingOutput(path)
        assert obj.instance_data == "pred_track_instances"
        assert obj.ids_field == "instances_id"
        assert obj.embs_field == "features"
        assert obj.columns == []
        assert obj._results == []
        assert obj.curr_frame_idx == 0
        assert obj.SUPPORTED_PRECISION == {16: "float16"}


@pytest.mark.parametrize("embs_field", ["features", "action_embeddings"])
def test_embedding_columns_and_dtype(embs_field, dummy_data_samples):
    path = "test_embeddings.csv"
    with temp_csv_file(path):
        obj = CsvEmbeddingOutput(path, embs_field=embs_field)
        assert obj.columns == []
        for data_sample in dummy_data_samples:
            obj(data_sample)

        emb_dim = len(dummy_data_samples[0]["pred_track_instances"][embs_field][0])
        assert obj.columns == [f"emb_{i}" for i in range(emb_dim)]

        df = obj.to_dataframe()
        assert df.columns.tolist() == ["frame_id", "class_id", "instance_id"] + obj.columns
        assert all(str(df[col].dtype) == "float16" for col in obj.columns)
        assert str(df["frame_id"].dtype) == "uint32"
        assert str(df["class_id"].dtype) == "uint16"
        assert str(df["instance_id"].dtype) == "int16"


def test_embedding_dimension_is_resolved_once(dummy_data_samples):
    """The width is derived from the first embedding and never re-checked, so a later
    frame of a different dimension surfaces as a ragged frame at save() time."""
    path = "test_embeddings.csv"
    with temp_csv_file(path):
        obj = CsvEmbeddingOutput(path)
        obj(dummy_data_samples[0])
        assert len(obj.columns) == 4

        wider = copy.deepcopy(dummy_data_samples[2])
        wider["pred_track_instances"]["features"] = [[0.5, -0.25, 0.75, 1.5, 0.125, -0.5]]
        obj(wider)
        assert obj.columns == ["emb_0", "emb_1", "emb_2", "emb_3"]

        with pytest.raises(ValueError):
            obj.to_dataframe()


def test_embedding_missing_field_still_advances_the_frame_mapping(dummy_data_samples):
    """``action_embeddings`` is absent from pred_track_instances on frames where the
    analyzer saw no subject, which must not desynchronize the frame index."""
    path = "test_embeddings.csv"
    with temp_csv_file(path):
        obj = CsvEmbeddingOutput(path, embs_field="action_embeddings")
        obj(dummy_data_samples[0])

        without = copy.deepcopy(dummy_data_samples[2])
        without["pred_track_instances"].pop("action_embeddings")
        obj(without)

        assert obj.frame_id_mapping == OrderedDict([(0, (0, 2)), (2, (2, 2))])
        assert len(obj) == 3
        assert obj[2] == []


JOINABLE_OUTPUTS = {
    "tracked_bboxes.csv": lambda p: CsvBoundingBoxes(p, subtype="tracked_bboxes", instance_data="pred_track_instances"),
    "tracked_kpts.csv": lambda p: CsvKeypoints(p, instance_data="pred_track_instances"),
    "tracked_velocities.csv": lambda p: CsvVelocities(p),
    "tracked_actions.csv": lambda p: CsvActions(p),
    "detection_embeddings.csv": lambda p: CsvEmbeddingOutput(p),
    "action_embeddings.csv": lambda p: CsvEmbeddingOutput(p, embs_field="action_embeddings"),
}

JOIN_KEY = ["frame_id", "class_id", "instance_id"]


def test_joinable_on_the_first_three_columns(tmp_path, dummy_data_samples):
    """``(frame_id, class_id, instance_id)`` must be a unique key that joins exactly
    across every per-track CSV output, including the embedding outputs.

    The samples cover the two cases a per-frame output can disagree on: a frame with no
    instance at all, and a frame holding only a subset of the subjects.
    """
    outputs = {name: build(str(tmp_path / name)) for name, build in JOINABLE_OUTPUTS.items()}
    for data_sample in dummy_data_samples:
        for output in outputs.values():
            output(data_sample)
    for output in outputs.values():
        output.save()

    frames = pd.read_csv(tmp_path / "detection_embeddings.csv")[JOIN_KEY]
    assert len(frames) == 3, "two instances on frame 0, none on frame 1, one on frame 2"

    for name in outputs:
        df = pd.read_csv(tmp_path / name)
        keys = df[JOIN_KEY]

        # 1) the triple is a unique key
        assert not keys.duplicated().any(), f"{name} has a duplicated (frame_id, class_id, instance_id)"

        # 2) every file carries the exact same rows, in the same order
        assert keys.equals(frames), f"{name} does not carry the same rows as the embeddings output"

    # 3) an inner join of any pair returns every row of both sides, with nothing unmatched
    for left in outputs:
        for right in outputs:
            df_left = pd.read_csv(tmp_path / left)
            df_right = pd.read_csv(tmp_path / right)
            joined = df_left.merge(df_right, on=JOIN_KEY, how="inner", suffixes=("_l", "_r"))
            assert len(joined) == len(df_left) == len(df_right), f"{left} JOIN {right} lost or duplicated rows"


def test_embedding_rows_match_bboxes_frame_by_frame(dummy_data_samples):
    """Same invariant as the join test, but checked on the in-memory slices so a
    divergence is attributable to a single frame."""
    with temp_csv_file("test_embeddings.csv"), temp_csv_file("test_bboxes.csv"):
        emb = CsvEmbeddingOutput("test_embeddings.csv")
        bboxes = CsvBoundingBoxes("test_bboxes.csv", subtype="tracked_bboxes", instance_data="pred_track_instances")
        for data_sample in dummy_data_samples:
            emb(data_sample)
            bboxes(data_sample)

        assert emb.frame_id_mapping == bboxes.frame_id_mapping
        assert len(emb) == len(bboxes)
        for frame_id in range(len(emb)):
            emb_keys = [row[:3] for row in emb[frame_id]]
            bbox_keys = [row[:3] for row in bboxes[frame_id]]
            assert emb_keys == bbox_keys, f"frame {frame_id} diverges"


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__), "-v", "-s"])
