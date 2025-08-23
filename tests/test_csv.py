import os
from collections import OrderedDict

import numpy as np
import pytest
from utils import temp_csv_file

from precision_track.outputs.csv import CsvBoundingBoxes, CsvCorrections, CsvKeypoints, CsvSearchAreas, CsvValidations, CsvVelocities

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
        velocities=[],
    ),
    "pred_track_instances": dict(
        instances_id=[],
        labels=[],
        bboxes=[],
        scores=[],
        keypoints=[],
        keypoint_scores=[],
        velocities=[],
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
        velocities=[[-2, 3], [2, -1]],
    ),
    "pred_track_instances": dict(
        instances_id=[0, 1],
        labels=[5, 6],
        bboxes=[[30, 30, 40, 40], [10, 10, 20, 20]],
        scores=[0.5, 0.4],
        keypoints=[np.array([[35, 35], [45, 45]]), np.array([[30, 30], [40, 40]])],
        keypoint_scores=[np.array([0.8, 0.4]), np.array([0.7, 0.5])],
        velocities=[[-2, 3], [2, -1]],
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
    "CsvClass, path",
    [
        (CsvBoundingBoxes, "test_bboxes.csv"),
        (CsvKeypoints, "test_kpts.csv"),
        (CsvValidations, "test_vals.csv"),
        (CsvSearchAreas, "test_searchs.csv"),
        (CsvCorrections, "test_corrections.csv"),
    ],
)
def test_init(CsvClass, path):
    with temp_csv_file(path):
        with pytest.raises(ValueError):
            for precision in [1, 16, -5, 0.5, 1000, 34, 65]:
                CsvClass(path, precision=precision)

        with pytest.raises(AssertionError):
            for inst_data in ["labels", "classes", "tracks", "ids"]:
                CsvClass(path, instance_data=inst_data)

        if CsvClass is CsvBoundingBoxes:
            with pytest.raises(AssertionError):
                for unsup_frmt in ["xyxy", "cxcyah", "xywh", "bounding boxes"]:
                    CsvClass(path, bbox_format=unsup_frmt)
                CsvClass(path, bbox_format="unsupported_format")
        if CsvClass in [CsvBoundingBoxes, CsvKeypoints]:
            with pytest.raises(AssertionError):
                for unsup_inst_data in [
                    "validation_instances",
                    "correction_instances",
                    "search_areas",
                ]:
                    CsvClass(path, instance_data=unsup_inst_data)
            for sup_inst_data in ["pred_instances", "pred_track_instances"]:
                CsvClass(path, instance_data=sup_inst_data)
        else:
            with pytest.raises(AssertionError):
                for unsup_inst_data in ["pred_instances", "pred_track_instances"]:
                    CsvClass(path, instance_data=unsup_inst_data)
        if CsvClass is CsvCorrections:
            CsvClass(path, instance_data="correction_instances")
        if CsvClass is CsvSearchAreas:
            CsvClass(path, instance_data="search_areas")
        if CsvClass is CsvValidations:
            CsvClass(path, instance_data="validation_instances")

        obj = CsvClass(path)
        assert obj.path.endswith(path)
        assert obj.precision == 32
        assert obj.confidence_threshold == 0.5
        assert os.path.splitext(path)[1] == obj.EXTENSION
        dir_name = os.path.abspath(os.path.dirname(path))
        assert os.path.exists(dir_name)
        assert dir_name == os.path.dirname(obj.path)
        assert os.path.splitext(os.path.basename(path))[0] == os.path.splitext(os.path.basename(obj.path))[0]
        assert obj.results == []
        assert obj.curr_frame_idx == 0


@pytest.mark.parametrize(
    "CsvClass, path",
    [
        (CsvBoundingBoxes, "test_bboxes.csv"),
        (CsvKeypoints, "test_kpts.csv"),
        (CsvValidations, "test_vals.csv"),
        (CsvSearchAreas, "test_searchs.csv"),
        (CsvCorrections, "test_corrections.csv"),
        (CsvVelocities, "test_vels.csv"),
    ],
)
def test_call(CsvClass, path, dummy_data_samples):
    obj = CsvClass(path)
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
    "CsvClass, path",
    [
        (CsvBoundingBoxes, "test_bboxes.csv"),
        (CsvKeypoints, "test_kpts.csv"),
        (CsvValidations, "test_vals.csv"),
        (CsvSearchAreas, "test_searchs.csv"),
        (CsvCorrections, "test_corrections.csv"),
        (CsvVelocities, "test_vels.csv"),
    ],
)
def test_save_and_read(CsvClass, path, dummy_data_samples):
    with temp_csv_file(path):
        obj = CsvClass(path)
        for data_sample in dummy_data_samples:
            obj(data_sample)
        obj.save()
        path = os.path.abspath(path)
        assert os.path.exists(path)
        assert os.path.exists(obj.path)

        obj_read = CsvClass(path)
        obj_read.read()
        assert len(obj_read.results) == len(obj.results)
        assert obj.frame_id_mapping == obj_read.frame_id_mapping or obj_read.frame_id_mapping == OrderedDict([(0, (0, 2)), (2, (2, 3))])
        assert len(obj) == len(obj_read)

        for original, loaded in zip(obj.results, obj_read.results):
            assert np.allclose(original, loaded)


@pytest.mark.parametrize(
    "CsvClass, path",
    [
        (CsvBoundingBoxes, "test_bboxes.csv"),
        (CsvKeypoints, "test_kpts.csv"),
        (CsvValidations, "test_vals.csv"),
        (CsvSearchAreas, "test_searchs.csv"),
        (CsvCorrections, "test_corrections.csv"),
        (CsvVelocities, "test_vels.csv"),
    ],
)
def test_getitem(CsvClass, path, dummy_data_samples):
    with temp_csv_file(path):
        obj = CsvClass(path)
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
            assert obj[0] == [[0, 5, 0, -2, 3], [0, 6, 1, 2, -1]]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, -2, 3]]
        elif isinstance(obj, CsvValidations):
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
        elif isinstance(obj, CsvSearchAreas):
            assert obj[0] == [
                [0, 5, 0, 30, 30, 40, 40],
                [0, 6, 1, 10, 10, 20, 20],
            ]
            assert obj[1] == []
            assert obj[2] == [[2, 5, 0, 30, 30, 40, 40]]
        assert obj[3] == []


if __name__ == "__main__":
    pytest.main()
