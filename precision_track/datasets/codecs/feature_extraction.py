from precision_track.registry import KEYPOINT_CODECS

from .yolox import YOLOXPoseAnnotationProcessor


@KEYPOINT_CODECS.register_module()
class FEAnnotationProcessor(YOLOXPoseAnnotationProcessor):
    auxiliary_encode_keys = {"category_id", "bbox"}
    label_mapping_table = dict(
        bbox="bboxes",
        bbox_labels="labels",
        instance_id="instances_id",
        keypoints="keypoints",
        keypoints_visible="keypoints_visible",
        area="areas",
    )
    instance_mapping_table = dict(
        bbox="bboxes",
        bbox_score="bbox_scores",
        keypoints="keypoints",
        keypoints_visible="keypoints_visible",
        instance_id="instances_id",
    )
