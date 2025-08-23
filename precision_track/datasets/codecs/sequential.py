from precision_track.registry import KEYPOINT_CODECS

from .yolox import YOLOXPoseAnnotationProcessor


@KEYPOINT_CODECS.register_module()
class SequenceAnnotationProcessor(YOLOXPoseAnnotationProcessor):
    auxiliary_encode_keys = {"category_id", "bbox"}
    label_mapping_table = dict(
        bbox="bboxes",
        bbox_labels="labels",
        keypoints="keypoints",
        keypoints_visible="keypoints_visible",
        area="areas",
        id="ids",
        action_label="action_labels",
    )
    instance_mapping_table = dict(
        bbox="bboxes",
        bbox_score="bbox_scores",
        keypoints="keypoints",
        keypoints_visible="keypoints_visible",
        id="ids",
        action="actions",
    )
