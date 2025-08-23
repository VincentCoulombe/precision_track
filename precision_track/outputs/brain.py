# import json
# import os
# import numpy as np

from registry import PRECISION_TRACK

from .base import BaseOutput


# TODO Refactor
@PRECISION_TRACK.register_module()
class BrainOutput(BaseOutput):

    def __init__(self, save_dir: str, video_name: str) -> None:
        super().__init__(save_dir, video_name)

    def __call__(self, data: dict) -> None:
        # frame_name = f"frame{data.frame_id}"
        # out = {frame_name: {"data": {}}}
        # pred_instances = data.pred_track_instances
        # data_fields = pred_instances._data_fields
        # classes = pred_instances.classes
        # unique_classes = np.unique(classes)
        # pred_instances.idx = np.arange(len(pred_instances))
        # for unique_class in unique_classes:
        #     cls_mask = classes == unique_class
        #     cls_pred_instances = pred_instances[cls_mask]
        #     unique_class = str(unique_class)
        #     out[frame_name]["data"][unique_class] = {}
        #     for ints_id, idx in zip(
        #         cls_pred_instances.instances_id, cls_pred_instances.idx
        #     ):
        #         if ints_id >= 0:
        #             out[frame_name]["data"][unique_class][str(ints_id)] = {
        #                 "bbox": cls_pred_instances.bboxes[idx].tolist(),
        #             }
        #             # TODO support for masks and keypoints

        # with open(
        #     os.path.join(f"{self.save_dir}", f"{frame_name}.json"),
        #     "w",
        # ) as f:
        #     json.dump(out, f)
        raise NotImplementedError

    def save(self) -> None:
        return super().save()
