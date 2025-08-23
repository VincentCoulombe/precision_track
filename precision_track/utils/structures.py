# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List

from mmengine.structures import BaseDataElement, InstanceData, PixelData
from mmengine.utils import is_list_of


class PoseDataSample(BaseDataElement):
    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=(InstanceData, dict))

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def gt_instance_labels(self) -> InstanceData:
        return self._gt_instance_labels

    @gt_instance_labels.setter
    def gt_instance_labels(self, value: InstanceData):
        self.set_field(value, "_gt_instance_labels", dtype=(InstanceData, dict))

    @gt_instance_labels.deleter
    def gt_instance_labels(self):
        del self._gt_instance_labels

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=(InstanceData, dict))

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def gt_fields(self) -> PixelData:
        return self._gt_fields

    @gt_fields.setter
    def gt_fields(self, value: PixelData):
        self.set_field(value, "_gt_fields", dtype=type(value))

    @gt_fields.deleter
    def gt_fields(self):
        del self._gt_fields

    @property
    def pred_fields(self) -> PixelData:
        return self._pred_heatmaps

    @pred_fields.setter
    def pred_fields(self, value: PixelData):
        self.set_field(value, "_pred_heatmaps", dtype=PixelData)

    @pred_fields.deleter
    def pred_fields(self):
        del self._pred_heatmaps


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples into a single data sample.

    This function can be used to merge the top-down predictions with
    bboxes from the same image. The merged data sample will contain all
    instances from the input data samples, and the identical metainfo with
    the first input data sample.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample.
    """

    if not is_list_of(data_samples, PoseDataSample):
        raise ValueError("Invalid input type, should be a list of " ":obj:`PoseDataSample`")

    if len(data_samples) == 0:
        warnings.warn("Try to merge an empty list of data samples.")
        return PoseDataSample()

    merged = PoseDataSample(metainfo=data_samples[0].metainfo)

    if "gt_instances" in data_samples[0]:
        merged.gt_instances = InstanceData.cat([d.gt_instances for d in data_samples])

    if "pred_instances" in data_samples[0]:
        merged.pred_instances = InstanceData.cat([d.pred_instances for d in data_samples])

    return merged


def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if "bboxes" in instances:
            result["bbox"] = (instances.bboxes[i].tolist(),)
            if "bbox_scores" in instances:
                result["bbox_score"] = instances.bbox_scores[i]
        results.append(result)

    return results
