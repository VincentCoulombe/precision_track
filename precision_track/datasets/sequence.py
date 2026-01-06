import copy
import os
import shutil
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from logging import WARNING
from typing import Callable, List, Optional, Tuple, Union
import cv2
import numpy as np
import math
import torch
from addict import Dict
from mmengine import Config
from mmengine.dataset.base_dataset import BaseDataset, force_full_init
from mmengine.logging import MMLogger, print_log
from mmengine.structures import InstanceData
from tqdm import tqdm
from collections import defaultdict

from precision_track.apis.association_step import AssociationStep
from precision_track.apis.result import Result
from precision_track.models.backends import DetectionBackend
from precision_track.models.optimization.coaches import BaseCoach
from precision_track.registry import DATASETS, OUTPUTS, COACHES
from precision_track.outputs import BaseOutput
from precision_track.outputs.display import display_class_balance
from precision_track.utils import (
    PoseDataSample,
    VideoReader,
    find_path_in_dir,
    infer_paths,
    iou_batch,
    linear_assignment,
    noisify,
    parse_pose_metainfo,
    reformat,
    update_dynamics_2d,
    get_seq_from_img_folder,
    batch_tracking,
)

from .transforms.formatting import image_to_tensor

warnings.simplefilter("ignore", UserWarning)


@DATASETS.register_module()
class OnlineRandomSequenceDataset(BaseDataset):
    MANDATORY_PREFIX_KEYS = ["sequences", "bboxes_gt_paths"]
    METAINFO = dict()
    METAINFO_KEYS = [
        "dataset_name",
        "upper_body_ids",
        "lower_body_ids",
        "flip_pairs",
        "dataset_keypoint_weights",
        "flip_indices",
        "skeleton_links",
    ]
    LABEL_KEYS = [
        "sequence_name",
        "img_id",
        "img_path",
        "nb_instances",
        "bbox",
        "bbox_score",
        "category_id",
        "keypoints",
        "keypoints_visible",
        "id",
    ]
    DEFAULT_KEYS = (
        METAINFO_KEYS
        + LABEL_KEYS
        + [
            "gt_instance_labels",
            "gt_instances",
        ]
    )

    def __init__(
        self,
        from_file: str,
        bboxes_gt_format: str,
        keypoints_gt_format: Optional[str] = None,
        actions_gt_format: Optional[str] = None,
        data_root: Optional[str] = ".",
        data_prefix: dict = dict(
            sequences=["."],
            bboxes_gt_paths=[""],
        ),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        block_size: Optional[int] = 2,
        img_ext: Optional[str] = ".jpg",
        coach: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        self.logger = MMLogger.get_current_instance()
        self.METAINFO.update(from_file=from_file)
        assert block_size > 0
        self.block_size = block_size

        self.bboxes_gt_format = bboxes_gt_format
        self.keypoints_gt_format = keypoints_gt_format if keypoints_gt_format is not None else "CsvKeypoints"
        self.actions_gt_format = actions_gt_format if actions_gt_format is not None else "CsvActions"

        self.img_ext = img_ext
        self._length = 0

        self.action_counter = defaultdict(int)
        self.action_to_label_map = dict()

        self.idx_map = list()
        self._seq_lengths = dict()

        self.coach = COACHES.build(coach) if coach is not None else coach

        super().__init__(
            ann_file=None,
            data_prefix=data_prefix,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=False,
        )

    def _add_prefix_key(self, prefix_keys: list, new_prefix_key: str):
        prefix_keys = prefix_keys + [new_prefix_key]
        if new_prefix_key in self.data_prefix:
            self._init_data_prefix_key(new_prefix_key)
        else:
            self.data_prefix[new_prefix_key] = ["" for _ in self.data_prefix[self.MANDATORY_PREFIX_KEYS[0]]]
        return prefix_keys

    def _join_prefix(self):
        missing_keys = [k for k in self.MANDATORY_PREFIX_KEYS if k not in self.data_prefix]
        assert not missing_keys, f"Missing mandatory keys: {missing_keys}"

        for key in self.MANDATORY_PREFIX_KEYS:
            self._init_data_prefix_key(key)

        prefix_keys = self.MANDATORY_PREFIX_KEYS
        prefix_keys = self._add_prefix_key(prefix_keys, "keypoints_gt_paths")
        prefix_keys = self._add_prefix_key(prefix_keys, "actions_gt_paths")

        lengths = [len(self.data_prefix[key]) for key in self.MANDATORY_PREFIX_KEYS]
        assert len(set(lengths)) == 1, "Ensure that you have the same number of gt paths than of image directories."

        sequences, keypoints_outputs, bboxes_outputs, actions_outputs = [], [], [], []
        for seq, bboxes_path, kpts_path, actions_path in zip(*[self.data_prefix[k] for k in self.MANDATORY_PREFIX_KEYS]):
            full_dir = os.path.join(self.data_root, seq)
            sequences.append(full_dir)

            bboxes_output = OUTPUTS.build({"type": self.bboxes_gt_format, "path": os.path.join(self.data_root, bboxes_path)})
            bboxes_output.read()
            bboxes_outputs.append(bboxes_output)

            kpts_output = OUTPUTS.build({"type": self.keypoints_gt_format, "path": os.path.join(self.data_root, kpts_path)})
            if kpts_output.valid():
                kpts_output.read()
            else:
                self.logger.warning(
                    f"This training run will not consider the subject poses. To consider it, please provide path to the ground truth's files to the data_prefix dictionnary."
                )
            keypoints_outputs.append(kpts_output)

            actions_output = OUTPUTS.build({"type": self.actions_gt_format, "path": os.path.join(self.data_root, actions_path)})
            if actions_output.valid():
                actions_output.read()
            else:
                self.logger.warning(
                    f"This training run will not train the model for action recognition. To do so, please provide path to the ground truth's files to the data_prefix dictionnary."
                )
            actions_outputs.append(actions_output)

        self.data_prefix.update(
            {
                "sequences": sequences,
                "bboxes_outputs": bboxes_outputs,
                "keypoints_outputs": keypoints_outputs,
                "actions_outputs": actions_outputs,
            }
        )

    def get_data_info(self, idx: int, block_idx: int, subject_id: Optional[tuple] = None, **kwargs) -> dict:
        data_info = self.data_list[idx]
        out = dict(sequence_dir=data_info["sequence_dir"])

        if idx >= 0:
            out["sample_idx"] = idx
        else:
            out["sample_idx"] = len(self) + idx

        for output_k in ["bboxes_output", "kpts_output", "actions_output"]:
            output = data_info.get(output_k)
            if isinstance(output, BaseOutput) and output.valid() and output.results:
                block_output = deepcopy(np.array(output[block_idx : block_idx + self.block_size]))
                if isinstance(subject_id, tuple) and len(subject_id) == 2:
                    # Skim down the memory footprint by only preserving the relevant id's output
                    cat = int(subject_id[0])
                    id_ = int(subject_id[1])
                    id_mask = (block_output[:, 1, ...].astype(int) == cat) & (block_output[:, 2, ...].astype(int) == id_)
                    block_output = block_output[id_mask]
                out[output_k] = block_output

        metainfo_keys = ["dataset_name", "upper_body_ids", "lower_body_ids", "flip_pairs", "dataset_keypoint_weights", "flip_indices", "skeleton_links"]
        for key in metainfo_keys:
            out[key] = deepcopy(self._metainfo[key])
        for key, v in zip(["idx", "block_idx", "subject_id"], [idx, block_idx, subject_id]):
            out[key] = v
        for key in kwargs:
            assert key not in out, f"kwargs cannot override the sequence's {key} key."
            out[key] = kwargs[key]
        return out

    def set_sequence_transforms(self):
        for transform in self.pipeline.transforms:
            if hasattr(transform, "set_stochastic_params"):
                transform.set_stochastic_params()

    def get_sequence(self, idx):
        if self.test_mode:
            sequence_idx, block_idx = self.idx_map[idx]
            sequence = self.get_data_info(sequence_idx, block_idx)
        else:
            if isinstance(self.coach, BaseCoach):
                sequence_idx, block_idx, kwargs = self.coach.get_idx()
            else:
                kwargs = dict()
                sequence_idx = np.random.randint(low=0, high=len(self.data_list))
                seq_length = self._seq_lengths[sequence_idx]
                if seq_length < self.block_size:
                    block_idx = 0
                else:
                    block_idx = np.random.randint(low=0, high=seq_length - self.block_size)
            sequence = self.get_data_info(sequence_idx, block_idx, **kwargs)
            self.set_sequence_transforms()

        sequence_dir = sequence["sequence_dir"]
        images = get_seq_from_img_folder(block_idx, self.block_size, sequence_dir)

        return sequence_idx, sequence, block_idx, images

    def prepare_data(self, idx):
        # TODO ajouter rdm timestep (uniforme dans [1, 4])
        # TODO ajouter sequence coach
        # TODO ajouter rdm occlusion (transform)
        # TODO loader ground truth pour le block_size+1 frame

        sequence_idx, sequence, block_idx, images = self.get_sequence(idx)

        inputs = []
        metainfo = dict()
        img_shape = None

        gt_instance_labels = defaultdict(list)
        gt_instances = defaultdict(list)

        bboxes_output = sequence.get("bboxes_output")
        kpts_output = sequence.get("kpts_output")
        kpts_out_valid = isinstance(kpts_output, np.ndarray)
        actions_output = sequence.get("actions_output")
        actions_out_valid = isinstance(actions_output, np.ndarray)

        data_samples = []

        for running_idx, image_path in enumerate(images):
            abs_idx = block_idx + running_idx
            sequence["absolute_frame_id"] = abs_idx
            running_idx_mask = bboxes_output[:, 0] == abs_idx
            N = running_idx_mask.sum()
            sequence["category_id"] = bboxes_output[running_idx_mask, 1].astype(int)
            sequence["instance_id"] = bboxes_output[running_idx_mask, 2].astype(int)
            sequence["bbox"] = bboxes_output[running_idx_mask, 3:7]
            sequence["bbox"] = reformat(sequence["bbox"], "xywh", "xyxy")
            sequence["bbox_score"] = bboxes_output[running_idx_mask, 7]

            if kpts_out_valid:
                kpts = kpts_output[running_idx_mask, ...]
                assert np.all(
                    kpts[:, 1].astype(int) == sequence["category_id"]
                ), f"Incoherance found between the keypoints ground truth's class ids and the bounding boxes ground truth's class ids on frame: {block_idx+running_idx}."
                assert np.all(
                    kpts[:, 2].astype(int) == sequence["instance_id"]
                ), f"Incoherance found between the keypoints ground truth's instance ids and the bounding boxes ground truth's instance ids on frame: {block_idx+running_idx}."
                kpts = kpts[:, 3:]
                num_kpts = kpts.shape[1] // 3
                kpts = kpts.reshape(-1, num_kpts, 3)
                sequence["keypoints_visible"] = kpts[..., -1]
                sequence["keypoints"] = kpts[..., :2]
            else:
                num_kpts = 0
                sequence["keypoints"] = np.zeros((N, num_kpts, 2))
                sequence["keypoints_visible"] = np.zeros((N, num_kpts, 1))
            sequence["num_keypoints"] = np.array([num_kpts])

            if actions_out_valid:
                actions = actions_output[running_idx_mask, ...]
                assert np.all(
                    actions[:, 1].astype(int) == sequence["category_id"]
                ), f"Incoherance found between the actions ground truth's class ids and the bounding boxes ground truth's class ids on frame: {block_idx+running_idx}."
                assert np.all(
                    actions[:, 2].astype(int) == sequence["instance_id"]
                ), f"Incoherance found between the actions ground truth's instance ids and the bounding boxes ground truth's instance ids on frame: {block_idx+running_idx}."
                sequence["action_label"] = actions[:, 3]
                sequence["action"] = np.array([self.action_to_label_map[a] for a in actions[:, 3]])
            else:
                sequence["action_label"] = np.zeros((N, 1)) - 1
                sequence["action"] = np.zeros((N, 1)) - 1

            sequence["img_path"] = image_path

            if isinstance(self.coach, BaseCoach):
                sequence = self.coach.select_idx_labels(sequence)

            transformed_sequence = self.pipeline(sequence)
            metainfo["input_size"] = transformed_sequence["input_size"]
            metainfo["input_scale"] = transformed_sequence["input_scale"]
            metainfo["input_center"] = transformed_sequence["input_center"]
            metainfo["img_shape"] = transformed_sequence["img_shape"]
            if img_shape is None:
                img_shape = transformed_sequence["img_shape"]

            inputs.append(transformed_sequence.pop("img"))

            gt_instance_labels = InstanceData(action_labels=transformed_sequence["action_label"].reshape(-1, 1))

            gt_instances = InstanceData()
            gt_instances.set_data({"actions": torch.from_numpy(transformed_sequence["action"].reshape(-1, 1))})
            gt_instances.set_data({"bboxes": torch.from_numpy(transformed_sequence["bbox"])})

            x1, y1, x2, y2 = transformed_sequence["bbox"].T
            assert np.all(x2 > x1) and np.all(y2 > y1)
            gt_instances.set_data({"areas": torch.from_numpy(np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None))})

            gt_instances.set_data({"scores": torch.from_numpy(transformed_sequence["bbox_score"])})
            gt_instances.set_data({"keypoints": torch.from_numpy(transformed_sequence["keypoints"])})
            gt_instances.set_data({"keypoints_visible": torch.from_numpy(transformed_sequence["keypoints_visible"])})

            gt_instances.set_data({"instances_id": torch.from_numpy(transformed_sequence["instance_id"])})
            gt_instances.set_data({"labels": torch.from_numpy(transformed_sequence["category_id"])})
            gt_instances.set_data({"frame_id": torch.full((transformed_sequence["action"].shape[0],), running_idx)})

            data_sample = PoseDataSample(metainfo=metainfo)
            data_sample.action_label_counter = self.action_label_counter
            data_sample.seq_id = f"{sequence_idx}-{block_idx}"
            data_sample.img_id = running_idx
            data_sample.ori_shape = transformed_sequence["ori_shape"]
            data_sample.gt_instance_labels = gt_instance_labels
            data_sample.gt_instances = gt_instances
            data_samples.append(data_sample)

        # self.save_sequence(f"{sequence_idx}-{block_idx}", inputs, data_samples)  # TODO Externalize this sanity check hack

        return dict(inputs=image_to_tensor(inputs), data_samples=data_samples)

    def save_sequence(self, seq_idx: int, images: list, data_samples):
        save_dir = os.path.abspath(os.path.join(self.data_root, "augmented_sequences", str(seq_idx)))
        os.makedirs(save_dir, exist_ok=True)

        outputs = dict(
            bboxes=OUTPUTS.build({"type": self.bboxes_gt_format, "path": ""}),
            keypoints=OUTPUTS.build({"type": self.keypoints_gt_format, "path": ""}),
            actions=OUTPUTS.build({"type": self.actions_gt_format, "path": ""}),
        )
        for k, o in outputs.items():
            o.path = os.path.join(save_dir, f"{k}{o.EXTENSION}")

        for i, image in enumerate(images):
            gt_instances = data_samples[i].gt_instances.to_dict()
            os.makedirs(os.path.join(save_dir, "frames"), exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "frames", f"{i}.png"), image)
            N = gt_instances["frame_id"][i].shape[0]

            frame_ids = gt_instances["frame_id"][i].reshape(N).tolist()
            labels = gt_instances["labels"][i].reshape(N).tolist()
            instance_ids = gt_instances["instances_id"][i].reshape(N).tolist()

            bboxes = gt_instances["bboxes"][i].reshape(N, 4).tolist()
            scores = gt_instances["scores"][i].reshape(N).tolist()
            keypoints = (
                np.concatenate((gt_instances["keypoints"][i].reshape(N, -1, 2), gt_instances["keypoints_visible"][i].reshape(N, -1, 1)), axis=2)
                .reshape(N, -1)
                .tolist()
            )
            actions = gt_instances["actions"][i].reshape(N).tolist()
            for j in range(N):
                bbox = bboxes[j]
                outputs["bboxes"]._add_row(frame_ids[j], labels[j], instance_ids[j], *bbox, scores[j])
                outputs["keypoints"]._add_row(frame_ids[j], labels[j], instance_ids[j], keypoints[j])
                outputs["actions"]._add_row(frame_ids[j], labels[j], instance_ids[j], actions[j], scores[j])

        for output in outputs.values():
            output.save()

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(f"metainfo should be a dict, but got {type(metainfo)}")

        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def load_metadata(self, data_info: int) -> dict:
        for key in self.METAINFO_KEYS:
            if key not in data_info:
                data_info[key] = deepcopy(self._metainfo[key])
        return data_info

    @force_full_init
    def __len__(self) -> int:
        return self._length

    def _init_data_prefix_key(self, key):
        if isinstance(self.data_prefix[key], str):
            directory = os.path.join(self.data_root, self.data_prefix[key])
            assert os.path.isdir(directory), f"{key} is expected to be a list or a directory."
            new_prefix = []
            for file in os.listdir(directory):
                new_prefix.append(os.path.join(directory, file))
            self.data_prefix[key] = new_prefix
        else:
            assert isinstance(self.data_prefix[key], list), f"{key} is expected to be a list or a directory."

    def load_data_list(self) -> List[dict]:
        self.logger.info("Loading sequences...")
        prefix_map = []
        self.action_to_label_map = {ac: i for i, ac in enumerate(self.metainfo.get("actions", []))}
        sequence_dirs = [os.path.abspath(i) for i in self.data_prefix["sequences"]]
        sequence_names = [os.path.basename(os.path.normpath(i)) for i in sequence_dirs]
        for i, sequence_name in enumerate(sequence_names):
            j = find_path_in_dir(sequence_name, self.data_prefix["bboxes_gt_paths"])
            k = self.data_prefix.get("keypoints_gt_paths", "")
            if k:
                k = find_path_in_dir(sequence_name, k)
            a = self.data_prefix.get("actions_gt_paths", "")
            if a:
                a = find_path_in_dir(sequence_name, a)
            prefix_map.append([i, j, k, a])

        data_list = []
        for sequence_idx, kpts_idx, bboxes_idx, actions_idx in tqdm(prefix_map):

            actions_valid = isinstance(actions_idx, int) and actions_idx >= 0
            if not actions_valid:
                actions_idx = bboxes_idx

            kpts_valid = kpts_idx and kpts_idx >= 0
            if not kpts_valid:
                kpts_idx = bboxes_idx
            data = dict()
            seq_path = sequence_dirs[sequence_idx]
            data["sequence_dir"] = seq_path
            assert os.path.isdir(seq_path), f"{seq_path} is not a valid directory."
            data["kpts_output"] = self.data_prefix["keypoints_outputs"][kpts_idx]
            data["bboxes_output"] = self.data_prefix["bboxes_outputs"][bboxes_idx]
            data["actions_output"] = self.data_prefix["actions_outputs"][actions_idx]
            actions = data["actions_output"]
            if actions_valid:
                actions_really_valid = actions.valid() and actions.results
                assert actions_really_valid, f"The provided ground truths: {actions.path} is either invalid or contains no results."
                if actions_really_valid:
                    actions_np = np.array(actions.results)[:, -1]
                    uniques, counts = np.unique(actions_np, return_counts=True)
                    for unique, count in zip(uniques, counts):
                        if unique not in self.action_to_label_map.keys():
                            raise ValueError(
                                f"The action: {unique} from the dataset, is not in the actions defined in the metadata file: {list(self.action_to_label_map.keys())}"
                            )
                        self.action_counter[unique] += count

            ref_len = data["bboxes_output"]
            if actions_valid:
                assert len(ref_len) == len(data["actions_output"]), "Ground truths from the same sequence must have the same length."
            if kpts_valid:
                assert len(ref_len) == len(data["kpts_output"]), "Ground truths from the same sequence must have the same length."
            assert len(ref_len) <= len(
                os.listdir(seq_path)
            ), f"{len(ref_len)} frames are labelled, but only {len(os.listdir(seq_path))} frames are saved at the {seq_path} directory."

            block_idxs = np.arange(0, len(ref_len), self.block_size)[:-1].tolist()
            self._seq_lengths[sequence_idx] = block_idxs[-1]
            self._length += len(block_idxs)
            self.idx_map.extend([(sequence_idx, block_idx) for block_idx in block_idxs])
            data["sequence_idx"] = sequence_idx

            if isinstance(self.coach, BaseCoach):
                self.coach.load_info(data)

            data_list.append(data)
        if self.action_counter:
            self.logger.info("Your labelled action's proportions: ")
            display_class_balance(self.action_counter)
        self.action_label_counter = np.zeros(len(self.action_to_label_map))
        for action, a_count in self.action_counter.items():
            idx = self.action_to_label_map[action]
            self.action_label_counter[idx] = a_count
        self.action_label_counter = self.action_label_counter.tolist()
        return data_list


class OfflineRandomSequenceDataset(BaseDataset, metaclass=ABCMeta):
    MANDATORY_PREFIX_KEYS = ["sequences", "bboxes_gt_paths", "keypoints_gt_paths"]
    METAINFO = dict()
    METAINFO_KEYS = [
        "dataset_name",
        "upper_body_ids",
        "lower_body_ids",
        "flip_pairs",
        "dataset_keypoint_weights",
        "flip_indices",
        "skeleton_links",
        "num_keypoints",
    ]
    LABEL_KEYS = [
        "sequence_name",
        "img_id",
        "img_path",
        "nb_instances",
        "bbox",
        "bbox_score",
        "category_id",
        "keypoints",
        "keypoints_visible",
        "id",
    ]
    DEFAULT_KEYS = (
        METAINFO_KEYS
        + LABEL_KEYS
        + [
            "gt_instance_labels",
            "gt_instances",
        ]
    )

    def __init__(
        self,
        from_file: str,
        detector: Config,
        bboxes_gt_format: Optional[str] = "CsvBoundingBoxes",
        keypoints_gt_format: Optional[str] = "CsvKeypoints",
        actions_gt_format: Optional[str] = None,
        data_root: Optional[str] = ".",
        data_prefix: dict = dict(
            sequences=["."],
            bboxes_gt_paths=[""],
            keypoints_gt_paths=[""],
            actions_gt_paths=[None],
        ),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        block_size: Optional[int] = 2,
        inference_resolution: Optional[tuple] = None,
        *args,
        **kwargs,
    ):
        self.detector = DetectionBackend(**detector)
        self.actions_gt_format = actions_gt_format
        self.action_to_label_map = dict()
        self.input_scale = inference_resolution  # TODO rendre dynamique. 1 par video...
        self.input_center = None
        self.cat_warned = False
        if isinstance(self.input_scale, (Tuple, list, np.ndarray)):
            self.input_scale = np.array(self.input_scale)
            self.input_center = self.input_scale // 2

        self.logger = MMLogger.get_current_instance()
        self.METAINFO.update(from_file=from_file)
        assert block_size > 0
        self.block_size = block_size

        self.bboxes_gt_format = bboxes_gt_format
        self.keypoints_gt_format = keypoints_gt_format

        self._length = 0

        super().__init__(
            ann_file=None,
            data_prefix=data_prefix,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=False,
        )

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(f"metainfo should be a dict, but got {type(metainfo)}")

        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def load_metadata(self, data_info: int) -> dict:
        for key in self.METAINFO_KEYS:
            if key not in data_info:
                data_info[key] = deepcopy(self._metainfo[key])
        return data_info

    def set_sequence_transforms(self):
        for transform in self.pipeline.transforms:
            if hasattr(transform, "set_stochastic_params"):
                transform.set_stochastic_params()

    def _init_data_prefix_key(self, key):
        if isinstance(self.data_prefix[key], str):
            directory = os.path.abspath(os.path.join(self.data_root, self.data_prefix[key]))
            assert os.path.isdir(directory), f"{key} is expected to be a list or a directory."
            new_prefix = []
            for file in os.listdir(directory):
                new_prefix.append(os.path.join(directory, file))
            self.data_prefix[key] = new_prefix
        else:
            assert isinstance(self.data_prefix[key], list), f"{key} is expected to be a list or a directory."

    def _join_prefix(self):
        missing_keys = [k for k in self.MANDATORY_PREFIX_KEYS if k not in self.data_prefix]
        assert not missing_keys, f"Missing mandatory keys: {missing_keys}"

        for key in self.MANDATORY_PREFIX_KEYS:
            self._init_data_prefix_key(key)

        prefix_keys = self.MANDATORY_PREFIX_KEYS + ["actions_gt_paths"]
        if "actions_gt_paths" in self.data_prefix:
            self._init_data_prefix_key("actions_gt_paths")
        else:
            self.data_prefix["actions_gt_paths"] = [None for _ in self.data_prefix[self.MANDATORY_PREFIX_KEYS[0]]]

        lengths = [len(self.data_prefix[key]) for key in prefix_keys]
        assert len(set(lengths)) == 1, "Ensure that you have the same number of gt paths than of image directories."

        sequences, keypoints_outputs, bboxes_outputs, actions_outputs = [], [], [], []
        for seq, bboxes_path, kpts_path, actions_path in zip(*[self.data_prefix[k] for k in prefix_keys]):
            full_dir = os.path.join(self.data_root, seq)
            sequences.append(full_dir)

            bboxes_output = OUTPUTS.build({"type": self.bboxes_gt_format, "path": os.path.join(self.data_root, bboxes_path)})
            bboxes_output.read()
            bboxes_outputs.append(bboxes_output)

            kpts_output = OUTPUTS.build({"type": self.keypoints_gt_format, "path": os.path.join(self.data_root, kpts_path)})
            kpts_output.read()
            keypoints_outputs.append(kpts_output)

            if self.actions_gt_format is not None and actions_path is not None:
                actions_output = OUTPUTS.build({"type": self.actions_gt_format, "path": os.path.join(self.data_root, actions_path)})
                actions_output.read()
            else:
                actions_output = None
            actions_outputs.append(actions_output)

        self.data_prefix.update(
            {
                "sequences": sequences,
                "bboxes_outputs": bboxes_outputs,
                "keypoints_outputs": keypoints_outputs,
                "actions_outputs": actions_outputs,
            }
        )

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        pass

    @force_full_init
    def __len__(self) -> int:
        return self._length

    def load_data_list(self) -> List[dict]:
        self.logger.info("Loading sequences...")
        prefix_map = []
        self.action_to_label_map = {ac: i for i, ac in enumerate(self.metainfo.get("actions", []))}
        sequences_name = [os.path.basename(os.path.normpath(i)) for i in self.data_prefix["sequences"]]
        for i, sequence_name in enumerate(sequences_name):
            j = find_path_in_dir(sequence_name, self.data_prefix["bboxes_gt_paths"])
            k = find_path_in_dir(sequence_name, self.data_prefix["keypoints_gt_paths"])
            a = self.data_prefix.get("actions_gt_paths")
            if a is not None:
                a = find_path_in_dir(sequence_name, a)
            prefix_map.append([i, j, k, a])

        data_list = [[] for _ in sequences_name]
        for sequence_idx, kpts_idx, bboxes_idx, actions_idx in tqdm(prefix_map):
            self.set_sequence_transforms()
            vid_reader = VideoReader(self.data_prefix["sequences"][sequence_idx])
            sequence_name = sequences_name[sequence_idx]
            kpts_output = self.data_prefix["keypoints_outputs"][kpts_idx]
            bboxes_output = self.data_prefix["bboxes_outputs"][bboxes_idx]
            if actions_idx is None or actions_idx < 0:
                actions_output = copy.deepcopy(bboxes_output)
            else:
                actions_output = self.data_prefix["actions_outputs"][actions_idx]

            batch = defaultdict(list)
            for i, (frame_kpts, frame_bboxes, frame_actions) in enumerate(zip(kpts_output, bboxes_output, actions_output)):
                if frame_kpts and frame_bboxes:
                    frame_trk_info_kpts, frame_kpts = self._standardize_frame_data(frame_kpts)
                    frame_trk_info_bboxes, frame_bboxes = self._standardize_frame_data(frame_bboxes)
                    frame_trk_info_actions, frame_actions = self._standardize_frame_data(frame_actions)

                    assert np.allclose(
                        frame_trk_info_kpts, frame_trk_info_bboxes
                    ), f"the bounding boxes and the keypoints ground truth do not agree on the {i}th frame."
                    assert np.allclose(
                        frame_trk_info_kpts, frame_trk_info_actions
                    ), f"the actions and the keypoints ground truth do not agree on the {i}th frame."
                    assert np.all(frame_trk_info_kpts[:, 0] == i), f"The {i}th frame have an indexing problem."

                    category_ids = frame_trk_info_bboxes[:, 1]
                    invalid_cat = category_ids < 1
                    if invalid_cat.any() and not self.cat_warned:
                        self.logger.warning("Category ids smaller than 1 are bump up to 1.")
                        self.cat_warned = True
                    category_ids[category_ids < 1] = 1
                    instance_ids = frame_trk_info_bboxes[:, 2]
                    bboxes = frame_bboxes[:, :4]
                    scores = np.ones_like(category_ids)
                    areas = np.clip(bboxes[:, 2] * bboxes[:, 3] * 0.53, a_min=1.0, a_max=None)

                    kpts = frame_kpts[:, ...].reshape(-1, self._metainfo["num_keypoints"], 3)
                    kpt_scores = kpts[..., -1]
                    kpts = kpts[..., :2]

                    assert category_ids.shape[0] == bboxes.shape[0] == scores.shape[0] == kpt_scores.shape[0] == kpts.shape[0] == instance_ids.shape[0]
                    img = vid_reader.get_frame(i)
                    assert img is not None, f"Error while reading the {i}th frame of video {vid_reader.filename}."

                    frame_data = dict(
                        sequence_name=sequence_name,
                        img_id=i,
                        img=vid_reader.get_frame(i),
                        nb_instances=bboxes.shape[0],
                        id=instance_ids.astype(int),
                        bbox=reformat(bboxes, "xywh", "xyxy").astype(np.float32),
                        bbox_score=scores.astype(np.float32),
                        category_id=category_ids.astype(np.float32),
                        keypoints=kpts.astype(np.float32),
                        keypoints_visible=kpt_scores.astype(np.float32),
                        area=areas.astype(np.float32),
                    )

                    if actions_idx is not None and actions_idx >= 0:
                        actions = frame_actions.reshape(-1)
                        action_labels = np.zeros_like(actions, dtype=int)
                        for i, action in enumerate(actions):
                            action_label = self.action_to_label_map.get(action)
                            assert action_label is not None, f"The action: {action}, from the dataset, is not in the metadata's actions list."
                            action_labels[i] = self.action_to_label_map[action]

                        assert category_ids.shape[0] == action_labels.shape[0] == actions.shape[0]
                        frame_data.update(
                            dict(
                                action=actions,
                                action_label=action_labels,
                            )
                        )
                    data = self.pipeline(self.load_metadata(frame_data))
                    batch["inputs"].append(data["inputs"])
                    batch["data_samples"].append(data["data_samples"])
                    if len(batch["inputs"]) == 30 or i == len(kpts_output) - 1:
                        outputs = self.detector(inputs=batch["inputs"], data_samples=batch["data_samples"], mode="predict")

                        for j, data_sample in enumerate(batch["data_samples"]):
                            pred_bboxes_np = outputs[j]["pred_instances"]["bboxes"].cpu().numpy().astype(np.float32)
                            pred_labels_np = outputs[j]["pred_instances"]["labels"].cpu().numpy()

                            o_gt_bboxes = outputs[j]["gt_instances"]["bboxes"]
                            ds_gt_bboxes = data_sample.gt_instance_labels.bboxes
                            assert torch.allclose(o_gt_bboxes.cpu(), ds_gt_bboxes.cpu())

                            gt_bboxes_np = reformat(outputs[j]["gt_instances"]["bboxes"].cpu().numpy(), "xyxy", "cxcywh").astype(np.float32)
                            gt_labels_np = outputs[j]["gt_instances"]["labels"].cpu().numpy()

                            if pred_bboxes_np.size > 0 and gt_bboxes_np.size > 0:

                                dists = iou_batch(pred_bboxes_np, gt_bboxes_np)

                                labels_mask = pred_labels_np[:, None] == gt_labels_np[None, :]
                                dists = (dists * labels_mask).astype(np.float32)
                                matched_preds, matched_gts = linear_assignment(1 - dists, 0.9)
                                matched_preds = torch.from_numpy(matched_preds).long()
                                matched_gts = torch.from_numpy(matched_gts).long()

                            else:
                                matched_preds = torch.tensor([], dtype=torch.long)
                                matched_gts = torch.tensor([], dtype=torch.long)

                            ids = data_sample.gt_instance_labels.ids[matched_gts]
                            unique_idx = np.unique(ids.numpy(), return_index=True)[1]
                            ids = ids[unique_idx]

                            features = outputs[j]["pred_instances"]["features"][matched_preds][unique_idx].cpu()
                            bboxes = outputs[j]["pred_instances"]["bboxes"][matched_preds][unique_idx].cpu()
                            keypoints = outputs[j]["pred_instances"]["keypoints"][matched_preds][unique_idx].cpu()
                            keypoints_visible = outputs[j]["pred_instances"]["keypoint_scores"][matched_preds][unique_idx].cpu()
                            labels = outputs[j]["pred_instances"]["labels"][matched_preds][unique_idx].cpu()

                            action_labels = []
                            if hasattr(data_sample.gt_instance_labels, "action_labels"):
                                action_labels = data_sample.gt_instance_labels.action_labels[matched_gts][unique_idx]

                            actions = []
                            if hasattr(data_sample.gt_instances, "actions"):
                                actions = np.array(data_sample.gt_instances.actions[matched_gts]).reshape(len(matched_gts))[unique_idx]

                            input_size = data_sample.metainfo["input_size"]
                            input_scale = data_sample.metainfo["ori_shape"]

                            if isinstance(self.input_scale, np.ndarray) and (self.input_scale != input_scale).all():
                                scale = torch.tensor(self.input_scale, dtype=torch.float32, device=bboxes.device)
                                rescale = scale / torch.tensor(input_size, dtype=torch.float32, device=bboxes.device)
                                translation = torch.tensor(self.input_center, dtype=torch.float32, device=bboxes.device) - 0.5 * scale

                                keypoints = keypoints * rescale.view(1, 1, 2) + translation.view(1, 1, 2)
                                bboxes = bboxes * torch.tile(rescale, (bboxes.shape[0], 2)) + torch.tile(translation, (bboxes.shape[0], 2))

                            pred_track_instances = Dict()
                            pred_track_instances.bboxes = bboxes
                            pred_track_instances.kpts = keypoints
                            pred_track_instances.kpt_vis = keypoints_visible
                            pred_track_instances.instances_id = ids
                            pred_track_instances.labels = labels
                            pred_track_instances.features = features

                            gt_instance_labels = InstanceData()
                            gt_instance_labels.action_labels = action_labels
                            gt_instances = InstanceData()
                            gt_instances.actions = actions

                            light_ds = PoseDataSample()
                            light_ds.pred_track_instances = pred_track_instances
                            light_ds.gt_instance_labels = gt_instance_labels
                            light_ds.gt_instances = gt_instances
                            light_ds.img_id = data_sample.img_id
                            light_ds.seq_id = sequence_idx
                            light_ds.seq_name = os.path.splitext(sequence_name)[0]

                            data_list[sequence_idx].append(light_ds)
                            self._length += len(batch)
                        batch = defaultdict(list)
        return data_list

    @staticmethod
    def _standardize_frame_data(frame_data):
        frame_data = np.array(frame_data)
        return frame_data[:, :3].astype(float).astype(int), frame_data[:, 3:]


@DATASETS.register_module()
class ActionRecognitionDataset(OfflineRandomSequenceDataset):

    def __init__(
        self,
        from_file: str,
        detector: Config,
        n_feats: int,
        n_velocities: int,
        bboxes_gt_format: Optional[str] = "CsvBoundingBoxes",
        keypoints_gt_format: Optional[str] = "CsvKeypoints",
        actions_gt_format: Optional[str] = None,
        data_root: Optional[str] = ".",
        data_prefix: dict = dict(
            sequences=["."],
            bboxes_gt_paths=[""],
            keypoints_gt_paths=[""],
        ),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        block_size: Optional[int] = 2,
        negative_action: Optional[str] = "Other",
        weighted_selection: Optional[bool] = False,
        inference_resolution: Optional[tuple] = None,
        ignore_idx: Optional[int] = -100,
        *args,
        **kwargs,
    ):
        self.action_to_sequence_map = defaultdict(list)
        super().__init__(
            from_file=from_file,
            detector=detector,
            bboxes_gt_format=bboxes_gt_format,
            keypoints_gt_format=keypoints_gt_format,
            actions_gt_format=actions_gt_format,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            block_size=block_size,
            inference_resolution=inference_resolution,
            *args,
            **kwargs,
        )
        self.negative_label = self.action_to_label_map[negative_action]
        self.label_to_action_map = {v: k for k, v in self.action_to_label_map.items()}
        self.numpy_mapper = np.vectorize(lambda x: self.label_to_action_map[x])
        self.labels = list(self.action_to_label_map.values())

        weights = np.array([len(v) for v in self.action_to_sequence_map.values()])
        self.logger.info(f"ActionRecognitionDataset initialized with a total of {weights.sum(0)} labels.")
        if not weighted_selection:
            weights = np.ones_like(list(self.action_to_label_map.values()))
        self.p = weights / weights.sum(0)

        for size in [n_feats, n_velocities]:
            assert 0 < size
        self.n_feats = n_feats
        self.n_kpts = self.metainfo.get("num_keypoints", 0)
        assert self.n_kpts > 0, f"The metainfo's {self.METAINFO} 'keypoint_info' contains no keypoints."
        self.n_velocities = n_velocities

        self._ignore_idx = int(ignore_idx)

    def prepare_data(self, idx):
        random_action = np.random.choice(a=self.labels, p=self.p)

        nb_action_labels = len(self.action_to_sequence_map[random_action])
        assert self.block_size < nb_action_labels, f"An action have less labels ({nb_action_labels}) than the specified block size ({self.block_size})."

        random_action_idx = np.random.randint(0, nb_action_labels)
        inputs = torch.zeros((self.block_size, self.n_feats), dtype=torch.float32, device="cpu")
        kpts = torch.zeros((self.block_size, self.n_kpts, 2), dtype=torch.float32, device="cpu")
        kpt_vis = torch.zeros((self.block_size, self.n_kpts), dtype=torch.float32, device="cpu")
        dynamics = torch.zeros((self.block_size, self.n_velocities), dtype=torch.float32, device="cpu")
        actions = torch.zeros((self.block_size), dtype=torch.float32, device="cpu")
        positives = torch.zeros((self.block_size), dtype=torch.float32, device="cpu")
        inputs_ds = PoseDataSample()
        inputs_ds.gt_instance_labels = InstanceData()
        inputs_ds.gt_instances = InstanceData()
        inputs_ds.pred_track_instances = InstanceData()

        seq, seq_name, idx, id_ = self.action_to_sequence_map[random_action][random_action_idx]

        for block_idx in range(self.block_size):
            data_sample = self.data_list[seq][idx - self.block_size + block_idx + 1]
            id_idx = torch.where(data_sample.pred_track_instances.instances_id == id_)[0]
            if id_idx.numel() > 0:
                inputs[block_idx] = data_sample.pred_track_instances.features[id_idx]

                kpts[block_idx] = data_sample.pred_track_instances.kpts[id_idx]
                kpt_vis[block_idx] = data_sample.pred_track_instances.kpt_vis[id_idx]

                dynamics[block_idx] = data_sample.pred_track_instances.dynamics[id_idx]

                actions[block_idx] = data_sample.gt_instance_labels.action_labels[id_idx]
                positives[block_idx] = data_sample.gt_instance_labels.action_labels[id_idx] != self.negative_label
            else:
                actions[block_idx] = self._ignore_idx

        assert id_idx.numel() > 0, f"Bad synchronization for id {id_} of frame {idx} of sequence {seq}."
        action = data_sample.gt_instance_labels.action_labels[id_idx]
        assert action == random_action, (
            f"Action for sequence: {seq}, frame idx: {idx} and instance id: {id_} should be {self.label_to_action_map[random_action]},"
            f"but got {self.label_to_action_map[action.item()]}."
        )

        inputs_ds.gt_instance_labels.action_labels = actions
        inputs_ds.gt_instance_labels.positives = positives
        inputs_ds.gt_instances.actions = self.numpy_mapper(action.cpu().numpy())
        inputs_ds.pred_track_instances.kpts = kpts
        inputs_ds.pred_track_instances.kpt_vis = kpt_vis
        inputs_ds.pred_track_instances.velocities = dynamics
        inputs_ds.img_id = idx
        inputs_ds.seq_id = seq
        inputs_ds.seq_name = seq_name
        inputs_ds.instance_id = id_

        return dict(inputs=inputs, data_samples=inputs_ds)

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        for s, sequence in enumerate(data_list):
            seq_dynamics = dict()
            for i, data_sample in enumerate(sequence):
                frame_id = data_sample.img_id
                seq_name = data_sample.seq_name
                bboxes = data_sample.pred_track_instances.bboxes
                del data_sample.pred_track_instances.bboxes
                actions = data_sample.gt_instance_labels.action_labels
                ids = data_sample.pred_track_instances.instances_id
                frame_dynamics = torch.zeros((len(ids), 6), device=ids.device, dtype=bboxes.dtype)
                centroids = bboxes[:, :2].numpy()
                for j, (id_, action, location) in enumerate(zip(ids, actions, centroids)):
                    id_ = id_.item()
                    action = action.item()
                    if id_ not in seq_dynamics:
                        seq_dynamics[id_] = np.array([location[0], location[1], 0, 0, 0, 0, frame_id], dtype=np.float32)
                    else:
                        dynamics = seq_dynamics[id_][:6]
                        dt = frame_id - seq_dynamics[id_][-1]
                        dynamics = update_dynamics_2d(dynamics, location.astype(np.float32), seq_dynamics[id_][:2].copy(), 0.5, dt)
                        seq_dynamics[id_][:6] = dynamics
                        seq_dynamics[id_][-1] = frame_id
                    frame_dynamics[j] = torch.from_numpy(seq_dynamics[id_][:6]).to(torch.float32)

                    if frame_id >= self.block_size:  # Removing first self.block_size frames from each sequences.
                        self.action_to_sequence_map[action].append((s, seq_name, i, id_))
                data_sample.pred_track_instances.dynamics = frame_dynamics[:, 2:4]
        return data_list


@DATASETS.register_module()
class ActionRecognitionPerFrameDataset(ActionRecognitionDataset):
    def prepare_data(self, idx):
        seq, seq_idx = self.idx_to_seq[idx]

        inputs_ds = PoseDataSample()
        inputs_ds.gt_instance_labels = InstanceData()
        inputs_ds.gt_instances = InstanceData()
        inputs_ds.pred_track_instances = InstanceData()

        for i, block_idx in enumerate(reversed(range(self.block_size))):
            data_sample = self.data_list[seq][seq_idx - i]
            if i == 0:
                valid_ids = data_sample.pred_track_instances.instances_id

                inputs = torch.zeros((len(valid_ids), self.block_size, self.n_feats), dtype=torch.float32, device="cpu")
                kpts = torch.zeros((len(valid_ids), self.block_size, self.n_kpts, 2), dtype=torch.float32, device="cpu")
                kpt_vis = torch.zeros((len(valid_ids), self.block_size, self.n_kpts), dtype=torch.float32, device="cpu")
                dynamics = torch.zeros((len(valid_ids), self.block_size, self.n_velocities), dtype=torch.float32, device="cpu")

                actions = data_sample.gt_instance_labels.action_labels

            for valid_idx, valid_id in enumerate(valid_ids):
                id_idx = torch.where(data_sample.pred_track_instances.instances_id == valid_id)[0]
                if id_idx.numel() > 0:
                    inputs[valid_idx, block_idx] = data_sample.pred_track_instances.features[id_idx]

                    kpts[valid_idx, block_idx] = data_sample.pred_track_instances.kpts[id_idx]
                    kpt_vis[valid_idx, block_idx] = data_sample.pred_track_instances.kpt_vis[id_idx]
                    dynamics[valid_idx, block_idx] = data_sample.pred_track_instances.dynamics[id_idx]

        inputs_ds.gt_instance_labels.action_labels = actions
        inputs_ds.gt_instances.actions = self.numpy_mapper(actions.cpu().numpy())
        inputs_ds.pred_track_instances.kpts = kpts
        inputs_ds.pred_track_instances.kpt_vis = kpt_vis
        inputs_ds.pred_track_instances.dynamics = dynamics
        inputs_ds.pred_track_instances.instances_id = valid_ids

        inputs_ds.img_id = idx
        inputs_ds.seq_id = seq_idx

        return dict(inputs=inputs, data_samples=inputs_ds)

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        self.idx_to_seq = dict()
        self._length = 0
        for i, seq in enumerate(data_list):
            seq_length = len(seq)
            for j in range(self.block_size - 1, seq_length):
                self.idx_to_seq[self._length] = (i, j)
                self._length += 1
        return data_list

    def __len__(self):
        return self._length


@DATASETS.register_module()
class MAEDataset(OfflineRandomSequenceDataset):
    UNSUP_MANDATORY_KEYS = ["sequences"]
    UNSUP_MANDATORY_OUTPUTS = ["CsvBoundingBoxes", "CsvVelocities", "CsvKeypoints", "OnlinePthEmbeddingOutput"]

    def __init__(
        self,
        from_file,
        n_feats: int,
        n_velocities: int,
        detector,
        bboxes_gt_format="CsvBoundingBoxes",
        keypoints_gt_format="CsvKeypoints",
        data_prefix=dict(sequences=["."], bboxes_gt_paths=[""], keypoints_gt_paths=[""]),
        pipeline=[],
        test_mode=False,
        block_size=2,
        inference_resolution=None,
        nb_simulteneous_seq=3,
        supervized=True,
        tracking_batch_size=30,
        assigner=None,
        outputs=None,
        custom_length=None,
        *args,
        **kwargs,
    ):

        assert nb_simulteneous_seq > 0
        self.nb_simulteneous_seq = int(nb_simulteneous_seq)
        self._all_data_prefix = None
        self._nb_sequences = 0

        assert isinstance(supervized, bool)
        self.supervized = supervized

        if not self.supervized:
            assert isinstance(assigner, dict)
            assert isinstance(outputs, list)
            filtered_outputs = []
            already_loaded = []
            for output in outputs:
                assert isinstance(output, dict)
                name = output["type"]
                if name in self.UNSUP_MANDATORY_OUTPUTS and name not in already_loaded:
                    filtered_outputs.append(output)
                    already_loaded.append(name)
            assert len(filtered_outputs) == len(self.UNSUP_MANDATORY_OUTPUTS)
            assert tracking_batch_size > 0
        self.assigner = assigner
        self.tracking_outputs = filtered_outputs
        self.tracking_batch_size = int(tracking_batch_size)

        if custom_length is None:
            custom_length = -1
        custom_length = int(custom_length)
        self._custom_length = custom_length

        super().__init__(
            from_file=from_file,
            detector=detector,
            bboxes_gt_format=bboxes_gt_format,
            keypoints_gt_format=keypoints_gt_format,
            actions_gt_format=None,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            block_size=block_size,
            inference_resolution=inference_resolution,
        ),
        for size in [n_feats, n_velocities]:
            assert 0 < size
        self.n_feats = n_feats
        self.n_kpts = self.metainfo.get("num_keypoints", 0)
        assert self.n_kpts > 0, f"The metainfo's {self.METAINFO} 'keypoint_info' contains no keypoints."
        self.n_velocities = n_velocities

    @force_full_init
    def prepare_data(self, idx):

        id_ = np.random.choice(self.instances)
        seq = np.random.choice(list(self.instance_sequences[id_].keys()))
        instance_sequence = self.instance_sequences[id_][seq]
        seq_idx = np.random.randint(0, len(instance_sequence))
        idx = instance_sequence[seq_idx]

        inputs = torch.zeros((self.block_size, self.n_feats), dtype=torch.float32, device="cpu")
        kpts = torch.zeros((self.block_size, self.n_kpts, 2), dtype=torch.float32, device="cpu")
        kpt_vis = torch.zeros((self.block_size, self.n_kpts), dtype=torch.float32, device="cpu")
        dynamics = torch.zeros((self.block_size, self.n_velocities), dtype=torch.float32, device="cpu")
        inputs_ds = PoseDataSample()
        inputs_ds.gt_instance_labels = InstanceData()
        inputs_ds.gt_instances = InstanceData()
        inputs_ds.pred_track_instances = InstanceData()

        for block_idx in range(self.block_size):
            data_sample = self.data_list[seq][idx - self.block_size + block_idx + 1]
            id_idx = torch.where(data_sample.pred_track_instances.instances_id == id_)[0]
            if id_idx.numel() > 0:
                inputs[block_idx] = data_sample.pred_track_instances.features[id_idx]

                kpts[block_idx] = data_sample.pred_track_instances.kpts[id_idx]
                kpt_vis[block_idx] = data_sample.pred_track_instances.kpt_vis[id_idx]

                dynamics[block_idx] = data_sample.pred_track_instances.dynamics[id_idx]

        assert id_idx.numel() > 0, f"Bad synchronization for id {id_} of frame {idx} of sequence {seq}."

        inputs_ds.pred_track_instances.kpts = kpts
        inputs_ds.pred_track_instances.kpt_vis = kpt_vis
        inputs_ds.pred_track_instances.velocities = dynamics
        inputs_ds.img_id = idx
        inputs_ds.seq_id = seq
        inputs_ds.instance_id = id_

        return dict(inputs=inputs, data_samples=inputs_ds)

    def _pick_random_seqs(self):
        seq_idxs = np.arange(self._nb_sequences)
        selected_seq_idxs = []
        for _ in range(self.nb_simulteneous_seq):
            selected_seq_idxs.append(np.random.choice(seq_idxs, replace=False))
        self.data_prefix = copy.deepcopy(self._all_data_prefix)
        for k, v in self.data_prefix.items():
            v = infer_paths(v)
            assert len(v) >= self.nb_simulteneous_seq, f"Cannot radonmly select {self.nb_simulteneous_seq} sequences from a pool of only {len(v)} sequences."
            self.data_prefix[k] = np.array(v)[selected_seq_idxs].tolist()

    def _maybe_init_all_prefix(self):
        if self._all_data_prefix is None:
            for key in self.UNSUP_MANDATORY_KEYS:
                self._init_data_prefix_key(key)
            self._all_data_prefix = copy.deepcopy(self.data_prefix)
            self._nb_sequences = len(self._all_data_prefix["sequences"])

    def _join_prefix(self):
        if self.supervized:
            missing_keys = [k for k in self.MANDATORY_PREFIX_KEYS if k not in self.data_prefix]
            assert not missing_keys, f"Missing mandatory keys: {missing_keys}"

            self._maybe_init_all_prefix()
            self._pick_random_seqs()

            prefix_keys = self.MANDATORY_PREFIX_KEYS

            lengths = [len(self.data_prefix[key]) for key in prefix_keys]
            assert len(set(lengths)) == 1, "Ensure that you have the same number of gt paths than of image directories."

            sequences, keypoints_outputs, bboxes_outputs = [], [], []
            for seq, bboxes_path, kpts_path in zip(*[self.data_prefix[k] for k in prefix_keys]):
                sequences.append(seq)

                bboxes_output = OUTPUTS.build({"type": self.bboxes_gt_format, "path": bboxes_path})
                bboxes_output.read()
                bboxes_outputs.append(bboxes_output)

                kpts_output = OUTPUTS.build({"type": self.keypoints_gt_format, "path": kpts_path})
                kpts_output.read()
                keypoints_outputs.append(kpts_output)

            self.data_prefix.update(
                {
                    "sequences": sequences,
                    "bboxes_outputs": bboxes_outputs,
                    "keypoints_outputs": keypoints_outputs,
                }
            )
        else:
            missing_keys = [k for k in self.UNSUP_MANDATORY_KEYS if k not in self.data_prefix]
            assert not missing_keys, f"Missing mandatory keys: {missing_keys}"

            self._maybe_init_all_prefix()
            self._pick_random_seqs()

            sequences = []
            for prefix_list in [self.data_prefix[k] for k in self.UNSUP_MANDATORY_KEYS]:
                sequences.extend(iter(prefix_list))
            self.data_prefix.update(
                {
                    "sequences": sequences,
                }
            )

    def load_data_list(self) -> List[dict]:
        self.instance_sequences = defaultdict(dict)
        self._length = 0
        if self.supervized:
            data_list = super().load_data_list()
            for s, sequence in enumerate(data_list):
                seq_dynamics = dict()
                for data_sample in sequence:
                    frame_id = data_sample.img_id
                    bboxes = data_sample.pred_track_instances.bboxes
                    del data_sample.pred_track_instances.bboxes
                    ids = data_sample.pred_track_instances.instances_id
                    frame_dynamics = torch.zeros((len(ids), 6), device=ids.device, dtype=bboxes.dtype)
                    centroids = bboxes[:, :2].numpy()
                    for j, (id_, location) in enumerate(zip(ids, centroids)):
                        id_ = id_.item()
                        if id_ not in seq_dynamics:
                            seq_dynamics[id_] = np.array([location[0], location[1], 0, 0, 0, 0, frame_id], dtype=np.float32)
                        else:
                            dynamics = seq_dynamics[id_][:6]
                            dt = frame_id - seq_dynamics[id_][-1]
                            dynamics = update_dynamics_2d(dynamics, location.astype(np.float32), seq_dynamics[id_][:2].copy(), 0.5, dt)
                            seq_dynamics[id_][:6] = dynamics
                            seq_dynamics[id_][-1] = frame_id
                        frame_dynamics[j] = torch.from_numpy(seq_dynamics[id_][:6]).to(torch.float32)

                        if frame_id >= self.block_size:  # Removing first self.block_size frames from each sequences.
                            if s not in self.instance_sequences[id_]:
                                self.instance_sequences[id_][s] = []
                            self.instance_sequences[id_][s].append(frame_id)
                            self._length += 1
                    data_sample.pred_track_instances.dynamics = frame_dynamics[:, 2:4]
            assert self.block_size < self._length, f"The specified block size ({self.block_size}) is bigger than the biggest sequence ({self._length})."
        else:
            data_list = self.load_data_list_unsupervized()
            for s, sequence in enumerate(data_list):
                for data_sample in sequence:
                    frame_id = data_sample.img_id
                    if frame_id >= self.block_size:
                        ids = data_sample.pred_track_instances.instances_id
                        for id_ in ids:
                            id_ = id_.item()
                            if s not in self.instance_sequences[id_]:
                                self.instance_sequences[id_][s] = []
                            self.instance_sequences[id_][s].append(frame_id)
                            self._length += 1
        self.instances = list(self.instance_sequences.keys())

        seq_counter = 0
        seq_lenghts = []
        for id_sequences in self.instance_sequences.values():
            for seq in id_sequences.values():
                seq_lenghts.append(len(seq))
                seq_counter += 1
        seq_lengths_mean = int(np.mean(seq_lenghts))
        seq_lenghts_median = int(np.median(seq_lenghts))
        self.logger.info(
            f"The dataset contains {seq_counter} sequences. Sequences mean length: {seq_lengths_mean}, sequences median length: {seq_lenghts_median}, length of all sequences: {self._length}."
        )
        return data_list

    def load_data_list_unsupervized(self):
        sequences = self.data_prefix["sequences"]
        self.logger.info(f"Loading sequences: {sequences}...")
        data_list = [[] for _ in sequences]
        for sequence_idx, sequence in tqdm(enumerate(sequences)):
            self.set_sequence_transforms()
            vid_reader = VideoReader(sequence)
            seq_length = len(vid_reader)

            assigner = AssociationStep(**self.assigner)
            result = Result(self.tracking_outputs)
            results = batch_tracking(
                video=vid_reader,
                detector=self.detector,
                batch_size=self.tracking_batch_size,
                result=result,
                association_step=assigner,
                validator=None,
                analyzer=None,
                verbose=True,
            )
            for i in range(seq_length):
                pred_track_instances = Dict()
                for o in results.outputs:
                    frame_output = torch.tensor(o[i])
                    if len(frame_output) > 0:
                        if o.__class__.__name__ == self.UNSUP_MANDATORY_OUTPUTS[0]:
                            pred_track_instances.labels = frame_output[:, 1]
                            pred_track_instances.instances_id = frame_output[:, 2]
                        elif o.__class__.__name__ == self.UNSUP_MANDATORY_OUTPUTS[1]:
                            pred_track_instances.dynamics = frame_output[:, 3:5]
                        elif o.__class__.__name__ == self.UNSUP_MANDATORY_OUTPUTS[2]:
                            frame_output = frame_output[:, 3:]
                            num_kpts = frame_output.shape[1] // 3
                            frame_output = frame_output.view(-1, num_kpts, 3)
                            pred_track_instances.kpts = frame_output[..., :2]
                            pred_track_instances.kpt_vis = frame_output[..., 2]
                        elif o.__class__.__name__ == self.UNSUP_MANDATORY_OUTPUTS[3]:
                            pred_track_instances.features = frame_output
                    else:
                        pred_track_instances.instances_id = frame_output

                light_ds = PoseDataSample()
                light_ds.pred_track_instances = pred_track_instances
                light_ds.img_id = i
                light_ds.seq_id = sequence_idx

                data_list[sequence_idx].append(light_ds)

        return data_list

    def _init_data_list(self):
        self.data_list = self.load_data_list()

    def __len__(self):
        if self._custom_length > 0:
            return self._custom_length
        return self._length


@DATASETS.register_module()
class ReIDDataset(OfflineRandomSequenceDataset):

    def __init__(
        self,
        from_file: str,
        detector: Config,
        assigner: Config,
        n_feats: int,
        bboxes_gt_format: Optional[str] = "CsvBoundingBoxes",
        keypoints_gt_format: Optional[str] = "CsvKeypoints",
        data_root: Optional[str] = ".",
        data_prefix: dict = dict(
            sequences=["."],
            bboxes_gt_paths=[""],
            keypoints_gt_paths=[""],
        ),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        block_size: Optional[int] = 2,
        *args,
        **kwargs,
    ):
        self.n_feats = n_feats
        self.id_feats = defaultdict(list)
        super().__init__(
            from_file=from_file,
            detector=detector,
            assigner=assigner,
            bboxes_gt_format=bboxes_gt_format,
            keypoints_gt_format=keypoints_gt_format,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            block_size=block_size,
        )

    def prepare_data(self, idx):
        id_features = self.id_feats[idx]
        choices = np.arange(len(id_features))
        if len(choices) > 1:
            id_query, id_anchor = tuple(np.random.choice(choices, size=2, replace=False))
            anchor = self._prepare_features(id_features[id_anchor])
        else:
            id_query = choices[0]
            anchor = None

        query = self._prepare_features(id_features[id_query])

        ds = PoseDataSample()
        ds.gt_instance_labels = Dict()
        ds.gt_instance_labels.anchor = anchor
        ds.gt_instance_labels.id = idx

        return dict(inputs=query, data_samples=ds)

    def _prepare_features(self, features):
        features = features.view(-1, self.n_feats)
        random_topk_feat = np.random.randint(0, features.size(0))
        return noisify(features[random_topk_feat])

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        # id_feats = list()

        unique_id_map = dict()
        for sequence in data_list:
            # seen_ids = []
            for data_sample in sequence:
                features = data_sample.pred_track_instances.features

                del data_sample.pred_track_instances
                del data_sample.gt_instance_labels
                del data_sample.gt_instances

                for inst_id, feats in features.items():
                    # if inst_id in seen_ids:
                    #     split = 1
                    # else:
                    #     split = 0
                    #     seen_ids.append(inst_id)
                    unique_id = f"{data_sample.seq_id}_{inst_id}"
                    if unique_id not in unique_id_map:
                        unique_id_map[unique_id] = len(unique_id_map)
                    self.id_feats[unique_id_map[unique_id]].append(feats)
                    # id_feats.append((unique_id_map[unique_id], split, feats))

        return list(self.id_feats.keys())

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_list)


@DATASETS.register_module()
class VideoDataset:
    def __init__(self, video_paths, gt_paths):
        self.video_paths = infer_paths(video_paths)
        self.gt_paths = infer_paths(gt_paths)

        assert len(self.gt_paths) == len(self.video_paths)
        gt_r_map = []
        base_v_paths = [os.path.splitext(os.path.basename(v))[0] for v in self.video_paths]
        for i, gt in enumerate(self.gt_paths):
            assert os.path.exists(gt)
            gt_name = os.path.splitext(os.path.basename(gt))[0].replace("gt_", "")
            gt_idx = find_path_in_dir(gt_name, base_v_paths)
            if gt_idx == -1:
                print_log(
                    (
                        f"The ground truth file: {gt_name} have no corresponding video, meaning a video file with the same name,"
                        f"in the provided video paths: {base_v_paths}."
                    ),
                    logger="current",
                    level=WARNING,
                )
            else:
                gt_r_map.append((i, gt_idx))

        self.data_list = gt_r_map

    def __getitem__(self, idx: int) -> dict:
        gt_idx, v_gt_idx = self.data_list[idx]
        return dict(inputs=self.video_paths[v_gt_idx], data_samples=self.gt_paths[gt_idx])

    def __len__(self) -> int:
        return len(self.data_list)
