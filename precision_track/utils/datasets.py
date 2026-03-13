# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Mapping, Sequence
import json
import os
import os.path as osp
import textwrap

import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.logging import print_log
from tqdm import tqdm

from mmengine.registry import FUNCTIONS
from .io import SUPPORTED_IMG_BACKEND, SUPPORTED_VIDEO_BACKEND


def parse_pose_metainfo(metainfo: dict):
    if "from_file" in metainfo:
        cfg_file = metainfo["from_file"]
        if not osp.isfile(cfg_file):
            raise FileNotFoundError(f'The metainfo config file "{cfg_file}" does not exist.')
        metainfo = Config.fromfile(cfg_file).dataset_info
    else:
        cfg_file = ""

    # check data integrity
    assert "dataset_name" in metainfo
    assert "keypoint_info" in metainfo
    assert "skeleton_info" in metainfo
    assert "joint_weights" in metainfo
    assert "sigmas" in metainfo
    assert "classes" in metainfo

    # parse metainfo
    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed["actions"] = metainfo.get("actions", [])
    assert isinstance(parsed["actions"], list), f"Your metadata file: {os.path.abspath(cfg_file)} must have a list of actions as value to the 'actions' key."

    classes = metainfo["classes"]
    assert isinstance(
        classes, list
    ), f"Your metadata file: {os.path.abspath(cfg_file)} must have a list of classes, or species, as value to the 'classes' key."
    nb_classes = len(metainfo["classes"])
    assert nb_classes > 0, f"Your metadata file: {os.path.abspath(cfg_file)} must have at least one class, or species, listed as value to the 'classes' key."
    parsed["classes"] = metainfo["classes"]

    parsed["dataset_name"] = metainfo["dataset_name"]

    # parse keypoint information
    parsed["num_keypoints"] = len(metainfo["keypoint_info"])

    keypoint_names = []
    swapped_keypoints = []

    if parsed["num_keypoints"] > 0:
        for kpt_id, kpt in enumerate(metainfo["keypoint_info"]):
            assert "name" in kpt, f"The keypoints in your metadata file: {os.path.abspath(cfg_file)} must all have values attached to the 'name' key."
            kpt_name = kpt["name"]
            assert kpt_name not in keypoint_names, f"Two or more keypoints in your metadata file: {os.path.abspath(cfg_file)} have the same name ({kpt_name})"
            keypoint_names.append(kpt_name)
            parsed["keypoint_id2name"][kpt_id] = kpt_name
            parsed["keypoint_name2id"][kpt_name] = kpt_id
            parsed["keypoint_colors"].append(kpt.get("color", [0, 0, 0]))

            swap_kpt = kpt.get("swap", "")
            if swap_kpt == "":
                parsed["flip_indices"].append(kpt_name)
            else:
                assert (
                    swap_kpt not in swapped_keypoints
                ), f"Two or more keypoints in your metadata file: {os.path.abspath(cfg_file)} swap with the following keypoint: {kpt_name}"
                swapped_keypoints.append(swap_kpt)
                parsed["flip_indices"].append(swap_kpt)
                pair = (swap_kpt, kpt_name)
                if pair not in parsed["flip_pairs"]:
                    parsed["flip_pairs"].append(pair)
    else:  # For not breaking the data transformation pipelines when only training for object-detection, need at least 1 dummy kpts.
        parsed["keypoint_id2name"][0] = "dummy"
        parsed["keypoint_name2id"]["dummy"] = 0
        parsed["flip_indices"].append("dummy")
        parsed["num_keypoints"] = 1

    # parse skeleton information
    parsed["num_skeleton_links"] = len(metainfo["skeleton_info"])
    for sk in metainfo["skeleton_info"]:
        link = sk["link"]
        assert (
            len(link) == 2
        ), f"All the skeleton links in your metadata file ({os.path.abspath(cfg_file)}), must contain only teo keypoints (a source and a destination)."
        for node in link:
            assert (
                node in keypoint_names
            ), f"One of the skeleton link in your metadata file ({os.path.abspath(cfg_file)}) have a node that is not listed as a keypoint ({node})."
        parsed["skeleton_links"].append(sk["link"])
        parsed["skeleton_link_colors"].append(sk.get("color", [0, 0, 0]))

    # parse extra information
    parsed["dataset_keypoint_weights"] = np.array(metainfo["joint_weights"], dtype=np.float32)
    parsed["sigmas"] = np.array(metainfo["sigmas"], dtype=np.float32)

    if "stats_info" in metainfo:
        parsed["stats_info"] = {}
        for name, val in metainfo["stats_info"].items():
            parsed["stats_info"][name] = np.array(val, dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed["flip_pairs"] = _map(parsed["flip_pairs"], mapping=parsed["keypoint_name2id"])
    parsed["flip_indices"] = _map(parsed["flip_indices"], mapping=parsed["keypoint_name2id"])
    parsed["skeleton_links"] = _map(parsed["skeleton_links"], mapping=parsed["keypoint_name2id"])

    parsed["keypoint_colors"] = np.array(parsed["keypoint_colors"], dtype=np.uint8)
    parsed["skeleton_link_colors"] = np.array(parsed["skeleton_link_colors"], dtype=np.uint8)

    parsed["orientation_keypoint_pairs"] = metainfo.get("orientation_keypoint_pairs", [])
    parsed["distance_keypoint_pairs"] = metainfo.get("distance_keypoint_pairs", [])

    return parsed


def find_path_in_dir(path: str, dir_: list):
    found = False
    name1 = osp.splitext(osp.basename(path))[0]
    for i, file in enumerate(dir_):
        if not isinstance(file, str):
            continue
        name2 = osp.splitext(osp.basename(file))[0]
        if name1 == name2:
            found = True
            break
    if not found:
        return -1
    return i


def noisify(tensor: torch.Tensor, intensity=0.01):
    noise = torch.randn(tensor.size(), dtype=tensor.dtype, device=tensor.device) * intensity
    return tensor + (tensor * noise)


@FUNCTIONS.register_module()
def pseudo_collate_sequences(data_batch: Sequence) -> Any:
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, "_fields"):
        return data_item_type(*(pseudo_collate_sequences(samples) for samples in zip(*data_batch)))
    elif isinstance(data_item, Mapping):
        return data_item_type({key: pseudo_collate_sequences([d[key] for d in data_batch]) for key in data_item})
    else:
        return data_batch


def assert_coco_dataset_directory(coco_path):
    directory_tree = textwrap.dedent(
        """
        <data_root>/
        ├── annotations/
        │   ├── train.json
        │   ├── val.json
        ├── images/
        │   ├── image_1.jpg   # Images may have any filename.
        │   ├── image_2.jpg
        │   ├── ...
        """
    ).strip()

    coco_path = os.path.abspath(coco_path)
    if not os.path.isdir(coco_path):
        raise NotADirectoryError(f"Dataset root does not exist or is not a directory: {coco_path}")

    ann_dir = os.path.join(coco_path, "annotations")
    img_dir = os.path.join(coco_path, "images")

    if not os.path.isdir(ann_dir):
        raise NotADirectoryError(
            f"Missing 'annotations' directory in: {coco_path}. " f"For reference, your COCO-style dataset must take the following form:\n{directory_tree}"
        )
    if not os.path.isdir(img_dir):
        raise NotADirectoryError(
            f"Missing 'images' directory in: {coco_path}. " f"For reference, your COCO-style dataset must take the following form:\n{directory_tree}"
        )

    train_json = os.path.join(ann_dir, "train.json")
    val_json = os.path.join(ann_dir, "val.json")

    combined_json = os.path.join(ann_dir, "combined.json")

    if not os.path.isfile(train_json):
        if os.path.isfile(combined_json):
            print_log(
                logger="current",
                msg=(
                    f"Missing 'train.json' in: {ann_dir}, but a 'combined.json' file is found. "
                    f"Therefore, the system will create new 'train.json' and 'val.json' files "
                    f"by splitting the found 'combined.json' file."
                ),
            )
            split_combined_ann_file(ann_dir)
        else:
            raise FileNotFoundError(
                f"Missing 'train.json' in: {ann_dir}. " f"For reference, your COCO-style dataset must take the following form:\n{directory_tree}"
            )
    if not os.path.isfile(val_json):
        if os.path.isfile(combined_json):
            print_log(
                logger="current",
                msg=(
                    f"Missing 'val.json' in: {ann_dir}, but a 'combined.json' file is found. "
                    f"Therefore, the system will create new 'train.json' and 'val.json' files "
                    f"by splitting the found 'combined.json' file."
                ),
            )
            split_combined_ann_file(ann_dir)
        else:
            raise FileNotFoundError(
                f"Missing 'val.json' in: {ann_dir}. " f"For reference, your COCO-style dataset must take the following form:\n{directory_tree}"
            )

    # Check if the number of images in img_dir matches the number in annotation files
    image_files = set()
    for file in os.listdir(img_dir):
        extension = os.path.splitext(file)[1].lower()
        if extension in SUPPORTED_IMG_BACKEND:
            image_files.add(file)

    num_images_in_dir = len(image_files)

    with open(train_json, "r") as f:
        train_data = json.load(f)
    with open(val_json, "r") as f:
        val_data = json.load(f)

    train_image_filenames = set(img["file_name"] for img in train_data.get("images", []))
    val_image_filenames = set(img["file_name"] for img in val_data.get("images", []))

    overlapping_images = train_image_filenames.intersection(val_image_filenames)
    assert len(overlapping_images) == 0, f"Found {len(overlapping_images)} image(s) in both train and validation files: " f"{list(overlapping_images)}"

    num_images_in_train = len(train_image_filenames)
    num_images_in_val = len(val_image_filenames)
    num_images_in_annotations = len(train_image_filenames.union(val_image_filenames))

    if num_images_in_dir != num_images_in_annotations:
        print_log(
            logger="current",
            msg=(
                f"Mismatch between the number of images in the {img_dir} directory "
                f"and the number of images in the annotation files in the {ann_dir} directory:\n"
                f"  - Images in the {img_dir} directory: {num_images_in_dir}\n"
                f"  - Images in {ann_dir}/train.json file: {num_images_in_train}\n"
                f"  - Images in {ann_dir}/val.json file: {num_images_in_val}\n"
                f"  - Total unique images in annotations: {num_images_in_annotations}"
            ),
            level="WARNING",
        )

        # Check for images in directory but not in annotations
        missing_in_annotations = image_files - train_image_filenames - val_image_filenames
        if missing_in_annotations:
            print_log(
                logger="current",
                msg=f"These images are in the {img_dir} directory, but not in any of the annotation files: {sorted(missing_in_annotations)}",
                level="WARNING",
            )

        # Check for images in annotations but not in directory
        missing_in_directory = (train_image_filenames.union(val_image_filenames)) - image_files
        if missing_in_directory:
            print_log(
                logger="current",
                msg=f"These images are in the annotation files, but not in the {img_dir} image directory: {sorted(missing_in_directory)}",
                level="WARNING",
            )


def split_combined_ann_file(ann_dir, val_split=0.2):
    combined_path = os.path.join(ann_dir, "combined.json")

    with open(combined_path, "r") as f:
        combined_data = json.load(f)

    images = combined_data.get("images", [])
    annotations = combined_data.get("annotations", [])

    image_ids = [img["id"] for img in images]

    np.random.seed(42)
    np.random.shuffle(image_ids)

    val_count = int(len(image_ids) * val_split)
    val_image_ids = set(image_ids[:val_count])
    train_image_ids = set(image_ids[val_count:])

    train_images = [img for img in images if img["id"] in train_image_ids]
    val_images = [img for img in images if img["id"] in val_image_ids]

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": combined_data.get("categories", []),
        "info": combined_data.get("info", []),
        "licenses": combined_data.get("licenses", []),
    }

    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": combined_data.get("categories", []),
        "info": combined_data.get("info", []),
        "licenses": combined_data.get("licenses", []),
    }

    train_path = os.path.join(ann_dir, "train.json")
    val_path = os.path.join(ann_dir, "val.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=4)

    print_log(f"Split complete: {len(train_images)} train images, {len(val_images)} val images")
    print_log(f"Created {train_path} and {val_path}")


def resize_coco_dataset(coco_path, output_path, target_size=(640, 640), ann_name="train.json", pad_value=114):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)

    with open(os.path.join(coco_path, "annotations", ann_name), "r") as f:
        coco_data = json.load(f)

    if isinstance(pad_value, tuple):
        assert len(pad_value) == 3
    else:
        pad_value = (int(pad_value), int(pad_value), int(pad_value))

    width_target, height_target = target_size

    transforms = dict()

    for img in tqdm(coco_data["images"], desc="Resizing images"):
        img_path = os.path.join(coco_path, "images", img["file_name"])
        new_img_path = os.path.join(output_path, "images", img["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"An image from your {ann_name} file is either not found or unreadable: {os.path.abspath(img_path)}")

        height_orig, width_orig = image.shape[:2]

        max_dim = max(width_orig, height_orig)
        pad_x = (max_dim - width_orig) // 2
        pad_y = (max_dim - height_orig) // 2

        padded_image = np.full((max_dim, max_dim, 3), pad_value, dtype=image.dtype)
        padded_image[pad_y : pad_y + height_orig, pad_x : pad_x + width_orig] = image

        scale_x = width_target / max_dim
        scale_y = height_target / max_dim

        transforms[int(img["id"])] = (pad_x, pad_y, scale_x, scale_y)

        resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_img_path, resized_image)

        img["width"] = width_target
        img["height"] = height_target

    for ann in tqdm(coco_data["annotations"], desc="Resizing annotations"):
        transform = transforms.get(int(ann["image_id"]))
        assert transform is not None
        pad_x, pad_y, scale_x, scale_y = transform

        ann["bbox"][0] = (ann["bbox"][0] + pad_x) * scale_x
        ann["bbox"][1] = (ann["bbox"][1] + pad_y) * scale_y
        ann["bbox"][2] *= scale_x
        ann["bbox"][3] *= scale_y

        if "keypoints" in ann and len(ann["keypoints"]) > 0:
            for i in range(0, len(ann["keypoints"]), 3):
                ann["keypoints"][i] = (ann["keypoints"][i] + pad_x) * scale_x
                ann["keypoints"][i + 1] = (ann["keypoints"][i + 1] + pad_y) * scale_y

    with open(os.path.join(output_path, "annotations", ann_name), "w") as f:
        json.dump(coco_data, f, indent=4)


def check_if_mot_dataset_is_ok(dataset_root_dir: str):
    dataset_root_dir = os.path.abspath(dataset_root_dir)

    if not os.path.isdir(dataset_root_dir):
        return False, f"Your dataset root does not exist or is not a directory: {dataset_root_dir}"

    bboxes_dir = os.path.join(dataset_root_dir, "bboxes")
    videos_dir = os.path.join(dataset_root_dir, "videos")

    if not os.path.isdir(bboxes_dir):
        return False, "Your dataset does not contain a 'bboxes' subdirectory"

    if not os.path.isdir(videos_dir):
        return False, "Your dataset does not contain a 'videos' subdirectory"

    csv_stems = set()
    for root, _, files in os.walk(bboxes_dir):
        for fname in files:
            if fname.lower().endswith(".csv"):
                stem, _ = os.path.splitext(fname)
                csv_stems.add(stem)

    if not csv_stems:
        return False, f"Your {bboxes_dir} subdirectory does not contains any labels (.csv file)"

    video_stems = set()
    for root, _, files in os.walk(videos_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_VIDEO_BACKEND:
                stem, _ = os.path.splitext(fname)
                video_stems.add(stem)

    if not video_stems:
        return False, f"Your {videos_dir} subdirectory does not contains any videos"

    missing_videos = sorted(stem for stem in csv_stems if stem not in video_stems)
    if missing_videos:
        missing_str = ", ".join(missing_videos)
        return False, f"Your {videos_dir} subdirectory does not contains videos matching the {missing_str} .csv files."

    return True, _
