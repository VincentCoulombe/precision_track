import argparse
import os
import os.path
import pickle
import re
import time
import warnings
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from deeplabcut.core import inferenceutils, trackingutils
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.core.openvino.session import GetPoseF_OV, is_openvino_available
from deeplabcut.pose_estimation_tensorflow.predict_videos import convert_detections2tracklets
from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.utils import auxfun_models, auxfun_multianimal, auxiliaryfunctions
from scipy.optimize import linear_sum_assignment
from skimage.util import img_as_ubyte
from tabulate import tabulate
from tqdm import tqdm


def display_latency(times: np.ndarray, title, buffer_size=5, precision=3):
    assert len(times) >= buffer_size
    times = times[buffer_size:]
    mean = np.mean(times)
    table = [
        ["Mean", mean],
        ["Std", np.std(times)],
        ["Min", np.min(times)],
        ["Median", np.median(times)],
        ["Max", np.max(times)],
    ]
    print(
        "\n"
        + tabulate(
            table,
            headers=[title, "Inference Time (s)"],
            tablefmt="github",
            floatfmt=f".{precision}f",
            stralign="left",
        )
    )
    return mean


def analyze_videos(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    in_random_order=True,
    destfolder=None,
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    modelprefix="",
    robust_nframes=False,
    allow_growth=False,
    use_shelve=False,
    auto_track=True,
    n_tracks=None,
    animal_names=None,
    calibrate=False,
    identity_only=False,
    use_openvino="CPU" if is_openvino_available else None,
):
    """Makes prediction based on a trained network.

    The index of the trained network is specified by parameters in the config file
    (in particular the variable 'snapshotindex').

    The labels are stored as MultiIndex Pandas Array, which contains the name of
    the network, body part name, (x, y) label position in pixels, and the
    likelihood for each frame per body part. These arrays are stored in an
    efficient Hierarchical Data Format (HDF) in the same directory where the video
    is stored. However, if the flag save_as_csv is set to True, the data can also
    be exported in comma-separated values format (.csv), which in turn can be
    imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config: str
        Full path of the config.yaml file.

    videos: list[str]
        A list of strings containing the full paths to videos for analysis or a path to
        the directory, where all the videos with same extension are stored.

    videotype: str, optional, default=""
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed. If left unspecified,
        videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle: int, optional, default=1
        An integer specifying the shuffle index of the training dataset used for
        training the network.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int or None, optional, default=None
        Indicates the GPU to use (see number in ``nvidia-smi``). If you do not have a
        GPU put ``None``.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional, default=False
        Saves the predictions in a .csv file.

    in_random_order: bool, optional (default=True)
        Whether or not to analyze videos in a random order.
        This is only relevant when specifying a video directory in `videos`.

    destfolder: string or None, optional, default=None
        Specifies the destination folder for analysis data. If ``None``, the path of
        the video is used. Note that for subsequent analysis this folder also needs to
        be passed.

    batchsize: int or None, optional, default=None
        Change batch size for inference; if given overwrites value in ``pose_cfg.yaml``.

    cropping: list or None, optional, default=None
        List of cropping coordinates as [x1, x2, y1, y2].
        Note that the same cropping parameters will then be used for all videos.
        If different video crops are desired, run ``analyze_videos`` on individual
        videos with the corresponding cropping coordinates.

    TFGPUinference: bool, optional, default=True
        Perform inference on GPU with TensorFlow code. Introduced in "Pretraining
        boosts out-of-domain robustness for pose estimation" by Alexander Mathis,
        Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis.
        Source: https://arxiv.org/abs/1909.11229

    dynamic: tuple(bool, float, int) triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal).

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    robust_nframes: bool, optional, default=False
        Evaluate a video's number of frames in a robust manner.
        This option is slower (as the whole video is read frame-by-frame),
        but does not rely on metadata, hence its robustness against file corruption.

    allow_growth: bool, optional, default=False.
        For some smaller GPUs the memory issues happen. If ``True``, the memory
        allocator does not pre-allocate the entire specified GPU memory region, instead
        starting small and growing as needed.
        See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2

    use_shelve: bool, optional, default=False
        By default, data are dumped in a pickle file at the end of the video analysis.
        Otherwise, data are written to disk on the fly using a "shelf"; i.e., a
        pickle-based, persistent, database-like object by default, resulting in
        constant memory footprint.

    The following parameters are only relevant for multi-animal projects:

    auto_track: bool, optional, default=True
        By default, tracking and stitching are automatically performed, producing the
        final h5 data file. This is equivalent to the behavior for single-animal
        projects.

        If ``False``, one must run ``convert_detections2tracklets`` and
        ``stitch_tracklets`` afterwards, in order to obtain the h5 file.

    This function has 3 related sub-calls:

    identity_only: bool, optional, default=False
        If ``True`` and animal identity was learned by the model, assembly and tracking
        rely exclusively on identity prediction.

    calibrate: bool, optional, default=False
        If ``True``, use training data to calibrate the animal assembly procedure. This
        improves its robustness to wrong body part links, but requires very little
        missing data.

    n_tracks: int or None, optional, default=None
        Number of tracks to reconstruct. By default, taken as the number of individuals
        defined in the config.yaml. Another number can be passed if the number of
        animals in the video is different from the number of animals the model was
        trained on.

    animal_names: list[str], optional
        If you want the names given to individuals in the labeled data file, you can
        specify those names as a list here. If given and `n_tracks` is None, `n_tracks`
        will be set to `len(animal_names)`. If `n_tracks` is not None, then it must be
        equal to `len(animal_names)`. If it is not given, then `animal_names` will
        be loaded from the `individuals` in the project config.yaml file.

    use_openvino: str, optional
        Use "CPU" for inference if OpenVINO is available in the Python environment.

    Returns
    -------
    DLCScorer: str
        the scorer used to analyze the videos

    Examples
    --------

    Analyzing a single video on Windows

    >>> deeplabcut.analyze_videos(
            'C:\\myproject\\reaching-task\\config.yaml',
            ['C:\\yourusername\\rig-95\\Videos\\reachingvideo1.avi'],
        )

    Analyzing a single video on Linux/MacOS

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/reachingvideo1.avi'],
        )

    Analyze all videos of type ``avi`` in a folder

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos'],
            videotype='.avi',
        )

    Analyze multiple videos

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
        )

    Analyze multiple videos with ``shuffle=2``

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
            shuffle=2,
        )

    Analyze multiple videos with ``shuffle=2``, save results as an additional csv file

    >>> deeplabcut.analyze_videos(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
            shuffle=2,
            save_as_csv=True,
        )
    """
    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        auxfun_models.set_visible_devices(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    iteration = cfg["iteration"]

    if cropping is not None:
        cfg["cropping"] = True
        cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"] = cropping
        print("Overwriting cropping parameters:", cropping)
        print("These are used for all videos, but won't be save to the cfg file.")

    modelfolder = os.path.join(
        cfg["project_path"],
        str(auxiliaryfunctions.get_model_folder(trainFraction, shuffle, cfg, modelprefix=modelprefix)),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for iteration %s and shuffle %s and trainFraction %s does not exist." % (iteration, shuffle, trainFraction)
        )

    Snapshots = auxiliaryfunctions.get_snapshots_from_folder(
        train_folder=Path(modelfolder) / "train",
    )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg["init_weights"] = os.path.join(modelfolder, "train", Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
    # Update number of output and batchsize
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))

    if batchsize is None:
        # update batchsize (based on parameters in config.yaml)
        dlc_cfg["batch_size"] = cfg["batch_size"]
    else:
        dlc_cfg["batch_size"] = batchsize
        cfg["batch_size"] = batchsize

    if "multi-animal" in dlc_cfg["dataset_type"]:
        dynamic = (False, 0.5, 10)  # setting dynamic mode to false
        TFGPUinference = False

    if dynamic[0]:  # state=true
        # (state,detectiontreshold,margin)=dynamic
        print("Starting analysis in dynamic cropping mode with parameters:", dynamic)
        dlc_cfg["num_outputs"] = 1
        TFGPUinference = False
        dlc_cfg["batch_size"] = 1
        print("Switching batchsize to 1, num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode).")

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )
    if dlc_cfg["num_outputs"] > 1:
        if TFGPUinference:
            print("Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently.")
            TFGPUinference = False
        print("Extracting ", dlc_cfg["num_outputs"], "instances per bodypart")
        xyz_labs_orig = ["x", "y", "likelihood"]
        suffix = [str(s + 1) for s in range(dlc_cfg["num_outputs"])]
        suffix[0] = ""  # first one has empty suffix for backwards compatibility
        xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]
    else:
        xyz_labs = ["x", "y", "likelihood"]

    if use_openvino:
        sess, inputs, outputs = predict.setup_openvino_pose_prediction(dlc_cfg, device=use_openvino)
    elif TFGPUinference:
        sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_cfg, allow_growth=allow_growth)
    else:
        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg, allow_growth=allow_growth)

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype, in_random_order)
    if len(Videos) > 0:
        if "multi-animal" in dlc_cfg["dataset_type"]:
            from deeplabcut.pose_estimation_tensorflow.predict_multianimal import AnalyzeMultiAnimalVideo

            times = []
            for video in Videos:
                t0 = perf_counter()
                AnalyzeMultiAnimalVideo(
                    video,
                    DLCscorer,
                    trainFraction,
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    destfolder,
                    robust_nframes=robust_nframes,
                    use_shelve=use_shelve,
                )
                if auto_track:  # tracker type is taken from default in cfg
                    convert_detections2tracklets(
                        config,
                        [video],
                        videotype,
                        shuffle,
                        trainingsetindex,
                        destfolder=destfolder,
                        modelprefix=modelprefix,
                        calibrate=calibrate,
                        identity_only=identity_only,
                    )
                    stitch_tracklets(
                        config,
                        [video],
                        videotype,
                        shuffle,
                        trainingsetindex,
                        destfolder=destfolder,
                        n_tracks=n_tracks,
                        animal_names=animal_names,
                        modelprefix=modelprefix,
                        save_as_csv=save_as_csv,
                    )
                time = perf_counter() - t0
                cap = cv2.VideoCapture(video)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                times.append(frame_count / time)
        else:
            raise NotImplementedError(
                "For accountability reason, we recommend using the official DLC implementation for use cases other than compairing on multi-animal tracking."
            )

        os.chdir(str(start_path))
        display_latency(times, "FPS", buffer_size=0)
        return DLCscorer, times
    else:
        print("No video(s) were found. Please check your paths and/or 'video_type'.")
        return DLCscorer
