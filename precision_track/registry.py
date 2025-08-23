import os

from mmengine.registry import DATASETS, HOOKS, LOOPS, METRICS, MODELS, PARAM_SCHEDULERS, TASK_UTILS, TRANSFORMS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import Registry

OUTPUTS = Registry("output", locations=["precision_track.outputs"], scope="mmengine")
TRACKING = Registry("tracking", locations=["precision_track.tracking"], scope="mmengine")
KEYPOINT_CODECS = Registry("codecs", locations=["precision_track.datasets.codecs"], scope="mmengine")
VISUALIZERS = Registry("visualizer", parent=MMENGINE_VISUALIZERS, locations=["precision_track.visualization"], scope="mmengine")
CODEBASE = Registry("Codebases", scope="mmengine")
RUNTIMES = Registry("runtime", locations=["precision_track.models.runtimes"], scope="mmengine")
BACKENDS = Registry("backend", locations=["precision_track.models.backends"], scope="mmengine")

DATASETS._locations.append("precision_track.datasets")
MODELS._locations.append("precision_track.models")
MODELS._locations.append("precision_track.apis")
TASK_UTILS._locations.append("precision_track.models.priors")
TASK_UTILS._locations.append("precision_track.utils.formatting")
METRICS._locations.append("precision_track.evaluation.metrics")
HOOKS._locations.append("precision_track.hooks")
LOOPS._locations.append("precision_track.loops")
TRANSFORMS._locations.append("precision_track.datasets.transforms")
PARAM_SCHEDULERS._locations.append("precision_track.models")

ROOT = os.path.dirname(os.path.abspath(__file__))
