// Declarative description of user_configs.yaml, driving the Configure form.
//
// Field types: bool | text | number | path | num_subjects
// path fields carry:
//   picker: { mode: "file"|"dir", exts: [...] }
//   store:  "tools" (default, relative to tools/) | "data_root" | "basename"
//   baseField: dotted id of the directory this path is relative to (for the picker start + storage)
//   validate: true  -> POST /api/validate/<section.key> on blur

export const SCHEMA = [
  {
    id: "booleans",
    key: "booleans",
    title: "General on/off options",
    sub: "Enable or disable functionalities",
    fields: [
      { key: "pipelined", type: "bool", help: "Run processes in parallel for faster tracking." },
      { key: "with_validation", type: "bool", help: "Enable re-identification (validation)." },
      { key: "with_offline_correction_refinement", type: "bool", help: "Refine the timing of ID corrections after tracking. Needs validation on." },
      { key: "with_action_recognition", type: "bool", help: "Enable the MART model to recognize actions." },
      { key: "with_group_action_recognition", type: "bool", help: "Enable the GMART model for social actions. Needs action recognition on." },
      { key: "with_pose_estimation", type: "bool", help: "Enable full pose (keypoints + skeleton)." },
    ],
  },
  {
    id: "general",
    key: "general",
    title: "General directories and paths",
    fields: [
      {
        key: "metainfo",
        type: "path",
        help: "Python file describing your species (keypoints, skeleton, classes).",
        picker: { mode: "file", exts: [".py"] },
        validate: true,
      },
    ],
  },
  {
    id: "training",
    key: "training",
    title: "Training",
    sub: "Parameters, directories and paths",
    fields: [
      { key: "dataset_name", type: "text", help: "Label used in logs and run sub-directories." },
      { key: "data_root", type: "path", help: "Root of your COCO-style dataset.", picker: { mode: "dir" }, validate: true },
      { key: "resume", type: "bool", help: "Continue a stopped training from training_checkpoint." },
      {
        key: "training_checkpoint",
        type: "path",
        help: "Checkpoint to initialize training (transfer learning). Empty for scratch.",
        picker: { mode: "file", exts: [".pth"] },
        validate: true,
      },
      { key: "deploying_directory", type: "path", help: "Where deployment artifacts are saved. Created on save if missing.", picker: { mode: "dir" } },
      {
        key: "deploying_sanity_check_img_path",
        type: "path",
        help: "An image from your dataset (relative to data_root) to verify deployed checkpoints.",
        picker: { mode: "file", exts: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"] },
        store: "data_root",
        baseField: "training.data_root",
        validate: true,
      },
      { key: "batch_size", type: "number", min: 16, help: "Images processed at once. Must be at least 16.", validate: true },
      { key: "wandb_logging", type: "bool", help: "Enable Weights & Biases training visualization." },
    ],
  },
  {
    id: "tracking",
    key: "tracking",
    title: "Tracking",
    fields: [
      { key: "saving_directory", type: "path", help: "Where tracking outputs are written. Created on save if missing.", picker: { mode: "dir" } },
      { key: "num_subjects", type: "num_subjects", help: "How many subjects of each class are in the scene (-1 if animals enter/leave).", validate: true },
      {
        key: "tracking_checkpoint_name",
        type: "path",
        help: "Checkpoint name inside deploying_directory. Empty to auto-select.",
        picker: { mode: "file", exts: [".pth", ".onnx", ".engine"] },
        store: "basename",
        baseField: "training.deploying_directory",
        validate: true,
      },
      {
        key: "hyperparameters_file_name",
        type: "path",
        help: "Tracking hyperparameters file name inside deploying_directory.",
        picker: { mode: "file", exts: [".json"] },
        store: "basename",
        baseField: "training.deploying_directory",
        validate: true,
      },
      { key: "output_clustered_features", type: "bool", help: "Save clustered features (slows tracking down)." },
      { key: "mot_data_root", type: "path", help: "MOT dataset used to benchmark tracking.", picker: { mode: "dir" }, validate: true },
    ],
  },
  {
    id: "action_recognition",
    key: "action_recognition",
    title: "Action Recognition",
    gate: (c) => c.booleans?.with_action_recognition === true,
    fields: [
      {
        key: "mart_checkpoint_name",
        type: "path",
        help: "MART checkpoint name inside deploying_directory.",
        picker: { mode: "file", exts: [".pth", ".onnx", ".engine"] },
        store: "basename",
        baseField: "training.deploying_directory",
        validate: true,
      },
      { key: "action_recognition_data_root", type: "path", help: "Action-recognition dataset (MOT-style).", picker: { mode: "dir" }, validate: true },
      { key: "output_action_recognition_embeddings", type: "bool", help: "Save MART action embeddings (slows tracking down)." },
    ],
  },
  {
    id: "group_action_recognition",
    key: "group_action_recognition",
    title: "Group Action Recognition",
    gate: (c) => c.booleans?.with_action_recognition === true,
    fields: [
      {
        key: "gmart_checkpoint_name",
        type: "path",
        help: "GMART checkpoint name inside deploying_directory.",
        picker: { mode: "file", exts: [".pth", ".onnx", ".engine"] },
        store: "basename",
        baseField: "training.deploying_directory",
        validate: true,
      },
    ],
  },
  {
    id: "validation",
    key: "validation",
    title: "Validation",
    sub: "Re-identification",
    gate: (c) => c.booleans?.with_validation === true,
    fields: [
      {
        key: "validation_configuration_file",
        type: "path",
        help: "Validation config file (appearance or aruco). Use “Edit values” to configure it.",
        picker: { mode: "file", exts: [".yaml", ".yml"] },
        validate: true,
        editor: true,
      },
      { key: "output_appearance_database", type: "bool", help: "Save the appearance database (slows tracking down)." },
    ],
  },
  {
    id: "visualization",
    key: "visualization",
    title: "Visualization",
    fields: [
      { key: "display_bounding_boxes", type: "bool" },
      { key: "display_poses", type: "bool" },
      { key: "display_velocities", type: "bool" },
      { key: "display_species", type: "bool" },
      { key: "display_confidence_scores", type: "bool" },
      { key: "display_actions", type: "bool" },
      { key: "display_search_zones", type: "bool" },
      { key: "display_validations", type: "bool" },
      { key: "display_untracked_detections", type: "bool" },
      { key: "display_predicted_bounding_boxes", type: "bool" },
    ],
  },
];
