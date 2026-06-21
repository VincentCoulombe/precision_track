# PrecisionTrack Web UI

A small **local** web interface to configure PrecisionTrack (`configs/user_configs.yaml`)
and launch the tools in `tools/` — without hand-editing YAML or remembering CLI flags.

It is a single-user tool meant to run on your own machine, in the **same Python
environment as PrecisionTrack** (it imports `precision_track` to validate your
configuration exactly the way the tools will).

## Install

```bash
pip install -r requirements/web.txt
```

## Run

```bash
python -m web_ui            # serves http://127.0.0.1:8000 and opens your browser
python -m web_ui --port 8080 --no-browser
```

## What it does

- **Configure** — renders `user_configs.yaml` as a form. Every path/dataset field
  is validated *with PrecisionTrack's own utilities* (`parse_pose_metainfo`,
  `assert_coco_dataset_directory`, `check_if_mot_dataset_is_ok`, …). Problems show
  up as notifications. Path/directory fields use a file/folder picker. The
  Action-Recognition, Group-Action-Recognition and Validation panels appear only
  when their feature toggle is on.
  - Saving writes `user_configs.yaml` in place (comments and section banners are
    preserved; a timestamped `.bak` is created first) and creates the
    `deploying_directory` / `saving_directory` if they don't exist yet.
  - The **Edit values** button on `validation_configuration_file` opens an editor
    for the appearance / aruco config and the ReID identities.

- **Run** — pick a tool, set its flags, and watch its output stream live in an
  embedded terminal. One job runs at a time; you can stop it.

Nothing here changes how the tools run: the UI only writes `user_configs.yaml`,
which the tools load via `load_user_configs` exactly as before.
