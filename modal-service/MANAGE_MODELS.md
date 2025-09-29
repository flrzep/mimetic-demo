# Managing Models and Config in Modal Volume

This guide shows how to update files (models and `model_config.json`) in the Modal volume using the `manage_models.py` script.

## Prerequisites

- Windows PowerShell
- Python available via `py`
- Modal Python package installed (already used by the script)
- Modal CLI auth (one-time):

```powershell
py -m modal token set
```

## Navigate to the script

From the repo root:

```powershell
cd modal-service
```

## Update ONLY the model config

Run this after editing `modal-service/model_config.json` locally:

```powershell
py -m modal run manage_models.py::upload_model_config
```

## Upload all local model files

Sync everything under `modal-service/models` to the Modal volume:

```powershell
py -m modal run manage_models.py::upload_local_models
```

## Full setup (models + config + validation)

Uploads the local model folders, uploads the config, and validates that config matches the files:

```powershell
py -m modal run manage_models.py::setup_modal_storage
```

## Verify what’s in the volume

List all model folders and files present in the Modal volume:

```powershell
py -m modal run manage_models.py::list_models
```

## Validate configuration against files

Checks that each configured model in `model_config.json` has the expected files present in the volume:

```powershell
py -m modal run manage_models.py::validate_model_config
```

## Optional helpers

- Download YOLO weights into the volume:

```powershell
py -m modal run manage_models.py::download_yolo_model
```

- Download a custom model from Hugging Face:

```powershell
py -m modal run manage_models.py::download_custom_model --repo-id "owner/repo" --filename "path/in/repo/model.onnx" --model-name "my-model"
```

- Remove a model folder from the volume:

```powershell
py -m modal run manage_models.py::remove_model --model-name "yolo"
```

## Notes

- The script uses the shared Modal volume named `mimetic-models` and aligns the app name with your branch.
- Re-run the relevant upload function after local file changes to sync them to the volume.
- `upload_local_models` overwrites existing files with your local copies. If you want to start clean for a model, `remove_model` first, then upload again.
