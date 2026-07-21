# Canting API

`api/` exposes a small HTTP server on top of the existing 9-step image pipeline.

## Behavior

- accepts a single `image_path`
- runs all 9 existing pipeline steps in order
- returns only the Step 09 overlay plus total processing time
- does not persist anything to a database
- leaves a persistence stub in place for later implementation

Each request runs inside its own temporary workspace under `api/.runtime/jobs/...`, so it does not overwrite the shared `data/processed` tree.

## Start

Use any Python interpreter that already has the pipeline dependencies installed (`opencv-python`, `numpy`, `PyYAML`):

```powershell
python -m api --host 127.0.0.1 --port 8000
```

## Endpoints

### `GET /health`

Simple health check.

### `POST /analyze`

Request body:

```json
{
  "image_path": "data/working_png/IMG_0502.png",
  "response_mode": "json",
  "keep_artifacts": false,
  "include_step_logs": false
}
```

Supported `response_mode` values:

- `json`: returns `processing_time_ms` and `overlay_data_url`
- `binary`: returns only the Step 09 overlay as `image/png`; processing time is in the `X-Processing-Time-Ms` header

JSON response example:

```json
{
  "image_name": "IMG_0502.png",
  "input_image_path": "C:\\Users\\panonit\\Documents\\ml-ski-boot-canting\\data\\working_png\\IMG_0502.png",
  "processing_time_ms": 1432.77,
  "overlay_data_url": "data:image/png;base64,...",
  "artifacts_dir": null,
  "overlay_output_path": null,
  "metadata_output_path": null,
  "persistence": {
    "saved": false,
    "backend": "noop",
    "message": "Persistence is intentionally disabled for now."
  }
}
```

## Notes

- input currently supports `.png`, `.jpg`, and `.jpeg`
- `keep_artifacts: true` keeps the temporary job folder so you can inspect intermediate outputs later
- the pipeline config is injected per request through `PIPELINE_CONFIG`
