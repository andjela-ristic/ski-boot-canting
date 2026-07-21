# Canting API

`api/` exposes a small HTTP server on top of the existing 9-step image pipeline.
The same server now also serves the web PWA from `/`.

## Behavior

- accepts a single `image_path`
- accepts a single `video_path` on `/frames` and samples multiple frames
- runs all 9 existing pipeline steps in order
- returns only the Step 09 overlay plus total processing time
- does not persist anything to a database
- leaves a persistence stub in place for later implementation

Each request runs inside its own temporary workspace under `api/.runtime/jobs/...`, so it does not overwrite the shared `data/processed` tree.

## Start

Install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Then start the API:

```powershell
python -m api --host 127.0.0.1 --port 8000
```

If `python` is not on `PATH`, run the same command with the full interpreter path.

## Docker

Build:

```powershell
docker build -t ski-boot-canting-api .
```

Run:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}:/app-host" ski-boot-canting-api
```

Then call the API with an image path visible inside the container, for example `/app-host/data/working_png/IMG_0502.png`.

## Docker Compose

Start:

```powershell
docker compose up --build
```

Compose mounts `./data` into the container as `/app/data`, so the same relative request path still works:

```json
{
  "image_path": "data/working_png/IMG_0502.png"
}
```

Stop:

```powershell
docker compose down
```

## Web PWA

Posle pokretanja servera:

- `GET /` otvara web PWA klijent
- `GET /api` vraca servisni indeks
- `GET /health` ostaje health endpoint

Ako sve vrtis sa iste instance servera, frontend i API su na istom origin-u pa ne treba dodatni app server.

## Docker And Mobile

Backend je isti za Android, iPhone i desktop browser. Bitna razlika je kako otvaras frontend:

- desktop browser: `http://127.0.0.1:8000`
- drugi uredjaj u LAN-u: `http://<LAN-IP-tvog-racunara>:8000`
- fizicki iPhone za live browser kameru: preporucen HTTPS tunnel ka istom serveru

Compose vec podize backend sa otvorenim CORS headerima (`API_CORS_ALLOW_ORIGIN=*`), ali kada frontend ide sa istog origin-a CORS vise nije presudan za glavni tok.

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

### `POST /frames`

Request body:

```json
{
  "video_path": "data/videos/sample.mp4",
  "keep_artifacts": false,
  "include_step_logs": false
}
```

Ili `multipart/form-data` upload sa web PWA klijenta:

- `video`: binarni fajl videa
- `clip_duration_ms`: npr. `2000`
- `frame_count`: npr. `6`
- `keep_artifacts`: `true|false`

Behavior:

- samples `6-10` frames from config field `api.frames.sample_count`
- extracts frames uniformly across the video
- analyzes all sampled frames in parallel
- keeps one execution slot reserved for regular `/analyze` requests
- returns the overlay and metadata path from the best sampled frame
- keeps per-frame analysis plus averaged numeric metadata tree in the response

Za `multipart` upload trenutno vraca jedan validan stub overlay da web PWA flow moze da se testira end-to-end iako finalna multi-frame fuzija jos nije implementirana.

Response shape:

```json
{
  "video_path": "C:\\Users\\panonit\\Documents\\ml-ski-boot-canting\\data\\videos\\sample.mp4",
  "frame_count": 6,
  "processing_time_ms": 8123.45,
  "selected_frame_index": 3,
  "selected_timestamp_ms": 1042.5,
  "frame_sampling": {
    "sample_count": 6,
    "max_workers": 4
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0.0,
      "analysis": {
        "image_name": "frame_00.png",
        "processing_time_ms": 1321.4,
        "overlay_data_url": "data:image/png;base64,..."
      },
      "metadata": {
        "...": "full step 09 metadata for that frame"
      }
    }
  ],
  "average_metadata": {
    "...": "recursive average over numeric metadata fields"
  },
  "artifacts_dir": null
}
```

## Notes

- input currently supports `.png`, `.jpg`, and `.jpeg`
- video input currently supports `.mp4`, `.mov`, `.avi`, `.mkv`, and `.m4v`
- `keep_artifacts: true` keeps the temporary job folder so you can inspect intermediate outputs later
- the pipeline config is injected per request through `PIPELINE_CONFIG`
