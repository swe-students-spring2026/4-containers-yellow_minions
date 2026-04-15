![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Sound Alert Assistant

Sound Alert Assistant is a containerized app that helps deaf and hard-of-hearing pedestrians notice important environmental sounds in uploaded audio or video.

## What it does

- lets a user upload an audio or video clip
- runs ML-based sound event detection in a worker container
- stores timestamped alerts in MongoDB
- shows playback overlays and dashboard analytics in a web app

## Project structure

- web-app: Flask UI for uploads and alert history
- machine-learning-client: background worker that analyzes audio
- mongo: database for jobs and predictions

## How the pieces fit together

1. The web app creates a job in `analysis_jobs`.
2. The ML worker picks up the job.
3. The worker analyzes the clip and stores timestamped detections in `predictions`.
4. The web app displays the latest alert, overlays, and history.

### Web-to-ML contract

- User uploads audio or video in the web app.
- The web app stores the original file and, if the upload is video, extracts an audio track.
- The job document sent to ML must include `audio_path`.
- Optional helper fields: `media_path`, `media_type`, `original_filename`, `duration_seconds`.
- The worker only needs `audio_path` to analyze the clip.

## Quick start

1. Copy `env.example` to `.env`.
2. Start the stack:
   - `docker compose up --build`
3. Open the web app at `http://localhost:5000`.

## ML behavior

- The worker uses a pretrained audio-classification model.
- It detects sound-event labels such as `siren`, `car_horn`, `bicycle`, `dog_bark`, and `alarm`.
- It also runs speech-to-text and stores timestamped captions plus a full transcript.
- It stores normalized labels, confidence scores, timestamp ranges, captions, transcripts, and status updates in MongoDB.

## Useful collections

- `analysis_jobs`: queued clips waiting to be analyzed
- `predictions`: ML output and metadata

## ML output fields

- `alert_type`, `alert_confidence`, `category`, `confidence`
- `detections`: timestamped sound-event ranges
- `captions`: timestamped speech-to-text overlays
- `full_transcript`: searchable text for the clip

## Team workflow

- DB team: own schema, indexes, and seed data
- Web team: own upload flow, dashboard, and job creation
- ML team: own worker reliability and inference results

## Notes

- Real secrets should stay in local `.env` files only.
- `env.example` is template data only.
- The current MVP uses file upload instead of live recording because it is faster and more reliable to finish in one day.