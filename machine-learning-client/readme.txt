Machine learning worker for environmental sound detection.

Quick start
1) Ensure `.env` has `MONGO_URI`.
2) Optional env vars:
	- `MONGO_DB_NAME` (default: `sound_alerts`)
	- `HF_SOUND_MODEL_ID` (default: `MIT/ast-finetuned-audioset-10-10-0.4593`)
	- `HF_ASR_MODEL_ID` (default: `openai/whisper-tiny.en`)
	- `POLL_INTERVAL_SECONDS` (default: `5`)
3) Run worker:
	- `python client.py` (continuous polling)
	- `python client.py --once` (process one pending job)

Mongo contract for teammate integration
- Input collection: `analysis_jobs`
- Output collection: `predictions`
- Alert labels: `siren`, `car_horn`, `bicycle`, `dog_bark`, `alarm`, `engine`, `footsteps`
- Speech captions: `captions` array with `start_time`, `end_time`, `text`, and `confidence`
- Full transcript: `full_transcript` string for searchable text and playback captions
- If the user uploads video, the web app should extract the audio track and store the extracted path in `audio_path`
- Optional helper fields for the web team: `media_path`, `media_type`, `original_filename`, `duration_seconds`

Expected pending job document
{
  "audio_path": "/path/to/audio.wav",
  "status": "pending",
  "created_at": <datetime>
}

Worker updates
- Claims one pending job by setting `status=processing` and `started_at`
- On success: inserts a sound-alert prediction, marks job `status=done`, sets `prediction_id` and `completed_at`
- On failure: marks job `status=failed`, sets `error` and `completed_at`

Prediction document fields
- `alert_type`, `alert_confidence`, `category`, `confidence`
- `detections` for timestamped sound-event ranges
- `captions` for timestamped speech-to-text overlays
- `full_transcript` for the clip-level transcript
- `model_name`, `created_at`, `audio_path`, and `job_id`
