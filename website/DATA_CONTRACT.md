# Website data contract

The site is static (no build step): `website/index.html` + `website/assets/*` read JSON from
`website/data/`. The live demo talks to a FastAPI backend served from the same origin
(`/api/...`, implemented in `app/server.py`). All times are seconds from the first frame.

## `data/site.json`
```json
{
  "team": {"name": "TEAM NAME", "tagline": "...",
           "members": [{"name": "...", "role": "...", "contributions": ["..."],
                        "github": "https://...", "linkedin": "https://...", "portfolio": "https://...",
                        "photo": "assets/img/member1.jpg or null",
                        "projects": [{"title": "...", "url": "...", "blurb": "..."}]}]},
  "links": {"repo": "https://github.com/...", "weights": "https://...", "predictions": "data/predictions_samples.json",
            "report_pdf": null}
}
```

## `data/samples/index.json`
```json
[{"id": "C3902", "file": "C3902.MP4", "duration": 317.8, "fps": 29.97, "width": 3840, "height": 2160,
  "annotated_video": "data/samples/C3902_annotated.mp4", "poster": "data/samples/C3902_poster.jpg",
  "n_events": 46}]
```

## `data/samples/<id>.json`  (one per sample video; also the shape of the demo result)
```json
{
  "id": "C3902", "duration": 317.8,
  "annotated_video": "data/samples/C3902_annotated.mp4",
  "events": [{"start": 10.7, "end": 13.4, "label": "jaywalking", "tracks": [121],
              "thumb": "data/samples/thumbs/C3902_003.jpg", "info": {"crosswalk": "B"}}],
  "risk":   [[0.0, 0.01], [0.2, 0.01]],          // Part B curve, ~5 Hz
  "signal": [[0.0, "red"], [34.2, "amber"], [37.1, "green"]],   // change points of the main signal
  "counts": {"t": [0, 1, 2], "car": [12, 13, 12], "bus": [1, 1, 0], "truck": [0, 0, 0],
             "person": [30, 31, 29], "bicycle": [0, 0, 0], "motorcycle": [0, 0, 1]},  // 1 Hz
  "timing": {"part_a_sec": 148, "part_b_sec": 439, "video_sec": 317.8}
}
```
Event labels are the 14 official ids: accident, near_miss, red_light, wrong_way, illegal_u_turn,
stopped_vehicle, jaywalking, failure_to_yield, illegal_turn, solid_line_crossing, stop_line,
congestion, road_obstacle, fire_smoke.

## `data/eda/eda.json`
```json
{
  "videos": [{"id": "C3902", "duration": 317.8, "fps": 29.97, "width": 3840, "height": 2160,
              "codec": "H.264 High 4:2:2 10-bit", "bitrate_mbps": 147, "size_gb": 5.84,
              "brightness": {"t": [0, 10], "mean": [92.1, 91.7]}}],
  "images": {"vehicle_heatmap": "data/eda/vehicle_heatmap.jpg", "person_heatmap": "data/eda/person_heatmap.jpg",
             "flow_field": "data/eda/flow_field.jpg", "trajectories": "data/eda/trajectories.jpg",
             "scene_layout": "data/eda/scene_layout.jpg", "signal_heads": "data/eda/signal_heads.jpg"},
  "counts_over_time": {"<video id>": {"t": [], "car": [], "person": [], "...": []}},
  "signal_cycle": {"cycle_s": 80.0, "green_s": 33.0, "flash_s": 4.0, "amber_s": 3.0, "red_s": 37.0, "red_amber_s": 3.0,
                   "timeline": {"<video id>": [[0.0, "red"], [34.2, "amber"]]}},
  "queue": {"<video id>": {"t": [], "vehicles": [], "stationary": []}},
  "speeds": {"vehicle_px_s": [/* histogram */], "bins": []},
  "findings": [{"title": "...", "text": "...", "image": "data/eda/....jpg"}]
}
```

## `data/metrics.json`
```json
{
  "dev_set": "our own labels of the sample videos",
  "per_class": {"jaywalking": {"f1_03": 0.6, "f1_05": 0.5, "f1_07": 0.3, "tp": 5, "fp": 2, "fn": 3}},
  "score_a": 0.41, "score_b": null,
  "ablations": [{"name": "YOLO11s vs YOLO11m", "rows": [{"variant": "...", "score_a": 0.3, "sec_per_min": 20}]}],
  "timing": [{"video": "C3902", "duration": 317.8, "part_a_sec": 148, "part_b_sec": 439, "budget_sec": 953}]
}
```

## Demo API (same origin)
- `POST /api/jobs` multipart field `video` (.mp4, <= 500 MB, <= 2 min) -> `{"job_id": "..."}`
- `GET /api/jobs/<id>` -> `{"status": "queued|running|done|error", "progress": 0.42, "stage": "detecting", "error": null}`
- `GET /api/jobs/<id>/result` -> same shape as `data/samples/<id>.json` (annotated_video is a URL under `/api/jobs/<id>/video`)
