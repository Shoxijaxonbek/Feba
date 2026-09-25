# RoadSense — traffic events and accident risk from a fixed junction camera

WIUT Hackathon 2026, Computer Vision track. Given a video from the fixed camera, the
system reports every traffic event as `[start_sec, end_sec, label]` (Part A) and a causal
per-frame accident-risk score (Part B).

- Website (team, approach, EDA, results, live demo, report): https://huggingface.co/spaces/Shoxijaxonbek/Feba
- Repository: https://github.com/Shoxijaxonbek/Feba (the evaluated commit is tagged `submission`)

## Run it

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

Python ≥ 3.10. The model weights are in the repository (`weights/yolo11m.pt`, `weights/yolo11s.pt`,
~60 MB, official Ultralytics YOLO11 COCO checkpoints), so no download step and no internet
is needed. A CUDA GPU is used when present; everything also runs on CPU (slower).

`solution.py` is the only entry point the harness imports. `run_submission.py` and
`evaluate.py` are the unmodified starter-kit files.

Our output on the sample videos is `predictions_samples.json`
(`python run_submission.py --videos samples --out predictions_samples.json --team <team>`).

## How it works

```
video ──► sampled decoding ──► YOLO11m ──► ByteTrack ──► tracks ─┐
 (PyAV, skip non-ref frames,    (COCO:      (vehicles /          ├─► rules, one per class ──► segments ──► events
  ~10 fps, 1920x1080)           people,     two-wheelers /       │    (+ scene layout,
                                vehicles)   pedestrians)         │     signal state,
      └──► lamp colours of the signal heads ──► signal timeline ──┘     lane directions)

frames (harness, causal) ──► YOLO11m every 6th frame ──► ByteTrack ──► footprint time-to-contact ──► risk
```

**Decoding.** The camera files are 3840x2160 H.264 High 4:2:2 10-bit at ~150 Mbit/s.
Converting every frame to BGR (what `cv2.VideoCapture.read` does) runs at ~30 fps on a laptop,
so Part A decodes with PyAV, asks the decoder to skip non-reference frames (the camera's
I-B-B-P GOP then yields every third frame, ~10 fps, at ~5x realtime) and converts only those
frames, straight to 1920x1080 ([src/roadwatch/video.py](src/roadwatch/video.py)).

**Detection and tracking.** YOLO11m (COCO) at 1280 px, pre- and post-processing on the GPU
(the stock Ultralytics CPU letterbox and NMS cost as much as the network on crowded frames).
ByteTrack runs separately for vehicles, two-wheelers and pedestrians so ids never jump
between a cyclist and her bicycle. Persons inside vehicle boxes (bus passengers) and riders
are removed from the pedestrian set ([detector.py](src/roadwatch/detector.py),
[tracking.py](src/roadwatch/tracking.py), [rules/common.py](src/roadwatch/rules/common.py)).

**Scene knowledge.** The camera never moves, so the layout is annotated once in
[configs/scene.json](configs/scene.json) (scene pixels of a 1920x1080 frame): carriageway,
islands and median, crosswalks A/B/C, the stop line with its stop zone, the signal-queue zone,
the junction box, the bus stop, and small boxes around the red/amber/green lamps of the two
visible signal heads. `python scripts/draw_scene.py <video> --out overlay.jpg` draws it.
The vehicle signal is read from lamp glow (colour-specific, per-video two-level threshold,
median filter, short gaps such as flashing green or a passing bus bridged)
([signals.py](src/roadwatch/signals.py)). Lane directions are *learned* from the sample
videos' trajectories: mean direction and coherence per 40 px cell
([flow.py](src/roadwatch/flow.py), [configs/flow_field.npz](configs/flow_field.npz)).

**Rules** ([src/roadwatch/rules/](src/roadwatch/rules)) — the annotation conventions of the
task define start and end:

| class | rule |
|---|---|
| jaywalking | pedestrian foot point on the carriageway (≥ 18 px inside the curb, off islands), ≥ 25 px from every crosswalk, ≥ 1.5 s |
| failure_to_yield | moving vehicle's ground edge inside a crosswalk while a pedestrian is on the roadway part of the same crosswalk within 1.5 vehicle widths; event = the vehicle's passage through the crossing |
| red_light | vehicle crosses the stop-line segment while the signal has been red ≥ 1 s and enters the junction box still on red; ends when it leaves the box |
| stop_line | vehicle stationary ≥ 2 s inside the stop zone (past the line, before the junction) on red; ends at the next green |
| stopped_vehicle | vehicle stationary ≥ 10 s on the carriageway outside the signal queue and the bus stop; fragmented tracks at the same spot are joined |
| wrong_way | moving against the learned lane direction (coherent cells only) for ≥ 1.5 s and ≥ 80 px |
| illegal_u_turn | continuous track (no id switch) whose heading turns ≥ 150° over ≥ 1.5 s (the median island carries a keep-right sign) |
| congestion | ≥ 6 vehicles stationary/crawling in the approach while its signal is green, ≥ 10 s |
| accident | ground footprints of two road users touch while they were closing in; both stop within 2.5 s and stay stopped ≥ 4 s; not in the queue |
| near_miss | the Part B conflict score above the alarm level for ≥ 0.4 s with no contact afterwards |

`illegal_turn`, `solid_line_crossing`, `road_obstacle` and `fire_smoke` are never predicted:
Score A is a macro average over the classes present in the ground truth *or* in the
predictions, so a class we cannot detect reliably is better left out than guessed.

**Part B** ([risk.py](src/roadwatch/risk.py)). `RiskEstimator.step` only sees the frames the
harness passes. Every 6th frame (~5 fps) is downscaled on the CPU, detected, and tracked. For
each pair of moving road users on the carriageway (at least one a vehicle) the ground
footprints are extrapolated linearly; if they would start to overlap within 3 s, the pair
scores relative speed / (2 · time to contact), the deceleration needed to avoid contact
(a DRAC-style surrogate safety measure). Parked and queued road users are ignored, a pair
must stay dangerous in two consecutive processed frames, and a logistic maps the score to
[0, 1] with 0.5 just above the 99.9th percentile of normal traffic in the samples.

**Learned vs rule-based.** Learned: YOLO11 weights (COCO pre-training by Ultralytics, used
as is), the lane-direction field and the risk calibration (fitted on our sample videos).
Rule-based: everything else (tracking association, layout, signal reading, event rules,
segment post-processing).

## Datasets and licences

| data / model | use | licence |
|---|---|---|
| Organizers' sample videos | layout annotation, lane-direction field, risk calibration, our dev labels | hackathon use |
| COCO (via Ultralytics YOLO11 checkpoints) | detector pre-training (not re-trained) | CC BY 4.0 annotations; YOLO11 weights AGPL-3.0 |
| ByteTrack (Ultralytics implementation) | multi-object tracking | AGPL-3.0 |

No other external data was used.

## Runtime

On an RTX 4050 laptop GPU (the target is a T4): Part A ≈ 0.6x the video duration, Part B
≈ 1.3x (most of it is the harness decoding every 4K frame), total ≈ 1.9x against the 3x
budget. `python scripts/cache_observations.py` caches Part A's detections so rules can be
tuned in seconds.

## Determinism

`set_seed(0)` fixes Python, NumPy and PyTorch RNGs and disables cuDNN autotuning; there is no
sampling anywhere. The only nondeterminism is GPU floating-point accumulation order inside the
detector, which can move a box by a fraction of a pixel; event boundaries are rounded to 10 ms.

## Repository layout

```
solution.py               harness interface (Part A + Part B)
run_submission.py         starter kit, unchanged
evaluate.py               starter kit, unchanged
src/roadwatch/            the pipeline (video, detector, tracking, scene, signals, flow, rules/, risk, render, report)
configs/                  scene layout + learned lane-direction field
weights/                  YOLO11 checkpoints
scripts/                  data download, caching, layout drawing, EDA and website export, risk replay, clip cutting
app/server.py             live-demo backend (FastAPI), serves website/
website/                  static team website
labels/                   our dev labels of the sample videos
predictions_samples.json  our output on the sample videos
```

## Team Feba

| member | role | contributions |
|---|---|---|
| Shuxratov Shoxijaxonbek | team captain, computer vision | decoding, detection and tracking; scene layout, signal reader and event rules; dev labels, evaluation, Part B risk model |
| Ibragimov Diyorbek | frontend developer | the website: results viewer with clickable timelines and risk curves, EDA charts, operator dashboard |
| Lutfullayev Mirfayz | backend developer | live-demo API (uploads, job queue, progress), annotated rendering, Hugging Face deployment |

## Acknowledgements

Ultralytics YOLO11 and its ByteTrack implementation (AGPL-3.0), PyAV/FFmpeg, OpenCV, FastAPI.
