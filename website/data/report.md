<!-- _mock: true (replace this whole file with the final report) -->

## What we built

RoadSense turns a fixed 4K CCTV view of a signalised Tashkent junction into a list of traffic
events (`[start_sec, end_sec, label]`) and a causal, per-frame accident-risk score. It runs at
about **0.5× the video duration** for Part A, far inside the 3× budget.

The pipeline is a small amount of learning and a lot of explicit geometry:

1. **Decode** only the reference frames of the camera's I-B-B-P GOP, which gives about 10 fps at ~5× realtime.
2. **Detect** road users with YOLO11m (COCO weights) and **track** them with ByteTrack in three groups.
3. **Read the signal** from lamp colours and place every track in a **hand-annotated scene layout**.
4. Apply **one rule per event class**, then merge and trim segments.
5. For Part B, estimate **time-to-contact** between ground footprints, convert it to a required
   deceleration and calibrate it into a probability.

## What worked

- Skipping B-frames cost nothing in accuracy and made the whole pipeline fast enough for CPU demos.
- The fixed 80 s signal cycle made `stop_line` our most reliable class.
- Emitting only the classes we can detect reliably lifted the macro-F1: each class we predict
  wrongly adds a zero to the average.

## What did not work

- **Long U-turns.** Vehicles creeping in the turn pocket add up heading drift. See the failure cases.
- **Far-side red-light runners** are too small to track through the junction box.
- **Sub-second failure_to_yield fragments** can never reach tIoU 0.3.

## What next

- Fine-tune the detector on a few hundred frames from this camera (far-side pedestrians, two-wheelers).
- Add a parking-bay mask to the layout and a minimum duration per class.
- Label more accidents and near misses to fit the risk calibration on real positives.

| Metric | Value |
| --- | --- |
| Part A score (dev set) | 0.41 |
| Part A runtime | 0.47× video |
| Classes emitted | 5 of 14 |
