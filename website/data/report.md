## What we built

RoadSense turns the fixed 4K view of a signalised Tashkent junction into a list of traffic
events `[start_sec, end_sec, label]` (Part A) and a causal, frame-by-frame accident-risk score
(Part B). It is a little learning and a lot of explicit geometry:

1. **Decode only what we use.** The camera writes H.264 High 4:2:2 10-bit at ~150 Mbit/s. PyAV skips
   non-reference frames: the I-B-B-P GOP yields every third frame (~10 fps) at ~5× realtime.
2. **Detect and track.** YOLO11m (COCO weights, not fine-tuned) with pre- and post-processing on the
   GPU; ByteTrack separately for vehicles, two-wheelers, pedestrians and possible obstacles.
3. **Know the scene.** The camera never moves, so carriageway, islands, three zebras, the stop line,
   queue zones, the junction box and the bus stop were drawn once. Every video is registered to that
   reference view (ORB + RANSAC) in case the tripod was nudged. Lane directions are learned from our
   trajectories.
4. **Read the signal** from the colour of its lamps: an 80 s fixed cycle, recovered frame by frame.
5. **One rule per class**, written from the annotation conventions (start and end of each class).
6. **Part B**: causal tracking at 5 fps, time until two ground footprints would touch, the
   deceleration needed to avoid it, and a calibration that keeps normal traffic below the alarm level.

## What worked

- **Decoding only reference frames** cut Part A to ~0.5× the video length with no visible loss:
  10 fps is plenty for tracking at junction speeds.
- **Reading the signal from its lamps** gave a clean red/amber/green timeline, which the
  signal-dependent rules (red_light, stop_line, congestion) rely on.
- **Reviewing every candidate event on image strips** found almost all false positives: people
  waiting at zebra ends, parked cars, passengers inside buses, boxes clipped at the frame border,
  id switches between buses. Each became a rule condition, not a threshold tweak.
- **Predicting only the classes we can detect.** Score A averages over every class that appears in
  the ground truth *or* the predictions, so a class we guess wrongly adds a zero.

## What did not work

- **Our first Part B** raised 19 alarms in 5 minutes of normal traffic: centre-to-centre
  time-to-collision treats every car passing a parked taxi as a collision course. Footprint overlap,
  ignoring parked road users and a persistence filter fixed it, but we still have **no accident or
  near miss** to calibrate on, so the alarm level is set from negatives only.
- **failure_to_yield** is our noisiest class (about 1 false positive per true one on the dev labels):
  we cannot yet tell a pedestrian about to step off the curb from one who is waiting.
- **Solid-line crossing and illegal turns** need lane-level markings that the median background does
  not show clearly enough under queued cars; we left them out rather than guess.
- **Downloading the sample videos**: Google Drive's per-file quota blocked scripted downloads for hours.

## What we would do next

- Fine-tune the detector on a few hundred frames of this camera (small far-side pedestrians,
  scooters, which COCO does not have).
- Use pedestrian motion (heading onto the road vs. standing) for failure_to_yield.
- Collect or stage accident and near-miss clips to fit the Part B calibration on real positives.
- Trace the lane markings on a frame with an empty approach to add solid_line_crossing.
