## What we built

RoadSense turns the fixed 4K view of a signalised Tashkent junction into a list of traffic
events `[start_sec, end_sec, label]` (Part A) and a causal, frame-by-frame accident-risk score
(Part B). It is a little learning and a lot of explicit geometry:

1. **Decode only what we use.** The camera writes H.264 High 4:2:2 10-bit at ~150 Mbit/s. PyAV skips
   non-reference frames: the I-B-B-P GOP yields every third frame (~10 fps) at ~5× realtime.
2. **Detect and track.** YOLO11m (COCO weights, not fine-tuned) with pre- and post-processing on the
   GPU; ByteTrack separately for vehicles, two-wheelers, pedestrians and possible obstacles.
3. **Know the scene.** Carriageway, islands, three zebras, the stop line, queue zones, the junction box
   and the bus stop are drawn once on a reference view. Every video is registered to it with SIFT
   features, because the tripod moved by up to 80 px between the four recordings.
4. **Read the signal.** Each lamp is located as the spot whose colour switches on and off over the
   video, then read frame by frame. That works in harsh midday sun and at dusk.
5. **One rule per class**, written from the annotation conventions (start and end of each class).
6. **Part B**: causal tracking at 5 fps, the time until two ground footprints would touch, the
   deceleration needed to avoid it, a conflict that must hold for ~0.6 s, and a calibration that keeps
   normal traffic below the alarm level.

## What worked

- **Decoding only reference frames** made Part A take about half the video's length with no visible
  loss: 10 fps is plenty for tracking at junction speeds.
- **Reviewing every detection by eye.** We rendered each event as an image strip and looked at all of
  them. Almost every false positive had a reason we could turn into a rule condition, not a threshold
  tweak: people waiting at zebra ends, parked cars, passengers seen through bus windows, boxes cut off
  at the frame edge, ID switches between buses, left-turners and queues at other approaches waiting
  inside the junction, a bonnet over the stop line.
- **Telling queues from stopped vehicles by their neighbours.** A queue stands still together; a stopped
  vehicle stands still while the traffic around it keeps moving.
- **Predicting only the classes we can detect.** Score A averages over every class that appears in the
  ground truth *or* the predictions, so a class we guess wrongly adds a zero.

## What did not work

- **The first versions did not survive new footage.** Everything was first tuned on the one dusk video
  we could download. The midday videos broke the registration (ORB), the signal reader (a lit lamp in
  direct sun is barely brighter than its housing), wrong-way (cars from the other approach turn through
  the junction) and stopped_vehicle (queues everywhere). Each fix is listed under failure cases.
- **Part B** raised false alarms three times: first from centre-to-centre time-to-collision, then from
  a preprocessing mismatch with Part A, then from one-frame spikes of cars merging at midday. We still
  have **no accident or near miss** to calibrate on, so the alarm level is set from normal traffic only.
- **failure_to_yield** is our noisiest class: we cannot yet tell a pedestrian about to step off the curb
  from one who is waiting.
- **Illegal or legal?** Cars regularly U-turn around the median end at midday. We do not know whether the
  annotators treat these as illegal; we report them.
- **Our own labels** of the three videos we got last were made by reviewing our detections, so they
  measure precision, not recall. Only C3902's labels were built from loose-threshold candidates.

## What we would do next

- Fine-tune the detector on a few hundred frames of this camera (small far-side pedestrians and
  scooters, which COCO does not have).
- Use pedestrian motion (heading onto the road vs. standing) for failure_to_yield.
- Collect or stage accident and near-miss clips to fit the Part B calibration on real positives.
- Trace the lane markings on an empty-road frame to add solid_line_crossing and illegal_turn.
