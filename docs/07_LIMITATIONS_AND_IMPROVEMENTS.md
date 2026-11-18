# 07 — Limitations and Improvements

What this prototype can't do well, the evidence from my own runs, and how a production
footfall system would fix it. "Evidence" points to `06_RESULTS.md` or the progress log;
improvements are ideas, **not tested here** unless stated.

## Summary

| limitation | evidence from my runs | production fix |
|---|---|---|
| ID switches | 1.6–4.6 track IDs per real person (CAVIAR) | re-identification tracker, better camera placement |
| Occlusion | far/grouped people lose tracks; IDs fragment | overhead camera, ByteTrack low-score stage, ReID |
| Line jitter | same ID IN + OUT within ~1 s (IDs 3, 171, 177, 179) | hysteresis band / N-frame confirmation |
| Camera angle & distance | close-up camera: 14 IDs for 1 person, false crossings | high or overhead mounting |
| Small, far-away people | at conf 0.50, people-in-view fell below the truth | larger `imgsz`, bigger model, fine-tuning |
| Dense crowds | phone video: 5–13 counted when 20–40 were visible | overhead view, density-map crowd counting |
| Moving camera | phone video: IN 29 / OUT 31 from panning alone | fixed camera; motion compensation |
| False positives | my chair tracked as a person for 6.3 s | higher `new_track_thresh`, ignore zones, fine-tuning |
| Staff vs customers | no way to tell staff from shoppers (every person box is the same) | staff filtering (see below) |
| Lighting | not tested | camera exposure settings, IR cameras, fine-tuning |
| CPU speed | 26.9–113.5 ms/frame vs GPU 11.5–11.8 ms | GPU / NPU edge device, TensorRT/ONNX export |

---

## 1. ID switches (the biggest weakness)

ByteTrack links boxes frame to frame using only **position and overlap** (IoU + Kalman
filter). When a person is hidden or missed for more than `track_buffer` = 30 frames, or
their box jumps, they come back with a **new ID**. Measured on CAVIAR:
**82 IDs for 18 real people** (ShopAssistant2cor). Consequences:

- "Unique visitors" from track IDs is **not usable** (over-counts 1.6–4.6×).
- An ID switch right at the line can make one person count twice.
- Line counts and people-in-view were still good (29 of 32 crossings; ~0.9 people error
  per frame), because they don't need an ID to survive long.

**Improvements:**
- **Re-identification (ReID):** add an appearance model that turns each person crop into
  a feature vector ("embedding"), so a lost person can be matched again by how they *look*,
  not just where they are. Trackers like **BoT-SORT** (with ReID enabled) and **Deep OC-SORT**
  do this — both are in the installed Ultralytics (`botsort.yaml`, `deepocsort` tracker type).
- Longer `track_buffer` for slow-moving shoppers (trade-off: risk of wrong re-matches).
- Better camera placement (below) removes most of the cause.

## 2. Occlusion

Shoppers walk in groups, behind trolleys and shelves. A half-hidden person gets a low
detection score; ByteTrack's second stage helps (it keeps tracks alive with 0.1–0.25 score
boxes), but a fully hidden person can't be detected at all. In the CAVIAR corridor,
removing the second stage (conf 0.10 → 0.25) cost little because occlusion is moderate; a
busy supermarket entrance would depend on it more.

**Improvements:** overhead cameras (people rarely hide each other from above), ReID to
reconnect after occlusion, multi-camera views of the same area.

## 3. Line jitter

When a box centre wobbles on the line, the sign flips back and forth. My counter allows
at most one IN and one OUT per ID, so jitter can add **one** false count, not many —
seen on CAVIAR (IDs 3, 171, 177, 179) and in the live test (stand-up next to the line).

**Improvements:**
- **Hysteresis band:** two parallel lines a few pixels apart; only count when a person
  goes fully from one outer line past the other.
- **N-frame confirmation:** only count when the centre stays on the new side for e.g. 5 frames.
- Use the **bottom-centre (feet)** instead of the box centre: the feet move steadily
  across a floor line, while the centre moves when someone leans, crouches, or the box is
  clipped at the frame edge.

## 4. Camera angle, distance and placement

The same code gave very different quality on different views:

- **Close-up camera** (live test): boxes filled the frame and were clipped at the edges,
  centres jumped → false crossings, 14 IDs for 1 person.
- **High corridor camera** (CAVIAR): good line counts, but far-away people are ~20 px tall
  and partly lost.
- **Where the line goes decides what is measured:** on the sample clip a horizontal line gave
  4 crossings, a vertical line 0, for the same people.

**Improvement:** mount footfall cameras **overhead, above the doorway**, looking straight
down. People don't overlap, boxes are similar sizes, and walking through the door is a
clean crossing. This is how commercial footfall counters are installed. Note that YOLO26n
was trained on COCO, which has few top-down views of people, so an overhead camera would
also call for fine-tuning (section 7).

## 5. Small and far-away people

At 384×288, people at the end of the corridor are ~20 px tall and score low. At conf 0.50
they disappeared (people-in-view fell to 4.21 vs 5.84 true on ThreePastShop1cor).

**Improvements:** keep conf low (0.10); a larger inference size (`--imgsz 960`) or a bigger
model (YOLO26s/m) — slower, trade-off to measure; a higher-resolution camera; only count
inside a region of interest near the door.

## 6. Dense crowds and moving cameras

The handheld protest video showed two hard limits (progress log 3.9):

- **Dense crowds:** only near, large people were detected — 5–13 counted when 20–40 were
  visible. For real crowd *counting*, density-map methods (estimate how many people per
  pixel area, without drawing a box per person) work better; overhead views help too.
- **Moving camera:** the counting line is fixed to the screen, so panning makes everyone
  "cross" it (IN 29 / OUT 31). Footfall cameras must be fixed. Trackers with camera-motion
  compensation (BoT-SORT's GMC) help tracking, but line counting still needs a fixed view.

## 7. False positives and model domain

My chair was tracked as a person for 6.3 s at conf 0.10. The model only knows COCO
pictures; shop-specific things (mannequins, posters of people, reflections in shop
windows) can fool it.

**Improvements:**
- Raise `new_track_thresh` (start tracks only from confident boxes) while keeping `conf` low
  for ByteTrack's second stage.
- **Ignore zones** (e.g. mannequin displays) and a minimum time-in-frame before counting.
- **Fine-tuning:** label a few hundred frames from the real store cameras and train YOLO26n
  further on them. This is where *training* becomes worth it — this project only ran
  inference with pretrained weights.

## 8. Staff vs customers

Footfall should count **customers**. Staff walk through the door many times a day and
inflate counts. In CAVIAR's ShopAssistant2cor the system has no way to tell the shop
assistant apart from shoppers — every person box is treated the same.

**Options:** a small classifier on the person crop for staff uniforms; staff wearing a
badge/beacon; ignore people who stay much longer than customers or repeat the same path;
exclude staff-only doors. Each has privacy and accuracy trade-offs.

## 9. Lighting

Not tested — CAVIAR is evenly lit indoor footage. Supermarket entrances face strong
backlight from glass doors, evening darkness, and reflections. Mitigations: camera
exposure / wide-dynamic-range settings, infrared cameras, and fine-tuning on footage from
those conditions.

## 10. Speed and edge deployment

Measured: GPU 11.5–11.8 ms/frame (≈ 86 FPS) and predictable; CPU 26.9–113.5 ms and
unpredictable. Inference is ~70% of GPU time; tracking only ~1.3 ms. 8 ms of inference is
slow for a 6.1-GFLOP model — likely because PyTorch launches hundreds of small layers one
by one.

**Improvements (not tested):**
- **Export** the model: ONNX → **TensorRT** on NVIDIA hardware (fuses layers, FP16/INT8
  maths), or OpenVINO on Intel CPUs. YOLO26 being **NMS-free** makes this export clean:
  the whole model is one graph with no separate post-processing step.
- Run on an **edge device** in the store (e.g. NVIDIA Jetson or a camera with an NPU) so
  video never leaves the building and only counts are sent.
- Process fewer frames (e.g. 10–15 FPS is often enough for walking speed).

## 11. Privacy

Counting people means filming people, so privacy is part of the design, not an add-on:

- **What this project stores:** only numbers — track IDs (random integers), timestamps,
  seconds in frame, crossing direction (`outputs/*.csv`). **No faces, no names, no images
  per person.** The model detects "a person", not *who* it is.
- **What to be careful with:** `--save` writes annotated video with people in it. In the
  repo, those videos are git-ignored, and frames showing my own face or the third-party
  phone video were deliberately **not** committed.
- **Production:** process **on the device**; send only aggregate counts (e.g. IN/OUT per
  15 minutes); don't keep raw video, or keep it briefly with access control; no face
  recognition; inform shoppers with signage; follow data-protection law (e.g. GDPR /
  local equivalent: purpose limitation, minimal data, retention limits).
- **Re-identification is a privacy trade-off:** appearance embeddings help unique-visitor
  counts but are closer to identifying a person — keep them short-lived (minutes, on device)
  and never link them to identity.

## 12. What a production system would look like

```
 overhead camera ──► edge device in store ───────────────────────────► cloud dashboard
 (fixed, above door)  YOLO26n (TensorRT, FP16)                          IN/OUT per 15 min,
                      ByteTrack / BoT-SORT + ReID                       occupancy, peaks,
                      line counter with hysteresis, feet point          per store / per door
                      staff filter, ignore zones
                      sends only counts — no video leaves the store
```

Plus: per-store calibration of the line, monitoring of count accuracy against occasional
manual counts, and fine-tuning on each store type's footage.
