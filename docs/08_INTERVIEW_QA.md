# 08 — Interview Q&A

Short answers in plain language. Numbers are from my own runs (`06_RESULTS.md`).

---

### The project

**1. What does your project do, in one sentence?**
It detects people in video with YOLO26n, gives each person a stable ID with ByteTrack,
and on top of that my own code counts people crossing a line (IN/OUT), counts people in
view, and records how long each person stays — saved to CSV.

**2. Which parts did you write yourself, and which come from a library?**
Ultralytics provides the YOLO26n detector and the ByteTrack tracker. I wrote the counting
and analytics (`counter.py`: cross-product line test, IN/OUT, double-count protection,
trails, time in frame, occupancy), the app (`main.py`), the experiments (`benchmark.py`,
including scoring against ground truth) and the tests. I deliberately did *not* use
Ultralytics' built-in counter.

**3. How accurate is it?**
On 4 CAVIAR shopping-centre clips with hand-labelled ground truth: line crossings were
right on 29 of 32 (IN 20/20 overall, OUT 11/12); people-in-view was off by 0.8–1.0 people
per frame on average. But "unique visitors" from track IDs over-counts 1.6–4.6× because of
ID switches — I wouldn't report that number without re-identification.

**4. How fast is it?**
On the RTX 4060 laptop GPU: 11.5–11.8 ms per frame, about 86 FPS, very consistent. On the
CPU: 27–114 ms, varying run to run. Inference is ~70% of the GPU time; tracking only ~1.3 ms.

### Detection (YOLO)

**5. How does YOLO detect objects?**
One pass of a neural network over the whole image. The image is turned into feature maps
— grids of 80×80, 40×40 and 20×20 cells for a 640 input — and every cell predicts a box
(4 numbers) and 80 class scores. That's 8400 candidate boxes; weak ones are filtered out
by the confidence threshold.

**6. What is IoU?**
Intersection over Union: overlap area ÷ combined area of two boxes, from 0 (no overlap) to
1 (identical). Example: two 100×100 boxes overlapping in a 50×50 square → 2500 / 17500 ≈ 0.14.
It's used for removing duplicate boxes and for matching detections to tracks.

**7. What is NMS, and why doesn't YOLO26 need it?**
Non-Maximum Suppression removes duplicate boxes: keep the best box, delete others that
overlap it too much. YOLO26 is trained "one-to-one" — each object gets one prediction — so
it outputs clean boxes directly. That makes speed predictable and export to edge devices
simpler. In my tests, changing `conf` didn't change speed (10.9–11.3 ms), consistent with
no NMS step whose work grows with the number of boxes.

**8. Why YOLO26n (nano) and not a bigger model?**
Real-time on modest hardware: 2.6 M parameters, 6.1 GFLOPs. A bigger model would detect
small far-away people better but run slower — a trade-off I'd measure for a real store.

### Tracking (ByteTrack)

**9. Why do you need tracking at all?**
The detector sees each frame separately. To count a person crossing a line *once*, or
measure how long they stay, I need to know it's the same person across frames — an ID.

**10. How does ByteTrack work?**
Each track has a Kalman filter that predicts where the box will be next. New detections
are compared to the predictions with IoU, and the Hungarian algorithm (here the faster
Jonker–Volgenant solver, `lap`) picks the best one-to-one matches. ByteTrack's trick: it
matches high-confidence boxes first, then uses **low-confidence** boxes to keep existing
tracks alive — those are often people who are partly hidden. Unmatched confident boxes
start new tracks; tracks lost for more than 30 frames are deleted.

**11. Why is your detection confidence threshold so low (0.10)?**
Because ByteTrack's second stage needs boxes scoring 0.1–0.25. If the detector drops them
first, the tracker never sees them. Ultralytics' `track()` defaults to 0.1 for the same
reason.

**12. What happened when you raised the threshold?**
At 0.50 the number of IDs fell (82 → 41) — but mostly because far-away people weren't
detected at all; people-in-view dropped below the ground truth. I'd predicted more ID
switches instead; the data showed a different effect. Fewer IDs isn't better tracking.

**13. What's an ID switch, and how would you reduce them?**
When the tracker loses a person and gives them a new ID. Causes: occlusion, missed
detections, crowds, a moving camera. Fixes: an overhead camera, re-identification (an
appearance model, e.g. BoT-SORT with ReID), tuning `track_buffer`.

### My counting logic

**14. How do you know which side of the line a person is on?**
The sign of a 2D cross product. For line A→B and point P:
`(Bx−Ax)(Py−Ay) − (By−Ay)(Px−Ax)`. Positive = one side, negative = the other, zero = on
the line. Example: A(100,200), B(500,200), P(300,150) → 400·(−50) = −20 000 → negative side.

**15. How do you detect a crossing and its direction?**
I store each ID's last side. If the sign flips between frames, it crossed: −1→+1 is IN,
+1→−1 is OUT. I also check with a dot product that the crossing is between the line's
two ends, and I keep the last *non-zero* side so landing exactly on the line doesn't lose
the crossing.

**16. How do you avoid counting someone twice?**
A set of IDs already counted IN and a set for OUT; each ID can be counted once per
direction. A person wobbling on the line can add at most one false count, not many. A
production version would add a hysteresis band or require a few frames on the new side.

**17. How do you know your counter is correct?**
12 unit tests with hand-calculable points (cross-product signs, jitter, crossing past the
line's end, exactly-on-line, time in frame…), all passing; and on real footage I compared
it against CAVIAR's ground truth by running the same counter on the human-labelled boxes.

### Data, privacy, production

**18. Why CAVIAR and not your own video?**
It's real footage from a shopping-centre corridor with a fixed high camera — close to a
footfall setup — and every frame is hand-labelled, so I could measure accuracy instead of
eyeballing it. My own camera only ever saw one person, and close up.

**19. What about privacy?**
I store only numbers — random track IDs, times, directions — no faces, no identities. The
model detects "a person", not who. In production I'd run on a device in the store, send
only aggregated counts, not keep raw video, and use signage and data-protection rules.

**20. How would you turn this into a product for supermarkets?**
Overhead fixed cameras above doors; an edge device running the model exported to TensorRT;
a counter with hysteresis using the feet point; staff filtering; fine-tuning on the
store's own footage; counts sent every 15 minutes to a dashboard; and regular accuracy
checks against manual counts.

### Bonus: what went wrong, and what did you learn?

- First benchmark numbers were garbage because another run was using the GPU — now I
  check the machine is idle and repeat runs.
- OpenCV said a 3700-frame video had 41 frames — never trust metadata.
- My close-up webcam made the counter look bad; the logic was fine, the camera placement
  wasn't. Placement matters as much as code.
