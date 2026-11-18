"""VisionTrack counting logic — written by hand (no Ultralytics "solutions" classes).

What it does, per tracked person (track ID):
  1. works out which side of the counting line their centre point is on (cross-product sign),
  2. detects a crossing when that side flips between frames,
  3. counts IN or OUT depending on the flip direction, at most once per ID per direction,
  4. keeps a short trail of recent centre points for drawing,
  5. records when the ID was first/last seen -> time in frame.

Points are (x, y) in pixels. In images y grows DOWNWARDS (row 0 is the top).
"""
from collections import deque

import cv2

# Colours are (Blue, Green, Red) in OpenCV
LINE_COLOUR = (0, 255, 255)     # yellow
TEXT_COLOUR = (255, 255, 255)   # white


# ---------------------------------------------------------------------------
# Pure geometry functions (no video, no model: easy to test with made-up points)
# ---------------------------------------------------------------------------

def side_of_line(point, line_start, line_end):
    """Return +1, -1 or 0: which side of the line A->B the point P is on.

    Uses the 2D cross product of the vectors AB and AP:
        cross = (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)
    The sign tells the side; 0 means P is exactly on the line.
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def is_within_segment(point, line_start, line_end):
    """True if the point lies "alongside" the segment A-B, i.e. between its two ends.

    Projects P onto the line: t = (AP . AB) / |AB|^2  (a dot product).
    t = 0 at A, t = 1 at B; anything outside [0, 1] is beyond an end of the line,
    so someone walking past the END of the line is not counted.
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    dx, dy = bx - ax, by - ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    return 0.0 <= t <= 1.0


def crossing_direction(previous_side, current_side):
    """Return "IN", "OUT" or None from the side before and the side now.

    IN  = moved from the negative side (-1) to the positive side (+1)
    OUT = moved from the positive side (+1) to the negative side (-1)
    None = no previous side yet, still on the same side, or exactly on the line (0).
    """
    if previous_side is None or current_side == 0 or previous_side == current_side:
        return None
    return "IN" if current_side == 1 else "OUT"


def colour_for_id(track_id):
    """Give each track ID its own stable colour (same ID -> same colour every frame)."""
    return (track_id * 37 % 256, track_id * 17 % 256, track_id * 91 % 256)


# ---------------------------------------------------------------------------
# The counter: remembers every track across frames
# ---------------------------------------------------------------------------

class LineCounter:
    """Counts line crossings per track ID and keeps trails and time-in-frame."""

    def __init__(self, line_start, line_end, trail_length=30):
        """line_start / line_end are (x, y) pixel points; trail_length = points kept per trail."""
        self.line_start = line_start
        self.line_end = line_end
        self.trail_length = trail_length

        self.last_side = {}       # track_id -> last NON-ZERO side (+1 or -1)
        self.trails = {}          # track_id -> deque of recent centre points
        self.first_seen = {}      # track_id -> timestamp (s) of first appearance
        self.last_seen = {}       # track_id -> timestamp (s) of latest appearance
        self.frames_seen = {}     # track_id -> number of frames it was present
        self.counted_in = set()   # IDs already counted IN  (a set can't hold duplicates)
        self.counted_out = set()  # IDs already counted OUT

    @property
    def in_count(self):
        """Number of people counted IN."""
        return len(self.counted_in)

    @property
    def out_count(self):
        """Number of people counted OUT."""
        return len(self.counted_out)

    def seconds_in_frame(self, track_id):
        """Time between the first and latest frame this ID was seen."""
        return self.last_seen[track_id] - self.first_seen[track_id]

    def update(self, track_ids, centres, timestamp):
        """Process one frame.

        track_ids: list of ints from ByteTrack; centres: list of (x, y), same order;
        timestamp: time of this frame in seconds.
        Returns the crossings counted in this frame, e.g. [(4, "IN")].
        """
        events = []
        for track_id, centre in zip(track_ids, centres):
            # --- time in frame ---
            self.first_seen.setdefault(track_id, timestamp)   # only set the first time
            self.last_seen[track_id] = timestamp
            self.frames_seen[track_id] = self.frames_seen.get(track_id, 0) + 1

            # --- trail: deque(maxlen=N) drops the oldest point automatically ---
            if track_id not in self.trails:
                self.trails[track_id] = deque(maxlen=self.trail_length)
            self.trails[track_id].append(centre)

            # --- crossing ---
            side = side_of_line(centre, self.line_start, self.line_end)
            if side == 0:
                continue          # exactly on the line: decide on a later frame
            direction = crossing_direction(self.last_side.get(track_id), side)
            self.last_side[track_id] = side

            if direction and is_within_segment(centre, self.line_start, self.line_end):
                counted = self.counted_in if direction == "IN" else self.counted_out
                if track_id not in counted:    # at most once per ID per direction
                    counted.add(track_id)
                    events.append((track_id, direction))
        return events

    def draw(self, frame, active_ids):
        """Draw the counting line, an IN arrow, trails of visible IDs and the IN/OUT totals."""
        a = tuple(int(v) for v in self.line_start)
        b = tuple(int(v) for v in self.line_end)
        cv2.line(frame, a, b, LINE_COLOUR, 2)

        # Arrow from the line's middle towards the positive ("IN") side.
        # (-dy, dx) is perpendicular to AB and points to the side where the cross product is > 0.
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        tip = (int(mid[0] - dy / length * 40), int(mid[1] + dx / length * 40))
        cv2.arrowedLine(frame, mid, tip, LINE_COLOUR, 2, tipLength=0.3)
        cv2.putText(frame, "IN", (tip[0] + 5, tip[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOUR, 2)

        for track_id in active_ids:                    # only people visible in this frame
            points = list(self.trails.get(track_id, []))
            for p, q in zip(points, points[1:]):       # join consecutive centre points
                cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour_for_id(track_id), 2)

        cv2.putText(frame, f"IN: {self.in_count}  OUT: {self.out_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOUR, 2)

    def rows(self):
        """One dict per track ID, ready to become a CSV row."""
        rows = []
        for track_id in sorted(self.first_seen):
            directions = [d for d, ids in (("IN", self.counted_in), ("OUT", self.counted_out)) if track_id in ids]
            rows.append({
                "track_id": track_id,
                "first_seen_s": round(self.first_seen[track_id], 2),
                "last_seen_s": round(self.last_seen[track_id], 2),
                "seconds_in_frame": round(self.seconds_in_frame(track_id), 2),
                "frames_seen": self.frames_seen[track_id],
                "direction_crossed": "+".join(directions),   # "IN", "OUT", "IN+OUT" or ""
            })
        return rows
