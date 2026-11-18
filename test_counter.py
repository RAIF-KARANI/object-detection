"""Unit checks for counter.py, using made-up points (no video, no model).

Run with:  python test_counter.py
Every expected value below can be worked out by hand (see docs/04_COUNTING_LOGIC.md).

Test line: A = (100, 200) -> B = (500, 200), a horizontal line drawn left -> right.
For this line, cross = 400 * (Py - 200), so:
    above the line (y < 200) -> -1,   below the line (y > 200) -> +1
    IN = moving DOWN across it,       OUT = moving UP across it
"""
from counter import LineCounter, crossing_direction, is_within_segment, side_of_line

A, B = (100, 200), (500, 200)


def test_side_of_line():
    """Cross-product sign gives the side; 0 exactly on the line."""
    assert side_of_line((300, 150), A, B) == -1   # 400*(-50) - 0*200 = -20000
    assert side_of_line((300, 260), A, B) == 1    # 400*60 = 24000
    assert side_of_line((300, 200), A, B) == 0    # on the line
    # A diagonal line (0,0) -> (4,4)
    assert side_of_line((4, 0), (0, 0), (4, 4)) == -1   # 4*0 - 4*4 = -16
    assert side_of_line((0, 4), (0, 0), (4, 4)) == 1    # 4*4 - 4*0 = +16


def test_crossing_direction():
    """A flip from -1 to +1 is IN, +1 to -1 is OUT, anything else is not a crossing."""
    assert crossing_direction(None, 1) is None    # first time we see this track
    assert crossing_direction(1, 1) is None       # same side
    assert crossing_direction(-1, 1) == "IN"
    assert crossing_direction(1, -1) == "OUT"
    assert crossing_direction(-1, 0) is None      # on the line: wait for next frame


def test_is_within_segment():
    """Only points alongside the segment (0 <= t <= 1) count."""
    assert is_within_segment((300, 250), A, B)        # t = 0.5, middle
    assert is_within_segment((100, 250), A, B)        # t = 0.0, at A
    assert not is_within_segment((600, 250), A, B)    # t = 1.25, past B


def test_simple_crossing_counts_in_once():
    """Walking down across the line gives exactly one IN."""
    c = LineCounter(A, B)
    events = []
    for t, y in enumerate([150, 180, 220, 260]):
        events += c.update([1], [(300, y)], t)
    assert events == [(1, "IN")]
    assert (c.in_count, c.out_count) == (1, 0)


def test_jitter_on_line_is_not_double_counted():
    """Wobbling back and forth over the line counts at most one IN and one OUT per ID."""
    c = LineCounter(A, B)
    for t, y in enumerate([190, 210, 195, 205, 195, 210]):   # down, up, down, up, down
        c.update([2], [(200, y)], t)
    assert (c.in_count, c.out_count) == (1, 1)


def test_crossing_past_end_of_line_is_ignored():
    """Crossing the infinite line beyond B (x = 700) must not count."""
    c = LineCounter(A, B)
    for t, y in enumerate([150, 250]):
        c.update([3], [(700, y)], t)
    assert (c.in_count, c.out_count) == (0, 0)


def test_point_exactly_on_line_still_counts():
    """Landing exactly on the line (side 0) then continuing is still one crossing."""
    c = LineCounter(A, B)
    for t, y in enumerate([150, 200, 250]):
        c.update([4], [(300, y)], t)
    assert (c.in_count, c.out_count) == (1, 0)


def test_two_people_counted_separately():
    """Two IDs crossing in opposite directions in the same frames."""
    c = LineCounter(A, B)
    c.update([5, 6], [(150, 150), (450, 250)], 0)   # ID 5 above, ID 6 below
    c.update([5, 6], [(150, 250), (450, 150)], 1)   # both cross
    assert c.counted_in == {5} and c.counted_out == {6}


def test_time_in_frame_and_rows():
    """Time in frame = last seen - first seen; CSV row records the direction."""
    c = LineCounter(A, B)
    for ts, y in [(0.0, 150), (0.5, 180), (1.0, 230)]:
        c.update([7], [(300, y)], ts)
    assert c.seconds_in_frame(7) == 1.0
    row = c.rows()[0]
    assert row == {"track_id": 7, "first_seen_s": 0.0, "last_seen_s": 1.0,
                   "seconds_in_frame": 1.0, "frames_seen": 3, "direction_crossed": "IN"}


def test_vertical_line_in_is_right_to_left():
    """Default config line: top -> bottom at x = 320 (640x480). Left half is +1, so right->left = IN."""
    top, bottom = (320, 0), (320, 480)
    assert side_of_line((100, 240), top, bottom) == 1     # cross = 0*240 - 480*(100-320) = +105600
    assert side_of_line((500, 240), top, bottom) == -1    # cross = -480*180 = -86400
    c = LineCounter(top, bottom)
    for t, x in enumerate([500, 400, 250, 150]):          # walking right -> left on screen
        c.update([9], [(x, 240)], t)
    assert (c.in_count, c.out_count) == (1, 0)


def test_trail_keeps_only_last_n_points():
    """Trail length is capped by deque(maxlen=N)."""
    c = LineCounter(A, B, trail_length=3)
    for t in range(5):
        c.update([8], [(100 + t, 100)], t)
    assert list(c.trails[8]) == [(102, 100), (103, 100), (104, 100)]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()                       # an AssertionError here stops the run and shows the line
        print(f"PASS  {test.__name__}")
    print(f"All {len(tests)} tests passed")
