"""
Stock Chart Pattern & Edge Detection Engine
Pure-Python implementation using OpenCV, NumPy, SciPy, and Pillow.

Detects:
  - Double Top ("Double Header")
  - Double Bottom ("W" Pattern)
  - Head and Shoulders (Bearish)
  - Inverse Head and Shoulders (Bullish)
  - Triangles (Ascending, Descending, Symmetrical)

Computes exact matching percentage (0 - 100%) and marks patterns directly on the chart image.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, savgol_filter


@dataclass
class Point:
    x: int
    y: int
    label: str = ""
    is_peak: bool = True


@dataclass
class DetectedPattern:
    name: str
    pattern_type: str  # 'bearish_reversal', 'bullish_reversal', 'continuation'
    match_percentage: float
    points: List[Point]
    neckline: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


class ChartPreprocessor:
    """Preprocesses a chart screenshot, detects edges, suppresses gridlines,

    and extracts a clean 1D price trajectory across time X.
    """

    def __init__(self, target_width: int = 1000):
        self.target_width = target_width

    def preprocess(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs multi-stage edge detection and price curve extraction."""
        h, w = img_bgr.shape[:2]

        # 1. Scale image if needed
        scale = 1.0
        if w > self.target_width or w < 600:
            scale = self.target_width / float(w)
            img_bgr = cv2.resize(
                img_bgr,
                (self.target_width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            h, w = img_bgr.shape[:2]

        # 2. Determine background color
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        is_dark_theme = mean_brightness < 128

        # 3. Candlestick and Trend Line Colors Isolation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Bullish Green
        green_mask = cv2.inRange(
            hsv, np.array([35, 30, 30]), np.array([90, 255, 255])
        )
        # Bearish Red
        red_mask1 = cv2.inRange(
            hsv, np.array([0, 40, 40]), np.array([12, 255, 255])
        )
        red_mask2 = cv2.inRange(
            hsv, np.array([168, 40, 40]), np.array([180, 255, 255])
        )
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        candle_mask = cv2.bitwise_or(green_mask, red_mask)

        # 4. Canny Edge Detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 130)

        # 5. Suppress Long Horizontal & Vertical Grid Lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        grid_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        grid_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        grid_lines = cv2.bitwise_or(grid_h, grid_v)
        clean_edges = cv2.subtract(edges, grid_lines)

        # Combine clean edges with candlestick color masks
        price_mask = cv2.bitwise_or(clean_edges, candle_mask)

        # Active chart region (excluding outer axis margins)
        margin_top = int(h * 0.05)
        margin_bottom = int(h * 0.92)
        margin_left = int(w * 0.03)
        margin_right = int(w * 0.95)

        active_mask = np.zeros_like(price_mask)
        active_mask[
            margin_top:margin_bottom, margin_left:margin_right
        ] = price_mask[margin_top:margin_bottom, margin_left:margin_right]

        # 6. Extract Continuous 1D Price Trajectory
        x_coords = []
        y_coords = []

        for x in range(margin_left, margin_right):
            col = active_mask[margin_top:margin_bottom, x]
            active_ys = np.where(col > 0)[0]
            if len(active_ys) > 0:
                y_val = margin_top + float(np.median(active_ys))
                x_coords.append(x)
                y_coords.append(y_val)

        full_x = np.arange(margin_left, margin_right)
        if len(x_coords) > 20:
            full_y = np.interp(full_x, x_coords, y_coords)
            # Savitzky-Golay smoothing for noise removal
            window_len = min(21, len(full_y) if len(full_y) % 2 != 0 else len(full_y) - 1)
            if window_len > 5:
                try:
                    smooth_y = savgol_filter(full_y, window_length=window_len, polyorder=2)
                except Exception:
                    smooth_y = full_y
            else:
                smooth_y = full_y
        else:
            full_y = np.full_like(full_x, h / 2.0, dtype=float)
            smooth_y = full_y

        return {
            "image": img_bgr,
            "gray": gray,
            "edges": edges,
            "clean_edges": clean_edges,
            "price_mask": active_mask,
            "x": full_x,
            "y": smooth_y,
            "raw_y": full_y,
            "height": h,
            "width": w,
            "scale": scale,
            "is_dark_theme": is_dark_theme,
            "margins": (margin_left, margin_top, margin_right, margin_bottom),
        }


class ExtremaDetector:
    """Extracts local swing highs (peaks) and swing lows (valleys)."""

    def __init__(self, prominence_ratio: float = 0.035, min_distance: int = 20):
        self.prominence_ratio = prominence_ratio
        self.min_distance = min_distance

    def find_extrema(
        self, x: np.ndarray, y: np.ndarray, chart_height: int
    ) -> Tuple[List[Point], List[Point], List[Point]]:
        # Inverted price: lower y in image coords = higher price
        price_curve = -y
        y_range = max(1.0, float(np.max(price_curve) - np.min(price_curve)))
        prominence = max(8.0, y_range * self.prominence_ratio)

        peak_indices, _ = find_peaks(
            price_curve, distance=self.min_distance, prominence=prominence
        )
        peaks = [
            Point(x=int(x[idx]), y=int(round(y[idx])), label="Peak", is_peak=True)
            for idx in peak_indices
        ]

        valley_indices, _ = find_peaks(
            -price_curve, distance=self.min_distance, prominence=prominence
        )
        valleys = [
            Point(x=int(x[idx]), y=int(round(y[idx])), label="Valley", is_peak=False)
            for idx in valley_indices
        ]

        all_extrema = sorted(peaks + valleys, key=lambda pt: pt.x)
        return peaks, valleys, all_extrema


class PatternMatcher:
    """Structural Pattern Matcher with Quantitative Matching Percentages."""

    def __init__(self, min_match_percentage: float = 60.0):
        self.min_match_percentage = min_match_percentage

    def match_all(
        self,
        peaks: List[Point],
        valleys: List[Point],
        all_extrema: List[Point],
        x_series: np.ndarray,
        y_series: np.ndarray,
        chart_height: int,
    ) -> List[DetectedPattern]:
        patterns: List[DetectedPattern] = []

        # 1. Match Head & Shoulders first (higher structural complexity)
        hs_patterns = self._match_head_and_shoulders(
            peaks, valleys, chart_height
        )
        patterns.extend(hs_patterns)

        # 2. Match Inverse Head & Shoulders
        ihs_patterns = self._match_inverse_head_and_shoulders(
            peaks, valleys, chart_height
        )
        patterns.extend(ihs_patterns)

        # 3. Match Double Top ("Double Header")
        dt_patterns = self._match_double_top(
            peaks, valleys, chart_height, existing_hs=hs_patterns
        )
        patterns.extend(dt_patterns)

        # 4. Match Double Bottom ("W" Pattern)
        db_patterns = self._match_double_bottom(
            peaks, valleys, chart_height, existing_ihs=ihs_patterns, existing_hs=hs_patterns
        )
        patterns.extend(db_patterns)

        # 5. Match Triangles
        tri_patterns = self._match_triangles(all_extrema, chart_height)
        patterns.extend(tri_patterns)

        # Filter by minimum score and deduplicate / NMS
        valid = [
            p for p in patterns if p.match_percentage >= self.min_match_percentage
        ]
        deduped = self._non_max_suppression(valid)
        deduped.sort(key=lambda p: p.match_percentage, reverse=True)
        return deduped

    def _match_double_top(
        self,
        peaks: List[Point],
        valleys: List[Point],
        chart_height: int,
        existing_hs: List[DetectedPattern],
    ) -> List[DetectedPattern]:
        """Detects Double Top ('Double Header') Bearish Reversal Pattern."""
        results = []
        n_peaks = len(peaks)

        # Collect points belonging to Head & Shoulders to prevent overlapping misclassifications
        hs_head_xs = []
        hs_shoulder_pairs = []
        for hs in existing_hs:
            h_pt = next((pt for pt in hs.points if "Head" in pt.label), None)
            s_pts = [pt for pt in hs.points if "Shoulder" in pt.label]
            if h_pt:
                hs_head_xs.append(h_pt.x)
            if len(s_pts) == 2:
                hs_shoulder_pairs.append((min(s_pts[0].x, s_pts[1].x), max(s_pts[0].x, s_pts[1].x)))

        for i in range(n_peaks):
            for j in range(i + 1, n_peaks):
                p1, p2 = peaks[i], peaks[j]
                span_x = p2.x - p1.x

                if span_x < 40 or span_x > 650:
                    continue

                # If an H&S head lies between these two peaks, do not treat as double top
                if any(p1.x < hx < p2.x for hx in hs_head_xs):
                    continue

                # If these peaks match the shoulder peaks of an H&S, reject
                if any(abs(p1.x - sp[0]) < 20 and abs(p2.x - sp[1]) < 20 for sp in hs_shoulder_pairs):
                    continue

                # Rejection check: no intermediate peak between p1 and p2 should be higher than both
                intermediate_peaks = [
                    p for p in peaks if p1.x < p.x < p2.x
                ]
                if any(p.y < min(p1.y, p2.y) - 8 for p in intermediate_peaks):
                    continue

                # Find deepest valley between the two peaks
                inter_valleys = [v for v in valleys if p1.x < v.x < p2.x]
                if not inter_valleys:
                    continue

                valley = max(inter_valleys, key=lambda v: v.y)
                avg_peak_y = (p1.y + p2.y) / 2.0
                trough_depth = valley.y - avg_peak_y

                if trough_depth < 20:
                    continue

                # 1. Peak Parity Score (0 to 100)
                peak_diff = abs(p1.y - p2.y)
                rel_diff = peak_diff / max(1.0, trough_depth)
                parity_score = max(0.0, 100.0 * (1.0 - rel_diff / 0.18))

                # 2. Trough Depth Score (0 to 100)
                depth_ratio = trough_depth / float(chart_height)
                depth_score = min(
                    100.0, max(25.0, (depth_ratio / 0.16) * 100.0)
                )

                # 3. Horizontal Time Symmetry Score (0 to 100)
                dist1 = abs(valley.x - p1.x)
                dist2 = abs(p2.x - valley.x)
                sym_ratio = min(dist1, dist2) / max(1.0, max(dist1, dist2))
                symmetry_score = max(0.0, sym_ratio * 100.0)

                match_pct = (
                    0.50 * parity_score
                    + 0.30 * depth_score
                    + 0.20 * symmetry_score
                )

                if match_pct >= self.min_match_percentage:
                    xmin = p1.x - 15
                    xmax = p2.x + 15
                    ymin = min(p1.y, p2.y) - 18
                    ymax = valley.y + 18

                    pts = [
                        Point(p1.x, p1.y, "Top 1 (Header 1)", True),
                        Point(valley.x, valley.y, "Neckline Support", False),
                        Point(p2.x, p2.y, "Top 2 (Header 2)", True),
                    ]
                    neckline = ((xmin, valley.y), (xmax, valley.y))

                    results.append(
                        DetectedPattern(
                            name="Double Top (Double Header)",
                            pattern_type="bearish_reversal",
                            match_percentage=round(match_pct, 1),
                            points=pts,
                            neckline=neckline,
                            bbox=(xmin, ymin, xmax, ymax),
                            description="Bearish reversal: two consecutive peaks at similar resistance levels with neckline breakdown potential.",
                            metrics={
                                "Peak Parity": round(parity_score, 1),
                                "Trough Depth": round(depth_score, 1),
                                "Time Symmetry": round(symmetry_score, 1),
                            },
                        )
                    )

        return results

    def _match_head_and_shoulders(
        self, peaks: List[Point], valleys: List[Point], chart_height: int
    ) -> List[DetectedPattern]:
        """Detects Head and Shoulders (Bearish Reversal)."""
        results = []
        n_peaks = len(peaks)

        for i in range(n_peaks):
            for j in range(i + 1, n_peaks):
                for k in range(j + 1, n_peaks):
                    s1, head, s2 = peaks[i], peaks[j], peaks[k]
                    span_x = s2.x - s1.x
                    if span_x < 60 or span_x > 750:
                        continue

                    # In image coordinates: smaller y = higher price
                    # Head MUST be distinctly higher than both shoulders
                    if not (head.y < s1.y - 12 and head.y < s2.y - 12):
                        continue

                    # Verify head is highest point between s1 and s2
                    inter_peaks = [
                        p for p in peaks if s1.x < p.x < s2.x
                    ]
                    if any(p.y < head.y for p in inter_peaks):
                        continue

                    # Find Left Valley (v1) between s1 and head
                    v1_list = [v for v in valleys if s1.x < v.x < head.x]
                    # Find Right Valley (v2) between head and s2
                    v2_list = [v for v in valleys if head.x < v.x < s2.x]

                    if not v1_list or not v2_list:
                        continue

                    v1 = max(v1_list, key=lambda v: v.y)
                    v2 = max(v2_list, key=lambda v: v.y)

                    # Valley depths relative to shoulders
                    if v1.y < s1.y + 12 or v2.y < s2.y + 12:
                        continue

                    # 1. Shoulder Parity Score (0 to 100)
                    shoulder_diff = abs(s1.y - s2.y)
                    avg_shoulder_drop = abs(
                        (v1.y + v2.y) / 2.0 - (s1.y + s2.y) / 2.0
                    )
                    rel_sh_diff = shoulder_diff / max(1.0, avg_shoulder_drop)
                    shoulder_score = max(0.0, 100.0 * (1.0 - rel_sh_diff / 0.30))

                    # 2. Head Prominence Score (0 to 100)
                    min_prominence = min(s1.y - head.y, s2.y - head.y)
                    head_score = min(
                        100.0,
                        max(20.0, (min_prominence / max(1.0, avg_shoulder_drop * 0.45)) * 100.0),
                    )

                    # 3. Neckline Slope Score (0 to 100)
                    nl_dy = abs(v1.y - v2.y)
                    nl_dx = max(1.0, abs(v2.x - v1.x))
                    slope = nl_dy / nl_dx
                    neckline_score = max(0.0, 100.0 * (1.0 - slope / 0.22))

                    # 4. Spacing / Time Symmetry Score (0 to 100)
                    w_left = abs(head.x - s1.x)
                    w_right = abs(s2.x - head.x)
                    time_sym = min(w_left, w_right) / max(1.0, max(w_left, w_right))
                    spacing_score = max(0.0, time_sym * 100.0)

                    match_pct = (
                        0.35 * shoulder_score
                        + 0.30 * head_score
                        + 0.20 * neckline_score
                        + 0.15 * spacing_score
                    )

                    if match_pct >= self.min_match_percentage:
                        xmin = s1.x - 15
                        xmax = s2.x + 15
                        ymin = head.y - 18
                        ymax = max(v1.y, v2.y) + 18

                        # Extrapolate neckline line
                        m = (v2.y - v1.y) / max(1.0, v2.x - v1.x)
                        nl_y1 = int(round(v1.y + m * (xmin - v1.x)))
                        nl_y2 = int(round(v2.y + m * (xmax - v2.x)))
                        neckline = ((xmin, nl_y1), (xmax, nl_y2))

                        pts = [
                            Point(s1.x, s1.y, "Left Shoulder", True),
                            Point(v1.x, v1.y, "Left Valley", False),
                            Point(head.x, head.y, "Head (Peak)", True),
                            Point(v2.x, v2.y, "Right Valley", False),
                            Point(s2.x, s2.y, "Right Shoulder", True),
                        ]

                        results.append(
                            DetectedPattern(
                                name="Head and Shoulders",
                                pattern_type="bearish_reversal",
                                match_percentage=round(match_pct, 1),
                                points=pts,
                                neckline=neckline,
                                bbox=(xmin, ymin, xmax, ymax),
                                description="Major bearish reversal: prominent central head peak flanked by two lower shoulders with neckline support.",
                                metrics={
                                    "Shoulder Parity": round(shoulder_score, 1),
                                    "Head Prominence": round(head_score, 1),
                                    "Neckline Slope": round(neckline_score, 1),
                                    "Time Symmetry": round(spacing_score, 1),
                                },
                            )
                        )

        return results

    def _match_double_bottom(
        self,
        peaks: List[Point],
        valleys: List[Point],
        chart_height: int,
        existing_ihs: List[DetectedPattern],
        existing_hs: List[DetectedPattern],
    ) -> List[DetectedPattern]:
        """Detects Double Bottom ('W' Bullish Reversal Pattern)."""
        results = []
        n_valleys = len(valleys)

        # Exclude valleys that belong to an existing Head & Shoulders or Inverse H&S
        excluded_valley_xs = set()
        for ihs in existing_ihs:
            for pt in ihs.points:
                if not pt.is_peak:
                    excluded_valley_xs.add(pt.x)
        for hs in existing_hs:
            for pt in hs.points:
                if not pt.is_peak:
                    excluded_valley_xs.add(pt.x)

        for i in range(n_valleys):
            for j in range(i + 1, n_valleys):
                v1, v2 = valleys[i], valleys[j]
                span_x = v2.x - v1.x
                if span_x < 40 or span_x > 650:
                    continue

                # Exclude if both valleys are part of H&S neckline
                if v1.x in excluded_valley_xs and v2.x in excluded_valley_xs:
                    continue

                # Rejection: no intermediate valley deeper than both
                inter_valleys = [v for v in valleys if v1.x < v.x < v2.x]
                if any(v.y > max(v1.y, v2.y) + 8 for v in inter_valleys):
                    continue

                # Find highest rebound peak between the two troughs
                inter_peaks = [p for p in peaks if v1.x < p.x < v2.x]
                if not inter_peaks:
                    continue

                peak = min(inter_peaks, key=lambda p: p.y)
                avg_bottom_y = (v1.y + v2.y) / 2.0
                rebound_height = avg_bottom_y - peak.y

                if rebound_height < 20:
                    continue

                # 1. Bottom Parity Score (0 to 100)
                valley_diff = abs(v1.y - v2.y)
                rel_diff = valley_diff / max(1.0, rebound_height)
                parity_score = max(0.0, 100.0 * (1.0 - rel_diff / 0.18))

                # 2. Rebound Height Score (0 to 100)
                rebound_ratio = rebound_height / float(chart_height)
                rebound_score = min(
                    100.0, max(25.0, (rebound_ratio / 0.16) * 100.0)
                )

                # 3. Time Symmetry Score (0 to 100)
                dist1 = abs(peak.x - v1.x)
                dist2 = abs(v2.x - peak.x)
                sym_ratio = min(dist1, dist2) / max(1.0, max(dist1, dist2))
                symmetry_score = max(0.0, sym_ratio * 100.0)

                match_pct = (
                    0.50 * parity_score
                    + 0.30 * rebound_score
                    + 0.20 * symmetry_score
                )

                if match_pct >= self.min_match_percentage:
                    xmin = v1.x - 15
                    xmax = v2.x + 15
                    ymin = peak.y - 18
                    ymax = max(v1.y, v2.y) + 18

                    pts = [
                        Point(v1.x, v1.y, "Bottom 1", False),
                        Point(peak.x, peak.y, "Neckline Resistance", True),
                        Point(v2.x, v2.y, "Bottom 2", False),
                    ]
                    neckline = ((xmin, peak.y), (xmax, peak.y))

                    results.append(
                        DetectedPattern(
                            name="Double Bottom ('W' Pattern)",
                            pattern_type="bullish_reversal",
                            match_percentage=round(match_pct, 1),
                            points=pts,
                            neckline=neckline,
                            bbox=(xmin, ymin, xmax, ymax),
                            description="Bullish reversal: two consecutive troughs at similar support levels separated by a central rebound peak.",
                            metrics={
                                "Bottom Parity": round(parity_score, 1),
                                "Rebound Strength": round(rebound_score, 1),
                                "Time Symmetry": round(symmetry_score, 1),
                            },
                        )
                    )

        return results

    def _match_inverse_head_and_shoulders(
        self, peaks: List[Point], valleys: List[Point], chart_height: int
    ) -> List[DetectedPattern]:
        """Detects Inverse Head and Shoulders (Bullish Reversal)."""
        results = []
        n_valleys = len(valleys)

        for i in range(n_valleys):
            for j in range(i + 1, n_valleys):
                for k in range(j + 1, n_valleys):
                    s1, head, s2 = valleys[i], valleys[j], valleys[k]
                    span_x = s2.x - s1.x
                    if span_x < 60 or span_x > 750:
                        continue

                    # In image coordinates: larger y = lower price
                    # Inverse Head MUST be distinctly deeper than both shoulders
                    if not (head.y > s1.y + 12 and head.y > s2.y + 12):
                        continue

                    inter_valleys = [
                        v for v in valleys if s1.x < v.x < s2.x
                    ]
                    if any(v.y > head.y for v in inter_valleys):
                        continue

                    # Intermediate peaks forming neckline
                    p1_list = [p for p in peaks if s1.x < p.x < head.x]
                    p2_list = [p for p in peaks if head.x < p.x < s2.x]
                    if not p1_list or not p2_list:
                        continue

                    p1 = min(p1_list, key=lambda p: p.y)
                    p2 = min(p2_list, key=lambda p: p.y)

                    if p1.y > s1.y - 12 or p2.y > s2.y - 12:
                        continue

                    # Parity score
                    sh_diff = abs(s1.y - s2.y)
                    avg_sh_rise = abs(
                        (s1.y + s2.y) / 2.0 - (p1.y + p2.y) / 2.0
                    )
                    rel_diff = sh_diff / max(1.0, avg_sh_rise)
                    shoulder_score = max(0.0, 100.0 * (1.0 - rel_diff / 0.30))

                    # Head depth score
                    min_depth = min(head.y - s1.y, head.y - s2.y)
                    head_score = min(
                        100.0,
                        max(20.0, (min_depth / max(1.0, avg_sh_rise * 0.45)) * 100.0),
                    )

                    # Neckline slope
                    nl_dy = abs(p1.y - p2.y)
                    nl_dx = max(1.0, abs(p2.x - p1.x))
                    slope = nl_dy / nl_dx
                    neckline_score = max(0.0, 100.0 * (1.0 - slope / 0.22))

                    # Spacing
                    w_left = abs(head.x - s1.x)
                    w_right = abs(s2.x - head.x)
                    time_sym = min(w_left, w_right) / max(1.0, max(w_left, w_right))
                    spacing_score = max(0.0, time_sym * 100.0)

                    match_pct = (
                        0.35 * shoulder_score
                        + 0.30 * head_score
                        + 0.20 * neckline_score
                        + 0.15 * spacing_score
                    )

                    if match_pct >= self.min_match_percentage:
                        xmin = s1.x - 15
                        xmax = s2.x + 15
                        ymin = min(p1.y, p2.y) - 18
                        ymax = head.y + 18

                        m = (p2.y - p1.y) / max(1.0, p2.x - p1.x)
                        nl_y1 = int(round(p1.y + m * (xmin - p1.x)))
                        nl_y2 = int(round(p2.y + m * (xmax - p2.x)))
                        neckline = ((xmin, nl_y1), (xmax, nl_y2))

                        pts = [
                            Point(s1.x, s1.y, "Left Shoulder", False),
                            Point(p1.x, p1.y, "Neckline Peak 1", True),
                            Point(head.x, head.y, "Inverse Head", False),
                            Point(p2.x, p2.y, "Neckline Peak 2", True),
                            Point(s2.x, s2.y, "Right Shoulder", False),
                        ]

                        results.append(
                            DetectedPattern(
                                name="Inverse Head and Shoulders",
                                pattern_type="bullish_reversal",
                                match_percentage=round(match_pct, 1),
                                points=pts,
                                neckline=neckline,
                                bbox=(xmin, ymin, xmax, ymax),
                                description="Major bullish reversal: deep central trough flanked by two higher shoulder troughs with neckline resistance.",
                                metrics={
                                    "Shoulder Parity": round(shoulder_score, 1),
                                    "Head Prominence": round(head_score, 1),
                                    "Neckline Slope": round(neckline_score, 1),
                                    "Time Symmetry": round(spacing_score, 1),
                                },
                            )
                        )

        return results

    def _match_triangles(
        self, all_extrema: List[Point], chart_height: int
    ) -> List[DetectedPattern]:
        """Detects genuine Triangle consolidation patterns with contracting amplitude."""
        results = []
        n = len(all_extrema)
        if n < 5:
            return results

        for i in range(n - 4):
            window = all_extrema[i : i + 5]
            w_peaks = [p for p in window if p.is_peak]
            w_valleys = [p for p in window if not p.is_peak]

            # Require alternating structure with at least 2 peaks and 2 valleys
            if len(w_peaks) < 2 or len(w_valleys) < 2:
                continue

            # Check that amplitude is actually contracting (consolidation)
            amp1 = abs(w_peaks[0].y - w_valleys[0].y)
            amp2 = abs(w_peaks[1].y - w_valleys[1].y)
            if amp2 >= amp1 * 0.95 or amp1 < 25 or amp2 < 15:
                continue

            peak_dy = w_peaks[1].y - w_peaks[0].y
            peak_dx = max(1.0, w_peaks[1].x - w_peaks[0].x)
            p_slope = peak_dy / peak_dx

            val_dy = w_valleys[1].y - w_valleys[0].y
            val_dx = max(1.0, w_valleys[1].x - w_valleys[0].x)
            v_slope = val_dy / val_dx

            # Symmetrical: Lower Highs (p_slope > 0.08) and Higher Lows (v_slope < -0.08)
            if p_slope > 0.08 and v_slope < -0.08:
                conv = min(100.0, (abs(p_slope) + abs(v_slope)) * 220.0)
                match_pct = round(min(90.0, max(65.0, conv)), 1)
                name = "Symmetrical Triangle"
                ptype = "continuation"
            elif abs(p_slope) <= 0.06 and v_slope < -0.08:
                match_pct = round(
                    max(65.0, 100.0 * (1.0 - abs(p_slope) / 0.08)), 1
                )
                name = "Ascending Triangle"
                ptype = "bullish_continuation"
            elif p_slope > 0.08 and abs(v_slope) <= 0.06:
                match_pct = round(
                    max(65.0, 100.0 * (1.0 - abs(v_slope) / 0.08)), 1
                )
                name = "Descending Triangle"
                ptype = "bearish_continuation"
            else:
                continue

            xmin = min(p.x for p in window) - 10
            xmax = max(p.x for p in window) + 10
            ymin = min(p.y for p in window) - 15
            ymax = max(p.y for p in window) + 15

            results.append(
                DetectedPattern(
                    name=name,
                    pattern_type=ptype,
                    match_percentage=match_pct,
                    points=[Point(p.x, p.y, p.label, p.is_peak) for p in window],
                    neckline=None,
                    bbox=(xmin, ymin, xmax, ymax),
                    description="Consolidation pattern with converging boundary trendlines.",
                    metrics={"Convergence Score": match_pct},
                )
            )

        return results

    def _non_max_suppression(
        self, patterns: List[DetectedPattern], iou_threshold: float = 0.35
    ) -> List[DetectedPattern]:
        """Suppresses overlapping duplicate detections, prioritizing major reversal patterns."""
        if not patterns:
            return []

        # Reversal patterns get precedence over generic triangles
        priority_weights = {
            "Head and Shoulders": 15.0,
            "Inverse Head and Shoulders": 15.0,
            "Double Top (Double Header)": 10.0,
            "Double Bottom ('W' Pattern)": 10.0,
            "Ascending Triangle": 0.0,
            "Descending Triangle": 0.0,
            "Symmetrical Triangle": 0.0,
        }

        # Sort descending by priority-adjusted match percentage
        sorted_pats = sorted(
            patterns,
            key=lambda p: p.match_percentage + priority_weights.get(p.name, 0.0),
            reverse=True,
        )
        kept: List[DetectedPattern] = []

        for cand in sorted_pats:
            cx1, cy1, cx2, cy2 = cand.bbox
            c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            suppress = False

            for k in kept:
                kx1, ky1, kx2, ky2 = k.bbox
                k_area = max(1, (kx2 - kx1) * (ky2 - ky1))

                ix1 = max(cx1, kx1)
                iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2)
                iy2 = min(cy2, ky2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    iou = inter_area / float(c_area + k_area - inter_area)
                    overlap_ratio = inter_area / float(min(c_area, k_area))

                    if iou > iou_threshold or overlap_ratio > 0.60:
                        suppress = True
                        break

            if not suppress:
                kept.append(cand)

        return kept


class ChartAnnotator:
    """Renders visual pattern markings, necklines, points, and percentage badges directly on image."""

    COLOR_PALETTE = {
        "bearish_reversal": (240, 68, 85),  # Coral / Red
        "bullish_reversal": (38, 166, 91),  # Emerald Green
        "continuation": (66, 133, 244),  # Blue
        "gold": (255, 193, 7),  # Gold accent
        "cyan": (0, 229, 255),  # Sky blue
    }

    def annotate(
        self,
        base_img_bgr: np.ndarray,
        patterns: List[DetectedPattern],
        extrema: List[Point],
        x_series: Optional[np.ndarray] = None,
        y_series: Optional[np.ndarray] = None,
        draw_trajectory: bool = True,
        draw_all_extrema: bool = False,
    ) -> np.ndarray:
        annotated = base_img_bgr.copy()
        h, w = annotated.shape[:2]

        # 1. Price trajectory line
        if draw_trajectory and x_series is not None and y_series is not None:
            curve_pts = np.vstack((x_series, y_series)).T.astype(np.int32)
            cv2.polylines(
                annotated,
                [curve_pts],
                isClosed=False,
                color=(255, 185, 0),  # Blue/Cyan
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # 2. All extrema markers (optional)
        if draw_all_extrema:
            for pt in extrema:
                color = (0, 220, 0) if pt.is_peak else (0, 120, 255)
                cv2.circle(annotated, (pt.x, pt.y), 4, color, -1, cv2.LINE_AA)

        # 3. Draw each detected pattern
        for pat in patterns:
            theme_color_rgb = self.COLOR_PALETTE.get(
                pat.pattern_type, (255, 193, 7)
            )
            # OpenCV BGR
            theme_bgr = (
                theme_color_rgb[2],
                theme_color_rgb[1],
                theme_color_rgb[0],
            )

            xmin, ymin, xmax, ymax = pat.bbox
            xmin, xmax = max(0, xmin), min(w - 1, xmax)
            ymin, ymax = max(0, ymin), min(h - 1, ymax)

            # Translucent Highlight Box
            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (xmin, ymin),
                (xmax, ymax),
                theme_bgr,
                thickness=-1,
            )
            cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
            cv2.rectangle(
                annotated,
                (xmin, ymin),
                (xmax, ymax),
                theme_bgr,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Pattern Skeleton connecting vertices
            pts_arr = np.array([[pt.x, pt.y] for pt in pat.points], dtype=np.int32)
            cv2.polylines(
                annotated,
                [pts_arr],
                isClosed=False,
                color=(255, 255, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                annotated,
                [pts_arr],
                isClosed=False,
                color=theme_bgr,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Draw Neckline
            if pat.neckline is not None:
                (nl_p1, nl_p2) = pat.neckline
                # Bold Gold Neckline
                cv2.line(
                    annotated,
                    nl_p1,
                    nl_p2,
                    (0, 220, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                mid_x = (nl_p1[0] + nl_p2[0]) // 2
                mid_y = (nl_p1[1] + nl_p2[1]) // 2
                cv2.putText(
                    annotated,
                    "Neckline",
                    (mid_x - 30, mid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw Vertices and Labels
            for pt in pat.points:
                cv2.circle(annotated, (pt.x, pt.y), 7, theme_bgr, -1, cv2.LINE_AA)
                cv2.circle(annotated, (pt.x, pt.y), 3, (255, 255, 255), -1, cv2.LINE_AA)
                label_y = pt.y - 12 if pt.is_peak else pt.y + 18
                cv2.putText(
                    annotated,
                    pt.label,
                    (pt.x - 22, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Top Badge with Matching Percentage
            badge_text = f"{pat.name}: {pat.match_percentage:.1f}% MATCH"
            badge_x = xmin + 6
            badge_y = max(24, ymin - 8)

            (tw, th), _ = cv2.getTextSize(
                badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
            )
            cv2.rectangle(
                annotated,
                (badge_x - 4, badge_y - th - 6),
                (badge_x + tw + 6, badge_y + 4),
                (18, 22, 28),
                -1,
            )
            cv2.rectangle(
                annotated,
                (badge_x - 4, badge_y - th - 6),
                (badge_x + tw + 6, badge_y + 4),
                theme_bgr,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                badge_text,
                (badge_x, badge_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated


class ChartPatternDetector:
    """Complete Pipeline Orchestrator."""

    def __init__(
        self,
        min_match_percentage: float = 60.0,
        prominence_ratio: float = 0.035,
        target_width: int = 1000,
    ):
        self.preprocessor = ChartPreprocessor(target_width=target_width)
        self.extrema_detector = ExtremaDetector(
            prominence_ratio=prominence_ratio
        )
        self.matcher = PatternMatcher(
            min_match_percentage=min_match_percentage
        )
        self.annotator = ChartAnnotator()

    def detect_from_image(
        self,
        image_input: Any,
        draw_trajectory: bool = True,
        draw_all_extrema: bool = False,
    ) -> Dict[str, Any]:
        """Processes input chart screenshot and detects stock market patterns."""
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Unable to read image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = image_input.copy()
        else:
            raise TypeError("Unsupported image format")

        prep = self.preprocessor.preprocess(img_bgr)
        peaks, valleys, all_extrema = self.extrema_detector.find_extrema(
            prep["x"], prep["y"], prep["height"]
        )
        patterns = self.matcher.match_all(
            peaks, valleys, all_extrema, prep["x"], prep["y"], prep["height"]
        )

        annotated_bgr = self.annotator.annotate(
            prep["image"],
            patterns=patterns,
            extrema=all_extrema,
            x_series=prep["x"],
            y_series=prep["y"],
            draw_trajectory=draw_trajectory,
            draw_all_extrema=draw_all_extrema,
        )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(prep["image"], cv2.COLOR_BGR2RGB)

        return {
            "patterns": patterns,
            "extrema": all_extrema,
            "peaks": peaks,
            "valleys": valleys,
            "annotated_image": annotated_rgb,
            "original_image": original_rgb,
            "edges": prep["clean_edges"],
            "price_mask": prep["price_mask"],
            "x_series": prep["x"],
            "y_series": prep["y"],
            "is_dark_theme": prep["is_dark_theme"],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect stock market chart patterns in screenshots."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to chart screenshot image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_chart.png",
        help="Output annotated image path",
    )
    parser.add_argument(
        "--min_match",
        type=float,
        default=60.0,
        help="Minimum pattern match percentage (0-100)",
    )
    args = parser.parse_args()

    detector = ChartPatternDetector(min_match_percentage=args.min_match)
    print(f"Analyzing {args.image}...")
    res = detector.detect_from_image(args.image)

    patterns = res["patterns"]
    print(f"\nDetection complete! Found {len(patterns)} pattern(s):")
    for i, p in enumerate(patterns, 1):
        print(f" {i}. [{p.name}] - {p.match_percentage:.1f}% Match")
        print(f"    Type: {p.pattern_type}")
        print(f"    Description: {p.description}")
        for k, v in p.metrics.items():
            print(f"      * {k}: {v}%")

    out_img = Image.fromarray(res["annotated_image"])
    out_img.save(args.output)
    print(f"\nAnnotated chart saved to: {args.output}")
