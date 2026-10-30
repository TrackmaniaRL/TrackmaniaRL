"""Strict visual clock readback for the calibrated 1536x864 game HUD.

This reads rendered pixels; it never substitutes the requested TICK clock.
HUD pixels must be cropped out of the policy observation separately.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def glyphs(rgb):
    if rgb.shape != (864, 1536, 3):
        raise ValueError('HUD calibration requires 1536x864 RGB')
    mask = np.min(rgb[798:845, 682:854], axis=2) > 242
    seen = np.zeros_like(mask)
    components = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack, points = [(y, x)], []
        seen[y, x] = True
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = zip(*points)
        if max(ys)-min(ys) >= 28 and max(xs)-min(xs) >= 4:
            bitmap = np.zeros((max(ys)-min(ys)+1, max(xs)-min(xs)+1), dtype=np.uint8)
            for yy, xx in points:
                bitmap[yy-min(ys), xx-min(xs)] = 255
            normalized = np.asarray(Image.fromarray(bitmap).resize((20, 40), Image.Resampling.NEAREST)) > 127
            components.append((min(xs), normalized))
    if len(components) != 6:
        raise ValueError(f'expected six clock digits, found {len(components)}')
    return [x[1] for x in sorted(components, key=lambda item: item[0])]


class HudClock:
    def __init__(self, calibration):
        from tools.vision_qr1167.ink_clock import InkClock
        self.ink_clock = InkClock(calibration)
        self.templates = []
        self.outlines = []
        for digit in range(10):
            frame = np.asarray(Image.open(Path(calibration) / f'camera-{10000+digit*110}.png').convert('RGB'))
            self.templates.append(glyphs(frame)[-1])
            self.outlines.append(frame[798:845,825:849].max(2) < 90)

    def read(self, rgb):
        if rgb.shape != (864, 1536, 3):
            raise ValueError('HUD calibration requires 1536x864 RGB')
        digits, scores = [], []
        try:
            segmented = glyphs(rgb)
        except ValueError:
            segmented = [None] * 6
        for slot, glyph in enumerate(segmented):
            distances = [float(np.mean(glyph != template)) for template in self.templates] if glyph is not None else [1.] * 10
            order = np.argsort(distances)
            if distances[order[0]] > .16 or distances[order[1]] - distances[order[0]] < .035:
                # White road markings can join a digit's bright component.
                # Require a separately matching dark font outline, not a looser
                # foreground threshold and never the expected telemetry value.
                origin = (686, 710, 745, 768, 801, 825)[slot]
                outline_distances = []
                for template in self.outlines:
                    outline_distances.append(min(float(np.mean(
                        (rgb[798:845, origin+shift:origin+shift+24].max(2) < 90) != template))
                        for shift in (-1, 0, 1)))
                order = np.argsort(outline_distances)
                if outline_distances[order[0]] > .05 or outline_distances[order[1]]-outline_distances[order[0]] < .025:
                    return self.ink_clock.read(rgb)
                distances = outline_distances
            digits.append(int(order[0]))
            scores.append(distances[order[0]])
        minutes, seconds, centiseconds = (10*digits[i]+digits[i+1] for i in (0, 2, 4))
        if seconds >= 60:
            raise ValueError('invalid rendered seconds')
        return (minutes*60+seconds)*1000+centiseconds*10, scores
