"""Validated image-coordinate boxes and class-wise non-maximum suppression."""
import math


def clip_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        return None
    x0, y0, x1, y1 = [float(x) for x in box]
    x0, x1 = max(0., min(width, x0)), max(0., min(width, x1))
    y0, y1 = max(0., min(height, y0)), max(0., min(height, y1))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def suppress(boxes, scores, width, height, threshold=.5):
    if len(boxes) != len(scores) or not 0 <= threshold <= 1:
        raise ValueError("aligned boxes/scores and unit IoU threshold required")
    candidates = []
    for box, score in zip(boxes, scores):
        valid = clip_box(box, width, height)
        if valid is not None and math.isfinite(float(score)):
            candidates.append((valid, float(score)))
    selected = []
    for box, score in sorted(candidates, key=lambda x: -x[1]):
        if all(iou(box, kept) <= threshold for kept, _ in selected):
            selected.append((box, score))
    return selected


def overlay(image, boxes, label="candidate"):
    from PIL import ImageDraw
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for i, b in enumerate(boxes):
        box = clip_box(b, *image.size)
        if box is None:
            continue
        draw.rectangle(box, outline=(0, 160, 220), width=max(2, image.width//250))
        draw.text((box[0]+3, box[1]+3), f"{label} {i+1}", fill=(0, 160, 220))
    return result
