"""Read label text off a crop with Apple Vision (on-device, macOS only).

Usage: python src/ocr.py <image> [...]   -> prints the text found in each image
"""
import sys
from pathlib import Path

import Quartz
import Vision
from Foundation import NSURL


def read_text(path: Path, min_confidence: float = 0.5) -> list[tuple[str, float]]:
    """Text lines in the image with Vision's confidence, most confident first."""
    url = NSURL.fileURLWithPath_(str(Path(path).resolve()))
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # brand names are not dictionary words
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return []
    lines = []
    for obs in request.results() or []:
        top = obs.topCandidates_(1)[0]
        if top.confidence() >= min_confidence:
            lines.append((str(top.string()), float(top.confidence())))
    return sorted(lines, key=lambda t: -t[1])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: " + " | ".join(f"{t} ({c:.2f})" for t, c in read_text(Path(p))))
