"""Main-thread-only OpenCV display shared by both sequence runners."""

import os
from pathlib import Path

import numpy as np


class SequencePreview:
    def __init__(self, camera, *, name="Pick and place RGB-D"):
        import cv2
        self.cv2 = cv2
        self._bgr = None
        self.name = name
        fonts = Path("/usr/share/fonts/truetype/dejavu")
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.name, np.zeros((camera["height"], camera["width"], 3), dtype=np.uint8))
        self._last = None
        self._status = None
        if cv2.waitKey(1) & 255 in (27, ord("q")):
            self.close()
            raise RuntimeError("camera preview cancelled before sequence startup")

    def update(self, robot):
        cv2 = self.cv2
        frame = robot.latest_preview()
        if frame is not None and (frame is not self._last or robot.status != self._status):
            if frame is not self._last:
                self._bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            bgr = self._bgr.copy()
            cv2.putText(bgr, robot.status, (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 255, 255), 2)
            cv2.imshow(self.name, bgr)
            self._last, self._status = frame, robot.status
        key = cv2.waitKey(1) & 255
        if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            raise RuntimeError("sequence cancelled from camera window")

    def close(self):
        self.cv2.destroyWindow(self.name)
