import os
import cv2
import base64
import math
import numpy as np
from ultralytics import YOLO


def compute_iou(boxA, boxB):
    """Calculate Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)


class AuraVisionEngine:
    def __init__(
        self,
        vehicle_weights: str = "weights/yolo26n.pt",
        pothole_weights: str = "weights/pothole_yolo26n.pt",
        expected_flow: str = "down"  # "down" = oncoming traffic moves down screen (increasing Y)
    ):
        self.vehicle_model = YOLO(vehicle_weights)
        self.has_pothole_model = os.path.exists(pothole_weights)
        self.pothole_model = YOLO(pothole_weights) if self.has_pothole_model else None

        self.expected_flow = expected_flow
        # Track history per vehicle ID: {track_id: [(cx, cy), ...]}
        self.track_history = {}
        self.vehicle_classes = {"car", "motorcycle", "bus", "truck"}

    def analyze_frame(self, frame: np.ndarray):
        detections = []
        annotated = frame.copy()
        active_vehicles = []

        # ---------------------------------------------------------
        # 1. POTHOLE DETECTION (Custom YOLO26n)
        # ---------------------------------------------------------
        if self.has_pothole_model:
            p_results = self.pothole_model.predict(frame, conf=0.40, verbose=False)[0]
            if p_results.boxes is not None:
                for box in p_results.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    severity = "CRITICAL" if area > 18000 else "HIGH"

                    detections.append({
                        "hazard_type": "pothole",
                        "severity": severity,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })

                    # Amber-Red tactical bounding box
                    color = (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"POTHOLE {conf:.0%}",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        # ---------------------------------------------------------
        # 2. VEHICLE TRACKING & WRONG-WAY DETECTION
        # ---------------------------------------------------------
        v_results = self.vehicle_model.track(frame, persist=True, conf=0.45, verbose=False)[0]
        if v_results.boxes is not None:
            for box in v_results.boxes:
                cls_id = int(box.cls[0])
                label = self.vehicle_model.names[cls_id]
                if label not in self.vehicle_classes:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                speed = 0.0
                is_wrong_way = False

                if track_id != -1:
                    history = self.track_history.setdefault(track_id, [])
                    history.append((cx, cy))
                    if len(history) > 15:
                        history.pop(0)

                    if len(history) >= 6:
                        old_x, old_y = history[0]
                        dy = cy - old_y
                        speed = math.hypot(cx - old_x, dy) / len(history)

                        # Check if moving opposite to expected lane direction
                        if self.expected_flow == "down" and dy < -18:
                            is_wrong_way = True
                        elif self.expected_flow == "up" and dy > 18:
                            is_wrong_way = True

                active_vehicles.append({
                    "id": track_id,
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "speed": speed,
                    "conf": conf
                })

                if is_wrong_way:
                    detections.append({
                        "hazard_type": "wrong_way",
                        "severity": "CRITICAL",
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        f"WRONG-WAY {label.upper()} #{track_id}",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                else:
                    # Subtle green box for normal traffic
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 1)

        # ---------------------------------------------------------
        # 3. ACCIDENT / COLLISION HEURISTIC
        # ---------------------------------------------------------
        for i in range(len(active_vehicles)):
            for j in range(i + 1, len(active_vehicles)):
                v1, v2 = active_vehicles[i], active_vehicles[j]
                iou = compute_iou(v1["bbox"], v2["bbox"])
                # High overlap + both vehicles nearly stationary = potential collision
                if iou > 0.35 and v1["speed"] < 1.2 and v2["speed"] < 1.2:
                    x1 = min(v1["bbox"][0], v2["bbox"][0])
                    y1 = min(v1["bbox"][1], v2["bbox"][1])
                    x2 = max(v1["bbox"][2], v2["bbox"][2])
                    y2 = max(v1["bbox"][3], v2["bbox"][3])

                    detections.append({
                        "hazard_type": "accident",
                        "severity": "CRITICAL",
                        "confidence": round(min(v1["conf"], v2["conf"]), 2),
                        "bbox": [x1, y1, x2, y2]
                    })
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        "ACCIDENT ALERT",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 0, 255),
                        2,
                    )

        # Encode frame to Base64 JPEG for FastAPI WebSocket streaming
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_frame = base64.b64encode(buffer).decode("utf-8")

        return annotated, detections, b64_frame