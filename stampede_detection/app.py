import os
import uuid
import cv2
import math
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from scipy import stats
from flask import Flask, render_template, request
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Model loading (prefer best.pt if present) ---
models_dir = "models"
if not os.path.isdir(models_dir):
    print(f"FATAL ERROR: '{models_dir}' directory not found.")
    exit()

model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
if not model_files:
    print("FATAL ERROR: No .pt model found in 'models' directory.")
    exit()

preferred = None
for f in model_files:
    if f.lower() == "best.pt":
        preferred = f
        break
if preferred is None:
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
    preferred = model_files[0]

latest_model_path = os.path.join(models_dir, preferred)
print(f"[INFO] Loading model: {latest_model_path}")
try:
    model = YOLO(latest_model_path)
except Exception as e:
    print(f"FATAL ERROR: Could not load model. Ensure '{latest_model_path}' is valid. Error: {e}")
    exit()

# --- Parameters (tune these) ---
FRAME_SKIP = 5
PROCESSING_RESOLUTION = (640, 480)
CONFIDENCE_THRESHOLD = 0.3

ENTRY_ZONE_WIDTH_PCT = 0.18
EXIT_ZONE_WIDTH_PCT = 0.18
STAGNANT_DISPLACEMENT_THRESHOLD_PX = 15
STAGNANT_WINDOW_FRAMES = 25
NET_INFLOW_WINDOW_SECS = 3
NET_INFLOW_RATE_THRESH_HIGH = 6
NET_INFLOW_RATE_THRESH_MOD = 3
STAGNANT_RATIO_HIGH = 0.35

PPSM_SLOPE_THRESH = 0.1
ENTROPY_MULTIPLIER = 2.5

MOVEMENT_DIR_THRESHOLD_PX = 25  # threshold for movement-based classification
DRAW_YOLO_DETECTIONS = False    # set True to draw raw YOLO detection boxes for debugging

# --- Tracker (DeepSORT) ---
tracker = DeepSort(max_age=5, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)

# --- Helpers ---
def calculate_ppsm(boxes, frame_area):
    if not frame_area or not boxes:
        return 0
    total_person_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    return round((total_person_area / frame_area) * 100, 2)

def get_threat_level(ppsm):
    if ppsm > 60:
        return "🚨 Critical"
    elif ppsm > 40:
        return "⚠️ High"
    elif ppsm > 20:
        return "Moderate"
    else:
        return "Low"

def get_overall_threat_level(avg_risk_score):
    """
    Determines the overall threat level based on the average risk score.
    The risk score is a value from 0.0 to 1.0.
    """
    if avg_risk_score >= 0.7:
        return "Critical 🚨"
    elif avg_risk_score >= 0.5:
        return "High ⚠️"
    elif avg_risk_score >= 0.2:
        return "Moderate"
    else:
        return "Low"

def calculate_movement_entropy(prev_gray, current_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    if flow is None:
        return 0
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    angle_degrees = angle * 180 / np.pi / 2
    bins = np.arange(0, 361, 45)
    hist, _ = np.histogram(angle_degrees, bins=bins)
    if hist.sum() == 0:
        return 0
    prob_dist = hist / hist.sum()
    entropy = stats.entropy(prob_dist, base=2)
    return float(entropy) if not np.isnan(entropy) else 0.0

class TrackState:
    def __init__(self):
        self.centroid_history = deque(maxlen=STAGNANT_WINDOW_FRAMES)
        self.entered_entry = False
        self.exited_exit = False
        self.counted_inflow = False
        self.counted_outflow = False
        self.last_seen_frame = 0

def centroid_from_ltrb(l, t, r, b):
    cx = int((l + r) / 2)
    cy = int((t + b) / 2)
    return (cx, cy)

def track_to_ltrb_ints(track):
    try:
        l, t, r, b = track.to_ltrb()
    except Exception:
        try:
            l, t, r, b = track.to_tlbr()
        except Exception:
            l = getattr(track, "ltrb", None)
            if l and isinstance(l, (list, tuple)) and len(l) == 4:
                l, t, r, b = l
            else:
                raise
    return map(int, (l, t, r, b))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_source = None
        is_live_stream = False
        
        input_video_path = None # New variable to store the original input video path
        output_video_path = None
        input_name = None
        report_path = None

        stream_url = request.form.get("stream_url")
        video_file = request.files.get("video")

        if stream_url and stream_url.strip():
            video_source = stream_url.strip()
            is_live_stream = True
            input_name = "Live Stream"
        elif video_file and video_file.filename != '':
            filename_base = str(uuid.uuid4())
            filename = filename_base + ".mp4"
            input_video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename) # Store the original input video path
            video_file.save(input_video_path)
            video_source = input_video_path
            input_name = filename
            output_video_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
            report_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{filename_base}_report.csv")
        else:
            return render_template("index.html", error="No video file or stream URL provided.")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return render_template("index.html", error=f"Error opening video source: {video_source}. It might be a bad file or an offline stream.")

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        out = None
        if not is_live_stream and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

        entry_zone = (0, 0, 0, 0)
        exit_zone = (0, 0, 0, 0)
        
        report_data = []
        ppsm_history = deque(maxlen=int(fps / FRAME_SKIP * 3))
        entropy_history = deque(maxlen=int(fps / FRAME_SKIP * 3))
        ppsm_alert_count = 0
        entropy_alert_count = 0
        chart_labels, chart_ppsm_data, chart_entropy_data = [], [], []
        total_ppsm = 0
        total_risk_score = 0.0
        processed_frames_count = 0
        frame_count = 0
        last_known_ppsm = 0
        stampede_risk = "LOW"
        last_known_ppsm_alert = False
        last_known_entropy_alert = False
        last_known_tracks = []

        tracks_state = defaultdict(TrackState)
        track_first_pos = {}
        track_last_pos = {}
        inflow_history = deque(maxlen=max(1, int(fps / FRAME_SKIP * NET_INFLOW_WINDOW_SECS)))
        outflow_history = deque(maxlen=max(1, int(fps / FRAME_SKIP * NET_INFLOW_WINDOW_SECS)))

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            if out: out.release()
            return render_template("index.html", error="Could not read the first frame of the video.")

        prev_frame_resized = cv2.resize(prev_frame, PROCESSING_RESOLUTION)
        prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
        
        entry_zone = (0, 0, int(original_width * ENTRY_ZONE_WIDTH_PCT), original_height)
        exit_zone = (original_width - int(original_width * EXIT_ZONE_WIDTH_PCT), 0, original_width, original_height)

        print(f"[INFO] Starting video processing for source: {input_name}")
        print(f"[INFO] Entry zone (px): (0, 0, {int(original_width * ENTRY_ZONE_WIDTH_PCT)}, {original_height}) | Exit zone (px): ({original_width - int(original_width * EXIT_ZONE_WIDTH_PCT)}, 0, {original_width}, {original_height})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if is_live_stream and frame_count > 500:
                break

            if frame_count % FRAME_SKIP == 0:
                resized_frame = cv2.resize(frame, PROCESSING_RESOLUTION)
                results = model(resized_frame, classes=0, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

                detections = []
                boxes = []
                x_scale = original_width / PROCESSING_RESOLUTION[0]
                y_scale = original_height / PROCESSING_RESOLUTION[1]

                if results.boxes:
                    for r in results.boxes.data.tolist():
                        x1_res, y1_res, x2_res, y2_res, score, cls = r
                        x1 = int(x1_res * x_scale)
                        y1 = int(y1_res * y_scale)
                        x2 = int(x2_res * x_scale)
                        y2 = int(y2_res * y_scale)
                        boxes.append([x1, y1, x2, y2])
                        detections.append(([x1, y1, x2 - x1, y2 - y1], float(score), 'person'))

                updated_tracks = tracker.update_tracks(detections, frame=frame)
                last_known_tracks = updated_tracks

                if DRAW_YOLO_DETECTIONS:
                    for (x1, y1, x2, y2) in boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)

                inflow_this_frame_zone, outflow_this_frame_zone = 0, 0
                inflow_mov_this_frame, outflow_mov_this_frame = 0, 0
                stagnant_mov_this_frame, dir_counts = 0, {"down": 0, "up": 0, "left": 0, "right": 0}
                active_track_ids = set()

                for track in updated_tracks:
                    if not track.is_confirmed(): continue
                    track_id = track.track_id
                    try: l, t, r0, b = track_to_ltrb_ints(track)
                    except Exception: continue
                    cx, cy = centroid_from_ltrb(l, t, r0, b)
                    ts = tracks_state[track_id]
                    ts.centroid_history.append((cx, cy))
                    ts.last_seen_frame = frame_count
                    active_track_ids.add(track_id)
                    ex1, ey1, ex2, ey2 = entry_zone
                    ox1, oy1, ox2, oy2 = exit_zone
                    if ex1 <= cx <= ex2 and ey1 <= cy <= ey2: ts.entered_entry = True
                    if ox1 <= cx <= ox2 and oy1 <= cy <= oy2: ts.exited_exit = True
                    if ts.entered_entry and not ts.counted_inflow:
                        ts.counted_inflow = True
                        inflow_this_frame_zone += 1
                    if ts.exited_exit and not ts.counted_outflow:
                        ts.counted_outflow = True
                        outflow_this_frame_zone += 1
                    if track_id not in track_first_pos: track_first_pos[track_id] = (cx, cy)
                    track_last_pos[track_id] = (cx, cy)

                stale_ids = [tid for tid, s in tracks_state.items() if (frame_count - s.last_seen_frame) > (fps * 5)]
                for sid in stale_ids:
                    tracks_state.pop(sid, None)
                    track_first_pos.pop(sid, None)
                    track_last_pos.pop(sid, None)

                inflow_history.append(inflow_this_frame_zone)
                outflow_history.append(outflow_this_frame_zone)
                total_active = len(active_track_ids)
                stagnant_count = 0
                for tid in list(active_track_ids):
                    if tid not in track_first_pos or tid not in track_last_pos: continue
                    fx, fy = track_first_pos[tid]
                    lx, ly = track_last_pos[tid]
                    dx = lx - fx; dy = ly - fy
                    dist = math.hypot(dx, dy)
                    if dist < STAGNANT_DISPLACEMENT_THRESHOLD_PX:
                        stagnant_mov_this_frame += 1
                        stagnant_count += 1
                        continue
                    if abs(dy) >= abs(dx):
                        if dy > MOVEMENT_DIR_THRESHOLD_PX: inflow_mov_this_frame += 1; dir_counts["down"] += 1
                        elif dy < -MOVEMENT_DIR_THRESHOLD_PX: outflow_mov_this_frame += 1; dir_counts["up"] += 1
                    else:
                        if dx > MOVEMENT_DIR_THRESHOLD_PX: inflow_mov_this_frame += 1; dir_counts["right"] += 1
                        elif dx < -MOVEMENT_DIR_THRESHOLD_PX: outflow_mov_this_frame += 1; dir_counts["left"] += 1

                stagnant_ratio = (stagnant_count / total_active) if total_active > 0 else 0
                last_known_ppsm = calculate_ppsm(boxes, original_width * original_height)
                current_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                entropy = calculate_movement_entropy(prev_gray, current_gray)
                prev_gray = current_gray.copy()
                ppsm_history.append(last_known_ppsm); entropy_history.append(entropy)
                last_known_ppsm_alert = False
                if len(ppsm_history) == ppsm_history.maxlen and len(ppsm_history) > 1:
                    slope, _, _, _, _ = stats.linregress(np.arange(len(ppsm_history)), ppsm_history)
                    if slope > PPSM_SLOPE_THRESH: last_known_ppsm_alert = True; ppsm_alert_count += 1
                last_known_entropy_alert = False
                if len(entropy_history) == entropy_history.maxlen and len(entropy_history) > 1:
                    mean_entropy, std_entropy = np.mean(list(entropy_history)[:-1]), np.std(list(entropy_history)[:-1])
                    if std_entropy > 0 and entropy > mean_entropy + ENTROPY_MULTIPLIER * std_entropy:
                        last_known_entropy_alert = True; entropy_alert_count += 1
                net_inflow_zone = sum(inflow_history) - sum(outflow_history)
                stampede_score = 0.0
                if net_inflow_zone >= NET_INFLOW_RATE_THRESH_HIGH: stampede_score += 0.6
                elif net_inflow_zone >= NET_INFLOW_RATE_THRESH_MOD: stampede_score += 0.35
                if stagnant_ratio >= STAGNANT_RATIO_HIGH: stampede_score += 0.4
                elif stagnant_ratio > 0.15: stampede_score += 0.2
                if last_known_ppsm > 50: stampede_score += 0.3
                elif last_known_ppsm > 30: stampede_score += 0.15
                stampede_score = min(1.0, stampede_score)
                if stampede_score > 0.7: stampede_risk = "HIGH 🚨"
                elif stampede_score > 0.35: stampede_risk = "MODERATE ⚠️"
                else: stampede_risk = "LOW"
                chart_labels.append(frame_count)
                chart_ppsm_data.append(last_known_ppsm)
                chart_entropy_data.append(round(entropy, 4))
                report_data.append({
                    "frame": frame_count, "ppsm": last_known_ppsm, "entropy": round(entropy, 4),
                    "inflow_zone": int(inflow_this_frame_zone), "outflow_zone": int(outflow_this_frame_zone),
                    "inflow_mov": int(inflow_mov_this_frame), "outflow_mov": int(outflow_mov_this_frame),
                    "net_inflow_zone": int(net_inflow_zone), "stagnant_count": int(stagnant_count),
                    "stagnant_ratio": round(stagnant_ratio, 3), "dir_down": int(dir_counts["down"]),
                    "dir_up": int(dir_counts["up"]), "dir_left": int(dir_counts["left"]),
                    "dir_right": int(dir_counts["right"]), "stampede_score": round(stampede_score, 3),
                    "stampede_risk": stampede_risk, "ppsm_trend_alert": last_known_ppsm_alert,
                    "entropy_spike_alert": last_known_entropy_alert
                })
                total_ppsm += last_known_ppsm
                total_risk_score += stampede_score
                processed_frames_count += 1
                print(f"[FRAME {frame_count}] PPSM:{last_known_ppsm}% Ent:{round(entropy,4)} InZone:{inflow_this_frame_zone}/{outflow_this_frame_zone} InMov:{inflow_mov_this_frame}/{outflow_mov_this_frame} Stag:{stagnant_count}/{total_active} NetZone:{net_inflow_zone} Risk:{stampede_risk} Score:{round(stampede_score,3)}")

            # Create a copy of the original frame for drawing annotations
            annotated_frame = frame.copy()

            for track in last_known_tracks:
                try:
                    if not track.is_confirmed() or track.time_since_update > 1: continue
                    l, t, r0, b = track_to_ltrb_ints(track)
                    cv2.rectangle(annotated_frame, (l, t), (r0, b), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID:{track.track_id}", (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception: pass

            cv2.rectangle(annotated_frame, (entry_zone[0], entry_zone[1]), (entry_zone[2], entry_zone[3]), (0, 165, 255), 2)
            cv2.putText(annotated_frame, "Entry Zone", (entry_zone[0] + 5, entry_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.rectangle(annotated_frame, (exit_zone[0], exit_zone[1]), (exit_zone[2], exit_zone[3]), (0, 165, 255), 2)
            cv2.putText(annotated_frame, "Exit Zone", (exit_zone[0] + 5, exit_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(annotated_frame, f"PPSM: {last_known_ppsm}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Stampede: {stampede_risk}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            alert_y_pos = 120
            if last_known_ppsm_alert:
                cv2.putText(annotated_frame, "PPSM TREND ALERT!", (10, alert_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                alert_y_pos += 30
            if last_known_entropy_alert:
                cv2.putText(annotated_frame, "ENTROPY SPIKE ALERT!", (10, alert_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if out: out.write(annotated_frame)
        
        cap.release()
        if out: out.release()

        if processed_frames_count == 0:
            return render_template("index.html", error="The video is too short for analysis. Please upload a video with at least 5 frames or try a different live stream URL.")

        peak_stampede_score, peak_stampede_risk = 0.0, "LOW"
        total_inflow, total_outflow, total_inflow_mov, total_outflow_mov = 0, 0, 0, 0
        chart_stampede_data = []

        if report_data:
            df = pd.DataFrame(report_data)
            if report_path:
                df.to_csv(report_path, index=False)
            total_inflow = int(df.get("inflow_zone", pd.Series(dtype=int)).sum())
            total_outflow = int(df.get("outflow_zone", pd.Series(dtype=int)).sum())
            total_inflow_mov = int(df.get("inflow_mov", pd.Series(dtype=int)).sum())
            total_outflow_mov = int(df.get("outflow_mov", pd.Series(dtype=int)).sum())

            if "stampede_score" in df.columns and not df["stampede_score"].empty:
                peak_stampede_score = df["stampede_score"].max()
                peak_risk_idx = df["stampede_score"].idxmax()
                peak_stampede_risk = df.loc[peak_risk_idx, "stampede_risk"]
                chart_stampede_data = df["stampede_score"].tolist()

        avg_ppsm = total_ppsm / processed_frames_count
        avg_risk_score = total_risk_score / processed_frames_count
        overall_threat_level = get_overall_threat_level(avg_risk_score)

        print(f"[INFO] Done. AvgPPSM:{round(avg_ppsm,2)}% PeakRisk:{peak_stampede_risk} TotInZone:{total_inflow} TotOutZone:{total_outflow} TotInMov:{total_inflow_mov} TotOutMov:{total_outflow_mov}")
        
        output_video_display = output_video_path.replace('\\', '/') if output_video_path else None
        
        return render_template("index.html",
            input_video=input_video_path.replace('\\', '/') if input_video_path else None, # Pass the new input_video_path
            output_video=output_video_display,
            report_path=report_path.replace('\\', '/') if report_path else None,
            avg_ppsm=round(avg_ppsm, 2),
            ppsm_alerts=ppsm_alert_count,
            entropy_alerts=entropy_alert_count,
            avg_risk_score=round(avg_risk_score, 2),
            overall_threat_level=overall_threat_level,
            peak_stampede_risk=peak_stampede_risk,
            peak_stampede_score=round(peak_stampede_score, 2),
            total_inflow=int(total_inflow),
            total_outflow=int(total_outflow),
            total_inflow_mov=int(total_inflow_mov),
            total_outflow_mov=int(total_outflow_mov),
            uuid=uuid.uuid4(),
            chart_labels=chart_labels,
            chart_ppsm_data=chart_ppsm_data,
            chart_entropy_data=chart_entropy_data,
            chart_stampede_data=chart_stampede_data,
            is_live_stream=is_live_stream,
            stream_url=stream_url
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
