# 🚨 Stampede Detection & Crowd Analytics using YOLOv8 + DeepSORT

## 📌 Project Overview

This project is an **AI-powered crowd monitoring and stampede detection system** that uses:

* **YOLOv8** for real-time person detection
* **DeepSORT** for multi-object tracking
* **Flask** for interactive web deployment

It analyzes crowd behavior in videos or live streams and detects potential **stampede risks** using crowd density and motion dynamics.

Key metrics include:

* People Per Square Meter (PPSM)
* Movement Entropy
* Inflow / Outflow tracking
* Threat Level classification

---

## 🧠 Key Features

### 🔍 Detection & Tracking

* YOLOv8-based person detection in every frame
* DeepSORT assigns **unique IDs** for continuous tracking

### 📊 Crowd Analytics

* Real-time calculation of:

  * People Per Square Meter (PPSM)
  * Movement Entropy (crowd randomness)
  * Crowd inflow and outflow

### 🚨 Threat Detection System

Automatically classifies crowd risk into:

* Low
* Moderate
* High
* Critical 🚨

Triggers alerts when:

* PPSM spikes suddenly
* Movement entropy increases abnormally

### 📺 Video Processing Support

* Upload video files
* Process live IP camera feeds
* YouTube stream support (optional integration)

### 📈 Visualization & Reports

* Real-time graphs for PPSM and entropy
* CSV report generation after processing
* Annotated output videos with tracking overlays

---

## 🗂️ Project Structure

```
stampede-detection/
│
├── app.py                 # Flask backend (main server)
├── templates/
│   └── index.html         # Web UI
├── static/
│   ├── uploads/           # Input videos
│   ├── outputs/           # Processed videos + reports
│   ├── css/
│   └── js/
│
├── models/
│   ├── yolov8n.pt         # Pretrained YOLOv8 model
│   └── best.pt            # Custom trained model (optional)
│
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/stampede-detection.git
cd stampede-detection
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Download YOLOv8 Model

Place model inside `/models`:

* `best.pt` (custom trained)
* OR `yolov8n.pt` (pretrained)

---

## ▶️ Running the Project

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

You can:

* Upload video for analysis
* Provide live stream URL

---

## 📊 Output Details

### 📁 Generated Files

* Processed video → `static/outputs/`
* CSV report → saved alongside output

### 📈 Metrics Displayed

* Average PPSM (Crowd density)
* Movement Entropy
* Total Inflow / Outflow
* Threat Level classification
* Alert logs

---

## 🚨 Threat Level Logic

| Metric | Range | Level       |
| ------ | ----- | ----------- |
| PPSM   | 0–20  | Low         |
| PPSM   | 20–40 | Moderate    |
| PPSM   | 40–60 | High ⚠️     |
| PPSM   | >60   | Critical 🚨 |

**Entropy Alerts:**

* Sudden increase → Possible unstable crowd movement

---

## 🧩 Model Configuration

### Default Models

* Lightweight: `yolov8n.pt`
* Custom trained: `best.pt`

### Selection Priority

```
best.pt → preferred
else → yolov8n.pt
```

---

## 🛠️ Technologies Used

* Python 3.10+
* Flask
* YOLOv8 (Ultralytics)
* DeepSORT
* OpenCV
* NumPy
* Pandas
* SciPy

---

## 📈 Example Output

* Average PPSM: 45.3
* Peak Threat Level: 🚨 Critical
* Total Inflow: 112
* Total Outflow: 97
* Alerts Triggered: 3 PPSM alerts, 1 entropy alert

---

## 🔮 Future Improvements

* 🔴 Real-time CCTV deployment support
* 🧠 LSTM-based crowd prediction
* 🌐 Cloud deployment (AWS / Azure)
* 📡 Multi-camera fusion tracking
* 📱 Mobile alert system

---

## 🤝 Contributing

Contributions are welcome:

* Fork the repository
* Improve detection accuracy
* Optimize real-time performance
* Submit a pull request

---

⭐ If you like this project, consider starring the repo and improving real-time scalability.
