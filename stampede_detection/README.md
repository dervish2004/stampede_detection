🚨 Stampede Detection and Crowd Analytics using YOLOv8 + DeepSORT
This project is an advanced crowd monitoring and stampede detection system that uses YOLOv8 for object detection, DeepSORT for multi-object tracking, and Flask for an interactive web interface.
It detects people in real-time or from uploaded videos, tracks their movement, estimates crowd density, and evaluates threat levels based on metrics like People Per Square Meter (PPSM) and Movement Entropy.
🧠 Features
✅ YOLOv8-based Detection – Detects people in each frame with high accuracy.
✅ DeepSORT Tracking – Assigns unique IDs to each detected person for consistent tracking.
✅ Crowd Analytics Dashboard – Displays real-time statistics such as:
People Per Square Meter (PPSM)
Movement Entropy
Threat Level (Low / Moderate / High / Critical)
✅ Threat Alerts – Automatically flags sudden spikes in entropy or PPSM.
✅ Video Upload & Live Stream Support – Works with both local files and IP camera/YouTube streams.
✅ Automatic Report Generation (CSV) – Exports analysis data after processing.
✅ Dynamic Visualization – Graphs for PPSM, entropy, and stampede risk trends.
🗂️ Project Structure
stampede-detection/
│
├── app.py                         # Main Flask backend
├── templates/
│   └── index.html                 # Frontend UI for upload and visualization
├── static/
│   ├── uploads/                   # Uploaded videos
│   ├── outputs/                   # Processed output videos + reports
│   └── css/, js/                  # (Optional) Assets for UI
├── models/
│   └── yolov8n.pt                 # YOLOv8 model (replaceable with best.pt)
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/stampede-detection.git
cd stampede-detection
2️⃣ Create a virtual environment
python -m venv venv
3️⃣ Activate the environment
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
4️⃣ Install dependencies
pip install -r requirements.txt
5️⃣ Download a YOLOv8 model
Place your model file inside the models/ folder.
You can use:
Your trained model → best.pt
Or pretrained YOLOv8 model → yolov8n.pt
If both are present, the app will prefer best.pt automatically.
▶️ Running the App
python app.py
Then open your browser and go to:
http://127.0.0.1:5000
You can now:
Upload a video for analysis, or
Enter a stream URL (e.g., IP camera or YouTube stream)
📊 Output
Processed Video: Saved in static/outputs/
Report (CSV): Saved alongside the output video
Metrics Displayed:
Average PPSM (Crowd Density)
Movement Entropy
Threat Level (Low / Moderate / High / Critical)
Total Inflow / Outflow
Real-time Trend Graphs
🧩 Model Configuration
To switch models:
# Option 1: Use best.pt
models/best.pt

# Option 2: Use YOLOv8n (faster)
models/yolov8n.pt
The app automatically detects whichever model is available.
💡 Threat Level Criteria
Metric	Range	Level
PPSM	0–20	Low
PPSM	20–40	Moderate
PPSM	40–60	High ⚠️
PPSM	>60	Critical 🚨
Entropy Spikes	Sudden increases	Alert for unusual movement
🛠️ Technologies Used
Python 3.10+
Flask
YOLOv8 (Ultralytics)
DeepSORT
OpenCV
NumPy, SciPy, Pandas
📈 Example Results
Metric	Value
Average PPSM	45.3%
Peak Threat	🚨 Critical
Total Inflow	112
Total Outflow	97
Alerts	3 PPSM, 1 Entropy
🤝 Contributing
Feel free to fork, improve, and submit a pull request!
Suggestions for improving accuracy or real-time performance are welcome.
