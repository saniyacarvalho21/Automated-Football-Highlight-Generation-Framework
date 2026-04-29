# ⚽ Automated Football Highlight Generation Framework

> **An AI-powered multimodal pipeline for generating concise and meaningful football highlights using audio-visual intelligence.**

---

## 📌 Overview

Football matches typically last **90+ minutes**, but most viewers prefer **short, engaging highlights** that capture the essence of the game.

However, traditional highlight generation:

* Focuses only on **goals or fouls**
* Misses **critical moments** like saves, near misses, and build-up plays
* Requires **manual editing**, which is time-consuming and not scalable

👉 This project solves that by introducing an **automated, intelligent highlight generation system**.

---

## 🚀 Key Idea

> Instead of detecting only predefined events,
> this system treats highlight generation as a **continuous scoring problem over time**.

Each segment of the match is evaluated using:

* 🎥 Visual activity
* 🔊 Audio intensity
* 🧠 Contextual importance

---

## 🧠 Methodology

### 🔹 Multimodal Scoring Framework

Each video segment is assigned a **Semantic Importance Score (SIS):**

```
Final Score = 
0.45 × Visual Score + 
0.35 × Audio Score + 
0.20 × Context Score
```

### 🔹 Why Multimodal?

* **Visual** → Captures motion, player activity, key actions
* **Audio** → Captures crowd excitement, commentary spikes
* **Context** → Maintains stability across match phases

👉 Combining these leads to **more accurate and meaningful highlights**

---

## ⚙️ System Pipeline

```
Full Match Video
        ↓
Segmentation (Divide into clips)
        ↓
Feature Extraction
   ├── Visual (Motion, YOLO detection)
   ├── Audio (Energy, Pitch, Flux)
        ↓
Scoring Engine (Weighted SIS)
        ↓
Ranking & Selection
        ↓
Highlight Generation (MoviePy)
        ↓
Final Highlight Video 🎬
```

---

## 🧰 Tech Stack

* **Python** – Core development
* **OpenCV** – Video processing
* **MoviePy** – Video editing & concatenation
* **Librosa** – Audio feature extraction
* **YOLOv8** – Object detection (players, action zones)
* **NumPy / Pandas** – Data processing

---

## 📊 Results & Insights

### 🔹 Timeline Analysis

* Peaks in **audio + visual scores** correspond to high-impact moments
* System successfully detects:

  * Attacks
  * Saves
  * High-intensity plays

### 🔹 Component Analysis

* Visual and audio scores vary dynamically
* Context score remains stable → acts as a balancing factor

### 🔹 Activity Heatmap

* High activity concentrated near **goalpost regions**
* Helps identify attacking phases and transitions

---

## ✅ Achievements

* Automated highlight generation pipeline
* Reduced manual effort significantly
* Captured **diverse match moments**, not just goals
* Successfully combined **audio + visual intelligence**

---

## ⚠️ Limitations

* Limited dataset (tested on single full match)
* Weights are manually tuned (can be optimized further)
* Performance depends on input video quality

---

## 🔮 Future Scope

* ⚡ Real-time highlight generation during live matches
* 🎯 Player tracking and individual performance analysis
* 🏀 Extension to other sports (basketball, cricket, etc.)
* 📈 Model-based weight optimization with larger datasets

---

## 📁 Project Structure

```
├── data/
│   ├── segments/
│   ├── outputs/
│
├── src/
│   ├── segmenter.py
│   ├── video_features.py
│   ├── audio_features.py
│
├── run_pipeline.py
├── app.py (optional UI)
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# Clone repository
git clone https://github.com/your-username/Automated-Football-Highlight-Generation-Framework.git

# Navigate to project
cd Automated-Football-Highlight-Generation-Framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py
```

---

## 👩‍💻 Authors

* **Saniya Carvalho**
* **Abhishek Phatak**

---

## ⭐ Final Note

> This project demonstrates how combining **AI + multimedia understanding** can transform sports content consumption — making it faster, smarter, and more engaging.
