<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=900&color=00F7FF&center=true&vCenter=true&width=750&lines=Gesture-Controlled+Music+Player;Python+%7C+OpenCV+%7C+MediaPipe+%7C+Pygame;Doctor+Strange+Inspired+Magic+VFX;Touchless+%7C+Cinematic+%7C+Real-Time" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://forthebadge.com/images/badges/made-with-python.svg" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=yellow" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Pygame-Audio%20Engine-red?style=for-the-badge" />
</p>

<hr/>

# 🎶 Gesture‑Controlled Music Player (Doctor Strange Edition)

A **real‑time, touch‑free music player** that lets you control playback using **hand gestures detected via your webcam**. Built with **Python + Computer Vision**, and enhanced with **Doctor‑Strange‑style magical visual effects** for a cinematic experience.

> ✋ No mouse. ⌨️ No keyboard. 🎧 Just gestures.

---

## 🚀 Project Highlights

* 🎥 **Real‑time hand tracking** using MediaPipe
* 🖐️ **Gesture‑based controls** (Play / Pause / Stop / Next / Previous)
* 🔊 **Pinch‑based volume control** with smoothing
* 🎶 **MP3 playlist support**
* 🌀 **Doctor Strange inspired magic circle VFX**
* ⚡ Optimized for **macOS (CoreAudio + OpenCV)**
* 🧠 Modular, clean, exam‑ready architecture

---

## 🧠 Gesture Controls

| Gesture                | Action         |
| ---------------------- | -------------- |
| ✋ Single Open Palm     | Play / Resume  |
| ✊ Single Fist          | Pause          |
| ✋✋ Two Open Palms      | Stop           |
| ✊✊ Two Fists           | Pause          |
| 👉 Swipe Right         | Next Track     |
| 👈 Swipe Left          | Previous Track |
| 🤏 Thumb + Index Pinch | Volume Control |

> Gesture cooldowns are applied to avoid accidental triggers.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,opencv&theme=dark" />
</p>

* **Python 3.11+**
* **OpenCV** – Camera capture & rendering
* **MediaPipe** – Hand landmark detection
* **Pygame** – Audio playback engine
* **NumPy** – Math & smoothing filters

---

## 📁 Project Structure

```
HandTrakerMP3/
│
├── main.py              # Main application
├── songs/               # Place your MP3 files here
│   ├── song1.mp3
│   ├── song2.mp3
│
├── venv/                # Virtual environment (not pushed)
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/HandTrakerMP3.git
cd HandTrakerMP3
```

### 2️⃣ Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Songs

* Add **non‑DRM `.mp3` files** to the `songs/` folder
* Apple Music / DRM files ❌ not supported

---

## ▶️ Run the Application

```bash
python main.py
```

🟢 Press **`q`** to exit

---

## 🎨 Visual Effects

* Rotating **magic mandala circles** attached to the hand center
* Pulsing animation synced with playback state
* Glowing visuals when music is playing
* Dimmed effects when paused

Inspired by **Doctor Strange spell circles** for a cinematic feel ✨

---

## 🧩 Architecture Overview

* **MusicPlayer** → Audio loading, playback & volume
* **GestureController** → Hand state & gesture interpretation
* **VisualEffects** → Magic circle animations
* **App** → Camera loop, UI overlay & orchestration

Designed with **clear separation of concerns** for readability and scalability.

---

## ⚠️ Notes & Limitations

* Webcam access required
* Best results under good lighting
* Optimized mainly for **macOS**
* Single‑threaded loop (can be improved)

---

## 🌱 Future Enhancements

* 🎚️ Gesture calibration mode
* 🖥️ PyQt / GUI interface
* 🔥 Particle‑based spell effects
* 🎛️ Custom gesture mapping
* 🚀 FPS & performance optimization

---

## 🧑‍💻 Author

**Sarthak Bhopale**
Engineering Student | Python Developer | Computer Vision Enthusiast

---

## ⭐ Support the Project

If you like this project, consider giving it a ⭐ on GitHub — it really helps!

> *“The future of interaction is touchless.”* ✨
