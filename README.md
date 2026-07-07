# AirWrite – Touchless Handwriting Recognition

AirWrite is a real-time computer vision application that enables users to write characters in the air using their index finger. The system tracks fingertip movements using a webcam, renders them on a virtual canvas, and recognizes handwritten characters using a CNN trained on the EMNIST Letters dataset.



### Air Drawing
<img width="578" height="326" alt="image" src="https://github.com/user-attachments/assets/9c474e92-57d0-4f71-838b-b0ee1ab6eaae" />


<img width="578" height="313" alt="image" src="https://github.com/user-attachments/assets/000762ad-8976-4f72-99ca-677e4049d813" />


<img width="320" height="58" alt="image" src="https://github.com/user-attachments/assets/0677c315-2765-41e2-be51-f89f45c993c4" />


## Features

- Real-time fingertip tracking
- Gesture-based virtual drawing
- Virtual drawing canvas
- Character segmentation and preprocessing
- CNN-based handwritten character recognition
- Live prediction with confidence score



## Tech Stack

- Python
- OpenCV
- MediaPipe
- PyTorch
- NumPy



## Project Structure

```text
AirWrite/
│
├── app.py
├── tracker.py
├── canvas.py
├── gesture_detector.py
├── segmenter.py
├── recognizer.py
├── model.py
├── train_model.py
├── models/
│   └── character_cnn.pth
├── images/
│   ├── air_drawing.png
│   ├── recognition.png
│   └── cnn_input.png
└── README.md
```


## Model

The recognition model is a Convolutional Neural Network trained on the **EMNIST Letters** dataset.

| Metric | Value |
|---------|------:|
| Dataset | EMNIST Letters |
| Framework | PyTorch |
| Test Accuracy | **93.25%** |



## Workflow

```
Webcam
      │
      ▼
MediaPipe Hand Tracking
      │
      ▼
Virtual Canvas
      │
      ▼
Character Segmentation
      │
      ▼
Image Preprocessing
      │
      ▼
CNN Character Recognition
      │
      ▼
Predicted Character
```



## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AirWrite.git
cd AirWrite
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```



## Run

```bash
python app.py
```



## Controls

| Action | Input |
|--------|-------|
| Draw | Index Finger |
| Stop Drawing | Two Fingers |
| Clear Canvas | Closed Fist |
| Recognize Character | S |
| Exit | ESC |


## Future Improvements

- Word-level handwriting recognition
- Sentence generation
- Spell correction
- Support for digits and mathematical symbols
- Gesture-based text editing



## Author

**Ireshi Rawat**



