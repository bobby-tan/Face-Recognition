# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A two-script face recognition system using Python, OpenCV, and the `face_recognition` library (built on dlib). Based on the [Project Gurukul tutorial](https://projectgurukul.org/deep-learning-project-face-recognition-with-python-opencv/).

## Environment Setup

A `venv` directory is used to isolate dependencies. To set up:

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`dlib` (pulled in by `face_recognition`) requires CMake and a C++ compiler. On macOS: `brew install cmake`.

## Managing Dependencies

All third-party imports must be reflected in [requirements.txt](requirements.txt). When a third-party `import` statement is added or removed from any `.py` file, update `requirements.txt` accordingly. Standard library modules (`sys`, `pickle`, etc.) are not listed.

## Running

```bash
source venv/bin/activate

# Step 1: Generate and store face embeddings for a person
python embedding.py

# Step 2: Run live face recognition from webcam
python recognition.py
```

## Architecture

The project is split into two scripts that form a two-phase pipeline:

**`embedding.py`** — Enrollment phase
- Prompts for a person's name and ID
- Captures face images and extracts 128-dimensional face embeddings using `face_recognition.face_encodings()`
- Serializes embeddings to a `.pkl` pickle file for later use

**`recognition.py`** — Recognition phase
- Reads stored embeddings from the pickle file
- Captures live video frames via OpenCV
- For each frame, extracts face embeddings and compares them against stored embeddings using **L2 norm distance**
- The stored identity with the minimum L2 distance is chosen as the match

**Underlying model**: ResNet-34 variant trained on ~3 million faces, achieving 99.38% accuracy on the LFW benchmark. The model maps faces onto a 128-dimensional unit hypersphere where embeddings of the same person cluster together.
