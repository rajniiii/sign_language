# Real-Time Phrase-Level Sign Language Recognition

An on-device system that recognizes dynamic sign language gestures from a live camera feed and translates them into text and speech in real time. Instead of classifying static hand poses frame by frame, the system models how each gesture unfolds over time, enabling more natural, phrase-level recognition.

## Overview

Most sign language recognition systems classify isolated signs from single video frames, which works poorly for gestures that are defined by motion rather than a static hand shape. This project takes a sequence-based approach: each frame is converted into a compact pose-and-hand keypoint vector, and a Bidirectional LSTM model learns the temporal pattern of the full gesture.

## How It Was Built

- **Custom dataset:** Since no existing dataset matched the gesture set we needed, we recorded a custom dataset of 1,500 video clips, with each team member contributing recordings under varied viewpoints, hand dominance, speeds, and distances to help the model generalize.
- **Keypoint extraction:** Each frame is processed with MediaPipe Holistic to extract 33 pose landmarks and 21 landmarks per hand, concatenated into a 258-dimensional keypoint vector per frame.
- **Sequence modeling:** Sequences of roughly 30-40 frames are normalized, padded, and fed into a Bidirectional LSTM with TimeDistributed dense layers, dropout, and class weighting to handle class imbalance.
- **Baselines tried first:** Static CNN classifiers and frame-by-frame models were tested first but failed to capture the motion patterns that distinguish similar-looking gestures, which motivated the move to a sequence model.
- **Real-time inference pipeline:** A rolling sequence window, frame smoothing, confidence thresholding, and a label-stability rule work together to filter out noisy or incidental hand movements and produce stable predictions during live use.

## Results

- **Test accuracy:** 85.57%
- **Validation accuracy:** 81% (n = 305, stratified 70/15/15 train/val/test split)
- **Real-time speed:** 20-30 FPS on consumer hardware

## Tech Stack

Python, TensorFlow/Keras, MediaPipe Holistic, OpenCV, NumPy, scikit-learn

## Documentation

- 📄 [Research Paper (PDF)](docs/Research_Paper.pdf)
- 📊 [Project Presentation (PPTX)](docs/Project_Presentation.pptx)

> Place the report and presentation files inside a `docs/` folder at the root of this repo so the links above resolve correctly. Visitors can click either link to view or download the file directly from GitHub.

## Future Scope

- Expanding the gesture vocabulary toward sentence-level recognition of multi-sign phrases
- Increasing dataset diversity with more participants, environments, and motion variations
- Exploring transformer-based sequence models for longer or more complex gestures
- Building a mobile version of the pipeline for on-device inference on phones and wearables
- Deploying on smart glasses for hands-free, real-time gesture-to-text and gesture-to-speech translation, letting sign language users communicate naturally with people who don't know sign language
- Supporting multilingual speech output
