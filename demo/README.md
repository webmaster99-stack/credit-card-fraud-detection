---
title: Credit Card Fraud Detector
emoji: 🕵️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.38.2
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Score a simulated card transaction and see why
---

# Credit card fraud detector (demo)

Scores a single simulated card transaction or a CSV of them, explains each decision with SHAP, and
shows which model version answered. Built on the simulated Sparkov dataset; not for real decisions.

The model bundle is downloaded from the Hugging Face Hub model repo at startup. Source code,
training pipeline and documentation live in the project repository linked from the app's About tab.

This file is the Space's card. It is copied to the Space by `demo/build_space.py`; edit it here.
