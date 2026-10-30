#!/bin/bash
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
while true; do
  echo; echo "===== PolySign AI ====="
  echo "1) Install (first time only)  2) Record a sign  3) Train model  4) Run app  5) Exit"
  read -p "Choose: " c
  case $c in
    1) pip install -r requirements.txt ;;
    2) read -p "Class ID (0-11): " id; python collect_landmarks.py --class_id $id --samples 50 ;;
    3) python train_model.py ;;
    4) streamlit run app.py ;;
    5) exit 0 ;;
  esac
done
