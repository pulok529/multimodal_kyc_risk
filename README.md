# Multimodal KYC Risk (Image + Tabular) - Starter Project

Toy multimodal pipeline:
- creates synthetic 'ID document' images (simple patterns) + tabular attributes
- trains a small PyTorch model that combines CNN image features + MLP tabular features
- reports accuracy/AUC

This is a *template* to swap in real KYC document images and tabular metadata.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Notes
- Uses generated data, so it runs anywhere.
- Replace `data_gen.py` with your real preprocessing later.
