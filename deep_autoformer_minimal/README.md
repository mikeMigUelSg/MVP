
# Deep Autoformer (Minimal, Educational)

This is a compact PyTorch implementation of a **Deep-Autoformer**-style model for time-series forecasting.

- Based on **Autoformer** (Wu et al., NeurIPS 2021) with series **decomposition** and an **Auto-Correlation** attention.
- Adds extra **MLP blocks** around encoder/decoder ("deep" enhancement) following the idea reported by Jiang et al. (2022) for VSTLF.
- Aimed at clarity and hackability; it is **not** an exact clone of the original optimized implementation.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# try the toy demo (creates a CSV and a config)
python examples/sine_demo.py

# train
python train.py --config configs/sine.yaml
```

## Data format

Provide a CSV with columns: `timestamp, target, [covariates...]`. Update `configs/config.yaml`.

## References

- Autoformer: *Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*, NeurIPS 2021.  
- Deep-Autoformer (VSTLF): Adds MLP layers to Autoformer; see Applied Energy (Dec 2022).

This repository is educational. For production or benchmarking on ETT/ECL/etc., see the official Autoformer codebase.
