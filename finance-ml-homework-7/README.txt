Assignment 7 - Neural network option pricing, Greeks and no-arbitrage

The PDF report contains answers to tasks 1.1-1.5 and 2.1-2.6, plus Bonus A.
All tables and figures come from actual CPU training and evaluation runs.

FILES
  assignment7_report.pdf       English report and answers
  assignment7.py              Complete experiment code (PyTorch)
  requirements.txt            Tested package versions (Python 3.12)
  metadata.json               Seed, sample counts, versions and precision
  data_split.npz              Retained samples and exact train/test indices
  bonus_points.npz            Out-of-domain evaluation samples
  models/                     3 base and 5 fine-tuned checkpoints
  figures/                    Original-resolution figures
  results/                    CSV tables, JSON results, learning histories

INSTALL AND RUN (from this folder)
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python assignment7.py --threads 2

Evaluate the supplied models without training again:
  python assignment7.py --stage evaluate --threads 2

Run analytic/implementation checks only:
  python assignment7.py --stage check --threads 2

The default full run overwrites result files and model checkpoints in this folder.
Use --output-dir another_folder to preserve the supplied run.

REPRODUCIBILITY
  Fixed seed: 20261028. CPU; float32 training, float64 evaluation.
  Three hidden layers of 128 units, linear output, scaling inside the model.
  Same initial Linear weights and epoch-wise sample order across activations.
  Base: 40 epochs, batch 128, Adam lr 0.002, cosine decay at each batch.
  Fine-tuning: 10 epochs, batch 256, Adam lr 0.0005 held constant.
  Each fine-tune run resets optimizer and starts from the same MELU checkpoint.
  Adam uses default betas=(0.9,0.999), eps=1e-8, no weight decay.
  All reported models are the final epoch, with no test-based model selection.
  Changing package versions, hardware or thread count may change last digits.

SIGN TESTS
  Main report tables use strict signs, as specified in the handout.
  CSV/JSON additionally provide -1e-12-tolerance rates to separate roundoff.
  The discrete butterfly holds S fixed and rescales m and C when K changes.
  Its residual is divided by positive K2; this preserves the requested sign.
  No batch normalization, derivative supervision or extra training data is used.

SCOPE NOTE
  The handout introduction mentions inverse-map calibration, but gives no
  calibration task, inputs, loss or evaluation protocol. The report covers every
  numbered task actually supplied, and documents this discrepancy explicitly.

SOURCE
  Andrey Itkin, Assignment 7, Machine Learning in Financial Engineering,
  FA26-FRE-GY-7773, supplied as hw_dl_calibration.pdf.
  PyTorch API: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
