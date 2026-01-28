# Source-Free Open-World RF Fingerprint Identification

This repository contains an **anonymous** PyTorch implementation of the paper: **"Source-Free Open-World RF Fingerprint Identification"**.

## Overview

We propose a framework to address the **Source-Free Open-World (SF-OW)** problem in RFFI. The method consists of two core components:
1.  **Incremental Orthogonal ETF (IO-ETF):** An active geometric prior for structural interference isolation and confusion suppression.
2.  **Triple-Level Geometric Alignment (TLGA):** A strategy aligning unlabeled streams via semantic OT, manifold anchoring, and subspace retention.



## 📂 Project Structure

The project is organized as follows:

```text
.
├── data/                   # Datasets (see "Data Preparation")
├── models/                 # Model definitions
│   ├── build_model.py      # Model builder
│   ├── ETF_classifier.py   # IO-ETF head implementation
│   └── resnet.py           # Backbone 
├── utils/                  # Utilities
│   ├── evaluate_utils.py   # Hungarian matching
│   ├── losses.py           # Losses for TLGA
│   ├── sinkhorn_knopp.py   # Sinkhorn-Knopp (balanced OT)
│   └── utils.py            # General helpers
├── SFOW_RFFI_stage_1.py    # Stage 1: Source pre-training
├── SFOW_RFFI_stage_2.py    # Stage 2: SF-OW adaptation (main method)
└── README.md        
```

## ⚙️ Requirements

The code is implemented in Python 3.8 and PyTorch.

1. **Create a virtual environment (recommended):**
   ```bash
   conda create -n sfow_rffi python=3.8
   conda activate sfow_rffi
   ```
   
2. **Install PyTorch:**
   Please install the version compatible with your CUDA version from the [official website](https://pytorch.org/).

3. **Install Dependencies:**
   ```bash
   pip install numpy scipy scikit-learn tqdm matplotlib tensorboardX
   ```

## ⬇️ Data Preparation


Please download the public datasets from their official sources and place them under `./data/` directory.

* **WiSig:** https://cores.ee.ucla.edu/downloads/datasets/wisig/
    * Contains WiFi signals from different Atheros chipsets captured by USRPs on the ORBIT testbed.
* **Oracle:** https://genesys-lab.org/oracle
    * Bit-similar device identification with USRP X310 transmitters.
* **LoRa:** https://ieee-dataport.org/open-access/lorarffidataset
    * Large-scale IoT identification with commercial LoRa devices.

**Directory Structure:**

```text
./data/
├── wisig/
│   ├── [dataset files...]
├── oracle/
│   ├── [dataset files...]
└── lora/
    └── [dataset files...]
```
*(Note: Ensure the dataset paths in the arguments match your local structure if different from default.)*

## 🚀 Usage

The framework consists of two stages. Below are the commands for all three benchmarks.

### Stage 1: Source Pre-training
Train the model on old classes (Source) with labels.

```bash
# Oracle
python SFOW_RFFI_stage_1.py --dataset oracle --epochs 50 --tag quick_debug_oracle --no-class 10 --no-known 10

# WiSig
python SFOW_RFFI_stage_1.py --dataset wisig --epochs 50 --tag quick_debug_wisig --no-class 4 --no-known 4

# LoRa
python SFOW_RFFI_stage_1.py --dataset lora --epochs 10 --tag quick_debug_lora --no-class 20 --no-known 20
```

### Stage 2: SF-OW Adaptation 
Adapt to the mixed unlabeled stream (Old + New) without source data.

```bash
# Oracle
python SFOW_RFFI_stage_2.py --dataset oracle --rff-method spectrogram --no-progress --lbl-percent 10 --novel-percent 34 --epochs 100

# WiSig
python SFOW_RFFI_stage_2.py --dataset wisig --rff-method spectrogram --no-progress --lbl-percent 10 --novel-percent 40 --epochs 100

# LoRa
python SFOW_RFFI_stage_2.py --dataset lora --rff-method spectrogram --no-progress --lbl-percent 10 --novel-percent 20 --epochs 100
```

## 📊 Outputs

- **Logs:** Training logs, accuracy, H-score, and AUROC are saved in the `outputs/` directory.
- **Checkpoints:** Model weights are saved automatically based on best performance.

## ⚠️ Anonymity Note
This code is submitted for double-blind review. All author details have been removed.









