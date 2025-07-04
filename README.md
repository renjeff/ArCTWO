# A Framework for Emerging Nanoscale Memory Characterization using the ArC TWO platform

This repository contains the full implementation, documentation, and results of a semester project focused on developing a flexible framework for memristive device and crossbar array characterization using the **ArC TWO platform**.

## 🧠 Project Overview

As AI and neuromorphic computing demand new memory technologies, characterizing **memristive devices** efficiently and reliably becomes essential. This project implements a modular Python-based measurement framework using the [ArC TWO board](https://arc-instruments.com/) — a 64-channel FPGA platform tailored for analog and digital memory testing.

The system enables:
- IV sweeps
- Arbitrary pulse-based protocols (e.g., LTP/LTD, Retention, Endurance)
- Crossbar array readout with sneak-path mitigation
- Data logging in HDF5
- Jupyter notebook-based experimentation

## 📁 Repository Structure

```plaintext
├── src/                     # Core Python modules
│   ├── analysis/           # Plotting & data visualization
│   ├── instruments/        # Pulse sequence generation & device interface
│   ├── measurement/        # IV, pulsing, crossbar logic
│   ├── utils/              # Data I/O, hardware utilities
├── examples/               # Jupyter notebooks for interactive usage
│   └── main.ipynb          # Entry-point notebook for experiments
├── report/                 # Final report (PDF) and LaTeX source
├── presentation/           # Final presentation slides
├── crossbar_config/        # TOML files for pin mapping
├── docs/                   # Guides and documentation
│   └── WIP_ArC_TWO_Guide.pdf  # ETH Zurich user setup guide
└── README.md               # This file
```

## 🚀 Getting Started

### Requirements

- Python 3.8+
- `pyarc2` (Python bindings for ArC TWO)
- `numpy`, `matplotlib`, `h5py`, etc.
- Jupyter Lab or Notebook

> ℹ️ A conda environment file or `requirements.txt` may be added in future.

### Running an Experiment

Open the notebook:
```bash
cd examples
jupyter notebook main.ipynb
```

The notebook demonstrates:
- How to connect to the ArC TWO board
- IV sweep configuration and execution
- Arbitrary pulsing (LTP/LTD, retention, endurance)
- Crossbar array interaction with mock data

## 🧑‍💻 ETH Zurich Lab Setup

If you're working from ETH lab computers (e.g. `steghorn`), setup instructions, conda environment configuration, firmware flashing, and remote GUI usage are documented in the following guide:

📄 [WIP ArC TWO Guide (PDF)](./docs/WIP_ArC_TWO_Guide.pdf)

## 📝 Report
he final semester report and presentation slides are included in:
- [`/docs/Semester_Project_Report.pdf`](./docs/Semester_Project_Report.pdf)
- [`/docs/Semester_project_presentation_Jeff_Ren.pdf`](./docs/Semester_project_presentation_Jeff_Ren.pdf)

## 📚 References

Key technologies and concepts:
- Memristive devices (VCM, FTJ)
- In-memory computing and neuromorphic hardware
- ArC TWO platform by ArC Instruments
- Pulse-based characterization protocols

## 🤝 Acknowledgements

This project was conducted as part of the Spring 2025 Semester Project at ETH Zurich under the supervision of:

- Till Zellweger  
- Dr. Nikhil Garg  
- PD. Dr. Alexandros Emboras  
- Prof. Dr. Laura Bégon-Lours  
- Prof. Dr. Mathieu Luisier

---
