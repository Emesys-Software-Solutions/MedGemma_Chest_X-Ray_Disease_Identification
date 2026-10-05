# 🩻 MedGemma Chest X-Ray Disease Identification

> **A multimodal Vision-Language AI system for chest X-ray disease identification, abnormality localization, explainable analysis, and structured clinical report generation.**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c?logo=pytorch)
![MedGemma](https://img.shields.io/badge/Model-MedGemma-green)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)
![PEFT](https://img.shields.io/badge/Fine--Tuning-PEFT%20%7C%20LoRA-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [Target Diseases and Findings](#-target-diseases-and-findings)
- [System Workflow](#-system-workflow)
- [Image Processing Pipeline](#-image-processing-pipeline)
- [MedGemma Architecture](#-medgemma-architecture)
- [Fine-Tuning Strategy](#-fine-tuning-strategy)
- [Visual Explainability](#-visual-explainability)
- [Clinical Evaluation](#-clinical-evaluation)
- [Project Phases](#-project-phases)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Deployment Architecture](#-deployment-architecture)
- [Expected Deliverables](#-expected-deliverables)
- [Screenshots](#-screenshots)
- [Demo](#-demo)
- [Future Enhancements](#-future-enhancements)
- [Medical Safety Notice](#-medical-safety-notice)
- [License](#-license)

---

## 🩻 Project Overview

Chest radiography is one of the most widely used diagnostic imaging modalities. However, interpretation can involve substantial workload and inter-observer variability.

**MedGemma Chest X-Ray Disease Identification** is a multimodal Vision-Language AI project designed to analyze chest X-ray images, identify potential diseases and abnormalities, provide visual localization, and generate structured clinical narratives.

The proposed system combines a medical vision encoder with a language-model decoder to process radiographs and produce structured outputs.

The project focuses on:

- 🩻 Chest X-ray disease identification
- 🔍 Abnormality localization
- 📝 Automated structured report generation
- 🧠 Multimodal vision-language reasoning
- 📊 Multi-label pathology classification
- 🎯 Confidence-aware clinical findings
- 🔬 Visual explainability using heatmaps and attention maps
- ⚡ Parameter-efficient model adaptation using LoRA
- 🖥️ Interactive Streamlit demonstration

The supplied project document defines the goal as an advanced multimodal Vision-Language AI system leveraging the MedGemma architecture for automated chest X-ray disease identification, abnormality localization, and structured clinical report generation. fileciteturn3file0L29-L38

---

## ⚠️ Problem Statement

Chest X-ray interpretation requires clinical expertise and can become a significant workload bottleneck, particularly when large numbers of radiographs need to be reviewed.

The project addresses this challenge by developing an AI-assisted pipeline capable of:

1. Processing DICOM/PNG chest radiographs.
2. Normalizing images for model input.
3. Identifying multiple possible pathologies.
4. Localizing visually relevant abnormalities.
5. Generating structured clinical findings.
6. Providing confidence information.
7. Producing explainable visual outputs for clinician verification.

The system is designed as a **clinical decision-support research prototype**, not as a replacement for qualified medical professionals.

---

## 💡 Our Solution

The proposed system connects medical image processing, multimodal AI, fine-tuning, explainability, and automated reporting into a single pipeline.

```text
┌──────────────────────────────┐
│     Chest X-Ray Input        │
│      DICOM / PNG             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Image Processing             │
│ Windowing / Resize / Normalize│
│ CLAHE Enhancement            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Medical Vision Encoder       │
│ ViT / Dense CNN Encoder      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ MedGemma Vision-Language AI  │
│ Multimodal Representation    │
└──────────────┬───────────────┘
               │
        ┌──────┴───────┐
        ▼              ▼
┌──────────────┐ ┌────────────────┐
│ Disease /    │ │ Clinical Text  │
│ Abnormality  │ │ Report         │
│ Prediction   │ │ Generation     │
└──────┬───────┘ └───────┬────────┘
       │                  │
       └────────┬─────────┘
                ▼
┌──────────────────────────────┐
│ Explainability               │
│ Grad-CAM / Attention Maps    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Streamlit Clinical Demo     │
│ Findings + Confidence + Map │
└──────────────────────────────┘
```

---

## ✨ Key Features

### 🩻 1. Chest X-Ray Disease Identification

The system is designed for multi-label pathology classification from chest radiographs.

The supplied roadmap identifies examples including:

- Pneumonia
- Cardiomegaly
- Pleural effusion
- Pneumothorax

The final supported pathology set depends on the selected training dataset and project implementation. fileciteturn3file0L31-L38

---

### 🧠 2. Multimodal Vision-Language Analysis

The architecture couples an image encoder with a decoder-based language model.

```text
Chest X-Ray
     │
     ▼
Vision Encoder
     │
     ▼
High-Dimensional Visual Features
     │
     ▼
Multimodal Representation
     │
     ▼
Language Model Decoder
     │
     ▼
Clinical Findings / Narrative
```

The project specifies either a **Vision Transformer (ViT)** or dense convolutional image encoder combined with a decoder-based language model. fileciteturn3file0L39-L40

---

### 📝 3. Automated Clinical Reporting

The system is designed to generate structured radiology-style text reports from chest X-ray images.

Potential output structure:

```text
Clinical Findings
-----------------
Finding 1: ...
Finding 2: ...
Finding 3: ...

Confidence
----------
Finding 1: ...
Finding 2: ...

Generated Impression
--------------------
...
```

The supplied project scope specifically includes automated structured radiology text reports containing clinical findings and confidence scores. fileciteturn3file0L34-L38

---

### 🔍 4. Abnormality Localization

The system includes visual explainability mechanisms intended to help verify where the model is focusing on the radiograph.

Planned approaches include:

- Grad-CAM
- Attention-map extraction
- Heatmap visualization
- Bounding visual localization

These are intended to support clinician verification rather than independently establish a diagnosis. fileciteturn3file0L34-L38

---

## 🦠 Target Diseases and Findings

The supplied project document explicitly provides the following example pathologies:

| Pathology | Example Use |
|---|---|
| Pneumonia | Pulmonary abnormality identification |
| Cardiomegaly | Cardiac enlargement identification |
| Pleural Effusion | Pleural fluid abnormality |
| Pneumothorax | Abnormal pleural air detection |

The exact final disease-label taxonomy is expected to depend on the datasets and annotation strategy selected for training. fileciteturn3file0L31-L36

---

## 🔄 System Workflow

```text
                 ┌─────────────────────┐
                 │  DICOM / PNG Image  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ DICOM Parsing       │
                 │ Image Conversion    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Windowing & Contrast│
                 │ CLAHE Enhancement   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Resize & Normalize  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Vision Encoder      │
                 │ ViT / CNN           │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ MedGemma Model      │
                 │ Multimodal Analysis │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      ┌────────────────┐         ┌────────────────┐
      │ Pathology      │         │ Report         │
      │ Classification │         │ Generation     │
      └───────┬────────┘         └───────┬────────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
                  ┌──────────────────┐
                  │ Explainability   │
                  │ Heatmaps / Maps  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Streamlit Demo   │
                  └──────────────────┘
```

---

## 🖼️ Image Processing Pipeline

Medical radiographs require consistent preprocessing before model inference.

### Processing Steps

```text
DICOM / PNG
    │
    ▼
DICOM Parsing
    │
    ▼
Windowing
    │
    ▼
Lung / Mediastinum Contrast Optimization
    │
    ▼
Resize
    │
    ▼
Normalization
    │
    ▼
CLAHE Enhancement
    │
    ▼
Model Input
```

### Supported Processing Concepts

| Processing Stage | Purpose |
|---|---|
| DICOM Conversion | Convert medical imaging data into model-ready images |
| Windowing | Optimize lung/mediastinum contrast |
| Resizing | Standardize model input dimensions |
| Normalization | Match training/input statistics |
| CLAHE | Enhance subtle pulmonary opacities |

The supplied roadmap specifies example input dimensions of **512×512 or 1024×1024**, along with ImageNet/medical-dataset normalization and CLAHE enhancement. fileciteturn3file0L41-L42

---

## 🤖 MedGemma Architecture

The proposed architecture consists of two major components:

### Vision Encoder

The vision encoder extracts high-dimensional spatial representations from chest radiographs.

```text
Chest X-Ray
     │
     ▼
Image Encoder
     │
     ▼
Visual Embeddings
```

The roadmap allows for a **Vision Transformer (ViT)** or dense convolutional image encoder. fileciteturn3file0L39-L44

### Language Model Decoder

The language model uses the visual representation to generate clinical findings and structured narrative output.

```text
Visual Embeddings
       │
       ▼
Multimodal Decoder
       │
       ▼
Clinical Findings
       │
       ▼
Structured Report
```

---

## 🧩 Fine-Tuning Strategy

The project uses **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA (Low-Rank Adaptation)** to adapt language-generation weights for clinical findings.

### Training Concept

```text
Pretrained MedGemma
        │
        ▼
Chest X-Ray Dataset
        │
        ▼
Medical Vision-Language Alignment
        │
        ▼
LoRA / PEFT Adaptation
        │
        ▼
Multi-Label Classification
        +
Autoregressive Report Generation
        │
        ▼
Fine-Tuned Model
```

The specified loss objective combines:

- Multi-label Binary Cross-Entropy
- Autoregressive Language Modeling Loss

fileciteturn3file0L43-L46

---

## 📚 Datasets

The project roadmap identifies large-scale chest X-ray datasets such as:

- **MIMIC-CXR**
- **CheXpert**

These datasets are intended to support multi-label pathology classification and vision-language model adaptation. fileciteturn3file0L34-L36

> Dataset access, licensing, preprocessing, patient de-identification, and institutional requirements should be handled according to the respective dataset policies. The supplied document does not provide dataset credentials or access procedures.

---

## 🔬 Visual Explainability

Explainability is included to help users understand which image regions contribute to model outputs.

### Grad-CAM

```text
Chest X-Ray
     │
     ▼
Model Prediction
     │
     ▼
Activation Analysis
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Overlay on X-Ray
```

### Attention Maps

Attention-map extraction can be used to visualize regions receiving strong model attention.

Example:

```text
┌─────────────────────────────┐
│                             │
│        Chest X-Ray          │
│                             │
│      ███████                │
│    ███████████              │
│       █████                 │
│                             │
│    Attention / Heatmap      │
│                             │
└─────────────────────────────┘
```

The project specification explicitly includes Grad-CAM and attention-map extraction for visual explainability and clinician verification. fileciteturn3file0L34-L38

---

## 📊 Clinical Evaluation

The project roadmap proposes evaluation using both classification and report-generation metrics.

### Classification Metrics

| Metric | Purpose |
|---|---|
| AUROC | Evaluate multi-label classification discrimination |
| F1-Score | Measure precision/recall balance |

### Report Generation Metrics

| Metric | Purpose |
|---|---|
| BLEU | Compare generated text with reference reports |
| ROUGE | Evaluate text overlap and report-generation quality |

The supplied timeline explicitly identifies **AUROC, F1-score, BLEU, and ROUGE** as evaluation metrics. fileciteturn3file0L64-L66

> Final performance numbers should be added only after evaluation is completed. The supplied document does not provide measured accuracy, AUROC, F1, BLEU, or ROUGE results.

---

## 📅 Project Phases

The project is planned across **8 weeks and four structured phases**. fileciteturn3file0L47-L48

| Phase | Milestone | Key Deliverables | Timeline |
|---|---|---|---|
| **Phase 1** | Data Ingestion & Preprocessing | DICOM parsing, dataset splits, normalization scripts | Weeks 1–2 |
| **Phase 2** | Model Adaptation & Training | MedGemma alignment, LoRA fine-tuning, multi-GPU optimization | Weeks 3–4 |
| **Phase 3** | Evaluation & Clinical Metrics | AUROC, F1, BLEU/ROUGE evaluation | Weeks 5–6 |
| **Phase 4** | Report & Code Finalization | Technical report, inference scripts, Streamlit demo | Weeks 7–8 |

These phases and deliverables are defined in the supplied project document. fileciteturn3file0L54-L71

---

## 🧰 Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Vision-Language Model | MedGemma |
| Model Framework | Hugging Face Transformers |
| Fine-Tuning | PEFT / LoRA |
| Computer Vision | Torchvision |
| Medical Imaging | pydicom |
| Data Processing | Pandas |
| Dashboard | Streamlit |
| Training | Multi-GPU |
| GPU Environment | NVIDIA A100 / RTX 4090-class hardware |

The software and compute requirements are based directly on the supplied project roadmap. fileciteturn3file0L76-L79

---

## 🏗️ Project Architecture

```text
                        ┌─────────────────────┐
                        │   DICOM / PNG X-Ray │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Image Preprocessing │
                        │                     │
                        │ • DICOM Parsing     │
                        │ • Windowing         │
                        │ • Resize            │
                        │ • Normalize         │
                        │ • CLAHE             │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Vision Encoder    │
                        │   ViT / CNN         │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     MedGemma        │
                        │ Vision-Language AI  │
                        └──────────┬──────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
       ┌─────────────────────┐          ┌─────────────────────┐
       │ Disease / Pathology │          │ Clinical Report     │
       │ Classification      │          │ Generation          │
       └──────────┬──────────┘          └──────────┬──────────┘
                  │                                 │
                  └────────────────┬────────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │ Explainability      │
                        │ Grad-CAM / Attention│
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Streamlit Demo App  │
                        └─────────────────────┘
```

---

## 🖥️ Deployment Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  USER / CLINICIAN                   │
│                                                     │
│             Upload DICOM / PNG X-Ray                │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│               STREAMLIT APPLICATION                  │
│                                                     │
│       Image Upload + Prediction + Report            │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│               PREPROCESSING LAYER                    │
│                                                     │
│       DICOM / Resize / Normalize / CLAHE            │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                 MEDGEMMA INFERENCE                   │
│                                                     │
│         Vision Encoder + Language Decoder            │
└─────────────────────────┬───────────────────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
┌──────────────────────┐   ┌──────────────────────────┐
│ Disease Predictions  │   │ Clinical Report          │
│ + Confidence         │   │ + Findings               │
└──────────┬───────────┘   └────────────┬─────────────┘
           │                            │
           └────────────┬───────────────┘
                        ▼
              ┌─────────────────────┐
              │ Explainability      │
              │ Heatmaps / Attention│
              └─────────────────────┘
```

---

## 📦 Expected Deliverables

The project is intended to deliver:

- 📄 Comprehensive technical report
- 💻 Inference source-code repository
- 🤖 Fine-tuned model weights
- 🧠 Modular training pipeline
- 📊 Evaluation pipeline
- 🩻 Chest X-ray preprocessing pipeline
- 🔍 Explainability module
- 📝 Automated report-generation module
- 🖥️ Streamlit demonstration application

These deliverables are specified in the project roadmap. fileciteturn3file0L76-L79

---

## 📁 Suggested Repository Structure

```text
MedGemma-Chest-XRay-Disease-Identification/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── models/
│   ├── base/
│   └── fine_tuned/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── explainability/
│
├── configs/
│   └── config.yaml
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│
├── reports/
│
├── images/
│   ├── xray-prediction.png
│   ├── heatmap.png
│   └── dashboard.png
│
└── LICENSE
```

> This is a recommended repository organization for implementation. The supplied project document does not specify an exact repository structure.

---

## 📸 Screenshots

Add project screenshots to the `images/` directory.

### 🩻 X-Ray Prediction

```text
images/xray-prediction.png
```

![X-Ray Prediction](images/xray-prediction.png)

### 🔍 Explainability Heatmap

```text
images/heatmap.png
```

![Explainability Heatmap](images/heatmap.png)

### 🖥️ Streamlit Dashboard

```text
images/dashboard.png
```

![Streamlit Dashboard](images/dashboard.png)

> Replace the placeholder image paths with actual screenshots after implementation.

---

## 🎥 Demo

Add a project demonstration video or GIF here.

```text
demo/medgemma-chest-xray-demo.mp4
```

---

## 🚀 Future Enhancements

Potential future improvements include:

- Additional chest X-ray pathology classes
- Improved abnormality localization
- Larger and more diverse training datasets
- Better report-generation evaluation
- Advanced uncertainty estimation
- Human-in-the-loop report review
- PACS/RIS integration
- DICOM metadata-aware processing
- Multi-view chest X-ray analysis
- Clinical workflow integration
- Model monitoring and drift detection
- Optimized inference for clinical environments

These are proposed extensions and are not presented as completed capabilities in the supplied project document.

---

## ⚕️ Medical Safety Notice

**This project is intended for research, educational, and prototype purposes.**

AI-generated disease predictions, localization maps, confidence scores, and clinical reports **must not be treated as a standalone medical diagnosis**.

Any real-world clinical deployment would require appropriate:

- Clinical validation
- Regulatory review
- Data governance
- Privacy protections
- Security controls
- Human expert oversight
- Institutional approval
- Prospective evaluation

The supplied project document describes the technical roadmap but does not define a complete regulatory or clinical deployment framework.

---

## 📜 License

This project is intended for educational, research, and prototype development purposes unless a separate license is provided by the project owner.

---

<p align="center">

**MedGemma Chest X-Ray Disease Identification**

*Multimodal AI for research-oriented chest X-ray analysis and clinical decision support.*

</p>
