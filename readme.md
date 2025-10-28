# EmotionGANverse

A unified deep learning project implementing and comparing four powerful Generative Adversarial Network (GAN) models: **DCGAN**, **CGAN**, **BigGAN**, and **StarGAN**. Each model is developed from scratch using Tensorflow, enabling high-quality image generation and facial expression transformation.Each GAN model has its own specifications and usage.the website of our model will also be availible.


## 📂 Project Structure

```
EmotionGANverse/
├── BigGAN/
│   ├── sample op/
│   ├── training_metrics/
│   ├── biggan-test-code-2-fer-2013-ck.ipynb
│   └── requirements.txt
│
├── CGAN/
│   ├── evaluation/
│   ├── models/
│   ├── sample op/
│   ├── scripts/
│   ├── Test images/
│   ├── training/
│   ├── training metrics/
│   ├── utils/
│   ├── main.py
│   ├── requirements.txt
│   └── run_inference.py
│
├── DCGAN/
│   ├── __pycache__/
│   ├── models/
│   ├── results/
│   ├── samples/
│   ├── utils/
│   ├── config.py
│   ├── featureextraction.py
│   ├── main.py
│   ├── testcase.py
│   ├── train.py
│   └── training_history.json
│
├── StarGAN/
│   ├── metrices/
│   ├── results/
│   ├── sampleimages/
│   └── model.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🧠 Model Overview

### ✨ DCGAN (Deep Convolutional GAN)

* **Purpose**: Learn to generate realistic images from random noise using convolutional layers.
* **Highlights**: Built from scratch, uses batch normalization, LeakyReLU, and transposed convolutions.
* **Output**: Sampled images of a specific domain after training.

### 😊 CGAN (Conditional GAN)

* **Purpose**: Generate facial expressions based on emotion labels (happy, sad, angry).
* **Highlights**: Trained on FER-2013 dataset, includes condition vectors, works for emotion-to-face mapping.
* **Output**: Transformed facial images based on given expressions.

### 🔥 BigGAN

* **Purpose**: Class-conditional high-resolution image generation.
* **Highlights**: Uses large-scale training and improved class embeddings, known for superior image quality.
* **Output**: Image samples of diverse categories with better quality.

### 🚀 StarGAN

* **Purpose**: Multi-domain facial attribute and expression translation.
* **Highlights**: A single model handles transformations across multiple emotion domains.
* **Output**: Original face converted into different emotions like happy, sad, angry.

## 🎯 Use Cases

* Facial expression editing for image apps
* Emotion synthesis for avatars and virtual agents
* Conditional image generation for creative AI
* Benchmarking different GAN architectures

## 📅 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/deephabiswashi/EmotionGANverse.git
cd EmotionGANverse
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models

Each model has its own `train.py`:

```bash
cd CGAN
python train.py
```

### 4. Run Inference

```bash
cd CGAN
python run_inference.py
```

## 📊 Sample Results

* Generated faces with varying emotions
* Realistic outputs from BigGAN and StarGAN
* DCGAN-generated digits or faces depending on dataset used

## 🔧 Requirements

All dependencies are listed in `requirements.txt`. Common ones include:

```

numpy
matplotlib
opencv-python
Pillow
scikit-learn
tqdm
seaborn
jupyter
imageio
tensorboard
```

## 📄 License

This project is released for academic and non-commercial use only.

---


