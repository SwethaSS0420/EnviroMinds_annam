# Soil Type Classification – Challenge 1 and 2

This repository contains two challenges for soil type classification using image-based models. It includes scripts for data downloading, preprocessing, training, inference, and evaluation of performance metrics.


## Directory Structure

## Dataset

### Challenge 1:
- Dataset: [`soil-classification-1`](https://www.kaggle.com/datasets)  
- Kaggle path: `/kaggle/input/soil-classification-1`

### Challenge 2:
- Dataset: [`soil-binary`](https://www.kaggle.com/datasets)  
- Kaggle path: `/kaggle/input/soil-binary`

To download the datasets locally, run the script from the root directory of each challenge:

```bash
bash data/download.sh
```
## Setup and Installation

1. **Clone the repository:**

```bash
git clone https://github.com/SwethaSS0420/soil-classification.git
cd soil-classification/challenge-1  # or challenge-2
```
2. **Install required Python packages:**

```bash
pip install -r requirements.txt
```

## Training

* Run the training notebook to train the anomaly detection models on the soil images:
  notebooks/training.ipynb

## Inference

* Use the inference notebook to generate predictions on test data using trained models:
  notebooks/inference.ipynb

## Results

* Class-wise F1 scores are available in [docs/cards/ml-metrics.json]
* Model architecture diagram is in [docs/cards/architecture.png]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors
Sanjana Sudarsan - sanjanas01 (https://github.com/sanjanas01) \
Swetha Sriram - SwethaSS0420 (https://github.com/SwethaSS0420) \
Lohithaa KM - Lohi14 (https://github.com/Lohi14)
