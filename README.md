# Siamese Network for Few-Shot Learning

## Overview
This project implements a **Siamese Network** for Few-Shot Learning, designed to classify images by learning to compare pairs or triplets of images. The network is trained using the **Triplet Loss** function, which ensures that the distance between an anchor and a positive example (similar) is minimized, while the distance between the anchor and a negative example (dissimilar) is maximized. The implementation uses PyTorch with the EfficientNet-B0 backbone and is tailored for a custom dataset of images, loaded via a CSV file. The project includes training, validation, and inference phases, with embeddings saved for downstream tasks.

## Requirements
To run this project, install the following dependencies:
- Python 3.8+
- PyTorch
- torchvision
- timm
- pandas
- NumPy
- Matplotlib
- scikit-learn
- scikit-image
- tqdm

Install dependencies using:
```bash
pip install torch torchvision timm pandas numpy matplotlib scikit-learn scikit-image tqdm
```

## Dataset
The dataset is defined in a CSV file (specified by `CSV_FILE`) containing columns for **Anchor**, **Positive**, and **Negative** image filenames. Images are stored in a directory (specified by `DATA_DIR`). Each image is loaded, normalized to [0, 1], and transformed into a tensor with shape (channels, height, width). The dataset is split into training (80%) and validation (20%) sets.

## Model Architecture
The Siamese Network uses the **EfficientNet-B0** model (pretrained) as the backbone for feature extraction, with the following configuration:
- **Input**: Images of shape (channels, height, width)
- **Backbone**: EfficientNet-B0 (from `timm`)
- **Classifier**: Modified to output embeddings of size 512
- **Output**: 512-dimensional embedding vector for each input image

The network processes anchor, positive, and negative images to produce embeddings, which are then used to compute the triplet loss.

## Training
- **Batch Size**: 32
- **Epochs**: 15
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Triplet Margin Loss
- **Device**: CUDA (GPU) if available, otherwise CPU

The training loop:
1. Processes triplets (anchor, positive, negative) in batches.
2. Computes embeddings for each image in the triplet.
3. Applies triplet loss to minimize anchor-positive distance and maximize anchor-negative distance.
4. Saves the model with the lowest validation loss as `best_model.pt`.

## Code Structure
The project is implemented in a Python script or Jupyter Notebook with the following components:

1. **Data Loading**:
   - Loads dataset from a CSV file and splits into train/validation sets.
   - Uses a custom `APN_Dataset` class to load anchor, positive, and negative images.

2. **Model Definition**:
   - Defines `APN_Model` using EfficientNet-B0 with a modified classifier for 512-dimensional embeddings.

3. **Training and Evaluation**:
   - `train_fn`: Trains the model using triplet loss, updating weights via backpropagation.
   - `eval_fn`: Evaluates the model on the validation set without gradient updates.

4. **Embedding Generation**:
   - `get_encoding_csv`: Generates embeddings for anchor images and saves them to `encodings.csv`.

5. **Inference**:
   - Computes Euclidean distances between a test image's embedding and anchor embeddings to find the closest matches.

## Usage
1. Clone the repository or download the code.
2. Set `DATA_DIR` to the path of your image directory and `CSV_FILE` to the path of your CSV file.
3. Install dependencies (see Requirements).
4. Run the script or notebook:
   ```bash
   python siamese_network.py
   ```
   or execute the cells in a Jupyter Notebook.
5. Monitor training progress via printed train and validation loss.
6. After training, use the inference section to compute distances for a test image and find similar images.

## Example Output
- **Training**: Outputs train and validation loss per epoch. The best model is saved as `best_model.pt`.
- **Embeddings**: Generates `encodings.csv` with anchor image embeddings.
- **Inference**: Computes Euclidean distances to identify the closest anchor images to a test image.

## Notes
- Ensure the CSV file has columns `Anchor`, `Positive`, and `Negative` with valid image filenames.
- Use a GPU for faster training (set `DEVICE='cuda'`).
- Adjust hyperparameters (e.g., learning rate, epochs, embedding size) for better performance.
- The quality of few-shot learning depends on the diversity and quality of the dataset.

## References
- PyTorch Documentation: [pytorch.org](https://pytorch.org)
- TIMM Library: [github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
