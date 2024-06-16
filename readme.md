
# White Spot Detection and Analysis

## Overview

This repository contains scripts and notebooks for detecting and analyzing white spots in solar images using machine learning models. The project involves data preparation, model training, and evaluation using various neural network architectures including LSTM, GRU, and CNN-LSTM models.

## Contents

### Notebooks

- `data_preparation.ipynb`: Jupyter notebook for preparing the dataset by reading, processing, and merging data from various sources.
- `Cleaned_CHRONNOS.ipynb`: Cleaned version of the Chronnos notebook used for extracting white spots.

### Scripts

- `local_script.py`: Python script for preparing the dataset. It includes functions to read data, process images, and extract white spots.
- `base_lstm.py`: Python script for training and evaluating an LSTM model on the prepared dataset.
- `base_gru.py`: Python script for training and evaluating a GRU model on the prepared dataset.
- `cnn_lstm.py`: Python script for training and evaluating a CNN-LSTM model.
- `cnn_lstm_pretrained.py`: Python script for retraining a pretrained CNN-LSTM model.

### Data Preparation

The `data_preparation.py` script is responsible for preparing the dataset. It reads the `join.csv` file, processes image data, and extracts white spots using the following function:

```python
def quantify_white_spots_info_in_folder(folder_path):
    results_df = pd.DataFrame(columns=['Image', 'White_Spot_Percentage', 'Num_White_Spots'])
    for filename in os.listdir(folder_path):
        if filename.endswith('.zip'):
            zip_path = os.path.join(folder_path, filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
            for extracted_filename in os.listdir(folder_path):
                if extracted_filename.endswith('.fits.gz'):
                    fits_path = os.path.join(folder_path, extracted_filename)
                    with fits.open(fits_path) as hdul:
                        image_data = hdul[0].data
                    grayscale_image = np.squeeze(image_data)
                    threshold_value = np.mean(grayscale_image)
                    binary_image = np.where(grayscale_image >= threshold_value, 1, 0)
                    labeled_image = label(binary_image)
                    num_white_spots = np.max(labeled_image)
                    total_pixels = binary_image.size
                    white_spot_percentage = (num_white_spots / total_pixels) * 100
                    new_row = pd.DataFrame({
                        'Image': [extracted_filename],
                        'White_Spot_Percentage': [white_spot_percentage],
                        'Num_White_Spots': [num_white_spots]
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
            for extracted_filename in os.listdir(folder_path):
                if extracted_filename.endswith('.fits.gz'):
                    extracted_path = os.path.join(folder_path, extracted_filename)
                    os.remove(extracted_path)
    return results_df
```

### Model Training and Evaluation

The repository includes scripts for training and evaluating different neural network architectures:

- `base_lstm.py`: Trains and evaluates an LSTM model.
- `base_gru.py`: Trains and evaluates a GRU model.
- `cnn_lstm.py`: Trains and evaluates a CNN-LSTM model.
- `cnn_lstm_pretrained.py`: Retrains a pretrained CNN-LSTM model and evaluates its performance.

Each script follows a similar structure, including loading the dataset, defining the model architecture, training the model, and evaluating its performance.

### Setup and Usage

#### Running Locally

Clone the repository:

```bash
git clone https://github.com/your-username/white-spot-detection.git
cd white-spot-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data preparation script:

```bash
python local_script.py
```

Train and evaluate models:

```bash
python base_lstm.py
python base_gru.py
python cnn_lstm.py
python cnn_lstm_pretrained.py
```

#### Running in Jupyter/Google Colab

1. Upload the notebooks to your Jupyter environment or Google Colab.
2. Run the cells in `data_preparation.ipynb` to prepare the dataset.
3. Run the cells in `Cleaned_CHRONNOS.ipynb` to extract white spots.

### Acknowledgments

This project uses the Chronnos repository for white spot extraction. Special thanks to the authors of the Chronnos project. For more information, visit Chronnos on GitHub.

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.
