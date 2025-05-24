# AquaSAM
This repository is the official PyTorch implementation of AquaSAM: Foreground Underwater Image Segmentation. ([arxiv](https://arxiv.org/abs/2308.04218)). 
AquaSAM is the first attempt to extend the success of segment-anything model in the domains of underwater images.

## Installation
1. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
2. `git clone https://github.com/duooppa/AquaSAM`
3. Enter the AquaSAM folder `cd AquaSAM` and run `pip install -e .`

## Fine-tune SAM on customized dataset

We provide a step-by-step tutorial with a SUIM dataset to help you quickly start the training process.

### Data preparation and preprocessing

**Expected Data Format:**
The preprocessing scripts provided in this repository (e.g., `pre_mbl.py`, `pre_ml.py`, `pre_bw.py`) expect images to be in JPG format and corresponding masks in BMP format. If your dataset is in a different format (e.g., .nii files), you will need to convert your images and masks to JPG and BMP respectively before using these scripts.

**Preprocessing Steps:**
The preprocessing scripts typically perform the following:
- Split dataset: e.g., 80% for training and 20% for testing.
- Image normalization.
- Pre-compute image embeddings using a SAM model.

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip it. Ensure it follows the JPG/BMP format.

**Preprocessing Scripts and Arguments:**

There are three main preprocessing scripts tailored for different mask types:

1.  **`pre_mbl.py` (Multi-Label Masks):**
    This script is designed for scenarios where your BMP masks contain multiple distinct classes/labels, each represented by a different color. It identifies these colors, maps them to integer labels, and processes each label separately.
    ```bash
    python pre_mbl.py \
        -i /path/to/your/jpg_images \
        -gt /path/to/your/bmp_masks \
        -o /path/to/output_npz_files \
        --image_size 256 \
        --model_type "vit_b" \
        --checkpoint /path/to/base/sam_vit_b_01ec64.pth \
        --device "cuda:0" \
        --seed 2023 \
        --lower_threshold 128 \
        --upper_threshold 128 \
        --min_count 8 \
        --num_labels 8 
    ```
    Key arguments for `pre_mbl.py`:
    *   `-i, --jpg_path`: Path to the JPG images.
    *   `-gt, --bmp_path`: Path to the BMP masks.
    *   `-o, --npz_path`: Path to save the output NPZ files.
    *   `--image_size`: Target size for resizing images and masks.
    *   `--model_type`: SAM model type (e.g., "vit_b").
    *   `--checkpoint`: Path to the base SAM checkpoint.
    *   `--device`: CUDA device (e.g., "cuda:0").
    *   `--seed`: Random seed for reproducibility.
    *   `--lower_threshold`, `--upper_threshold`: Thresholds for initial mask processing.
    *   `--min_count`: Minimum pixel count for a color to be considered a label.
    *   `--num_labels`: Number of distinct labels to process.

2.  **`pre_ml.py` (Multi-Color Masks to Mapped Labels):**
    This script is suited for BMP masks where different colors represent different objects/classes. It maps these colors to integer labels for processing.
    ```bash
    python pre_ml.py \
        -i /path/to/your/jpg_images \
        -gt /path/to/your/bmp_masks \
        -o /path/to/output_npz_files \
        --image_size 256 \
        --model_type "vit_b" \
        --checkpoint /path/to/base/sam_vit_b_01ec64.pth \
        --device "cuda:0" \
        --seed 2023 \
        --show_plot
    ```
    Key arguments for `pre_ml.py`:
    *   `-i, --jpg_path`: Path to the JPG images.
    *   `-gt, --bmp_path`: Path to the BMP masks.
    *   `-o, --npz_path`: Path to save the output NPZ files.
    *   `--image_size`: Target image size.
    *   `--model_type`: SAM model type.
    *   `--checkpoint`: Path to the base SAM checkpoint.
    *   `--device`: CUDA device.
    *   `--seed`: Random seed.
    *   `--show_plot`: If present, shows plots during preprocessing (e.g., the mapped label mask).
    *   `--modality`, `--anatomy`, `--img_name_suffix`, `--label_id`: Additional arguments that are parsed but may have limited effect on the core logic of `pre_ml.py` which uses `color_to_label_mapping`.

3.  **`pre_bw.py` (Binary/Single-Label Masks):**
    This script is for binary segmentation tasks where the BMP mask uses a specific pixel value (label_id) to denote the foreground, and all other pixel values are considered background.
    ```bash
    python pre_bw.py \
        -i /path/to/your/jpg_images \
        -gt /path/to/your/bmp_masks \
        -o /path/to/output_npz_files \
        --image_size 256 \
        --label_id 255 \
        --model_type "vit_b" \
        --checkpoint /path/to/base/sam_vit_b_01ec64.pth \
        --device "cuda:0" \
        --seed 2023
    ```
    Key arguments for `pre_bw.py`:
    *   `-i, --jpg_path`: Path to the JPG images.
    *   `-gt, --bmp_path`: Path to the BMP masks.
    *   `-o, --npz_path`: Path to save the output NPZ files.
    *   `--image_size`: Target image size.
    *   `--label_id`: The pixel value in the BMP mask that represents the foreground.
    *   `--model_type`: SAM model type.
    *   `--checkpoint`: Path to the base SAM checkpoint.
    *   `--device`: CUDA device.
    *   `--seed`: Random seed.
    *   `--modality`, `--anatomy`, `--img_name_suffix`: Additional arguments that are parsed but may have limited effect on the core logic if image names are directly inferred from `jpg_path`.

Choose the script that best matches your mask format. After running preprocessing, NPZ files containing image embeddings and ground truth masks will be saved in the specified output directory, typically in `train` and `test` subfolders.

### Models and Checkpoints

**Base SAM Checkpoint:**
Download the base [SAM checkpoint (sam_vit_b_01ec64.pth)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it, for example, at `work_dir/SAM/sam_vit_b_01ec64.pth`. You will need to provide the path to this checkpoint when running preprocessing and training scripts.

**Fine-tuned AquaSAM Model Weights:**
Fine-tuned AquaSAM model weights are not part of this repository by default. If released by the authors, they would typically be announced and provided through their official project page or communication channels. The training script (`train.py`) will save your fine-tuned model checkpoints in your specified working directory.

### Model Training

The `train.py` script handles the fine-tuning of the SAM model on your preprocessed dataset.

**Key features of `train.py`:**
*   **Combined Loss Function:** The training process uses a combination of DiceCE loss (for segmentation) and an MSE regression loss (for predicting IoU scores). This helps the model learn both good segmentation quality and an accurate estimate of its own performance.
*   **Configurable Parameters:** Various aspects of training can be controlled via command-line arguments.

**Command-Line Arguments for `train.py`:**
```bash
python train.py \
    -i /path/to/your/preprocessed_npz_train_data \
    --task_name "AquaSAM-Experiment" \
    --model_type "vit_b" \
    --checkpoint /path/to/base/sam_vit_b_01ec64.pth \
    --device "cuda:0" \
    --work_dir ./my_experiment_output \
    --num_epochs 300 \
    --batch_size 8 \
    --lr 1e-5 \
    --weight_decay 0 \
    --show_plot
```
*   `-i, --npz_tr_path`: Path to the preprocessed NPZ training data.
*   `--task_name`: Name for the training task/experiment.
*   `--model_type`: SAM model type (e.g., "vit_b").
*   `--checkpoint`: Path to the base SAM checkpoint to start fine-tuning from.
*   `--device`: CUDA device (e.g., "cuda:0", "cuda:1").
*   `--work_dir`: Directory to save model checkpoints and logs.
*   `--num_epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--lr`: Learning rate.
*   `--weight_decay`: Weight decay for the optimizer.
*   `--show_plot`: If present, displays the loss plot window during training (the plot is always saved to file).

**Important Note on Training:**
A `RuntimeError` might occur during training if `dense_prompt_embeddings` are used (e.g., when providing bounding boxes as prompts). This is due to a potential shape mismatch within the SAM `MaskDecoder` when processing batched inputs. A workaround has been implemented in `train.py` which expands the `dense_prompt_embeddings` to match the expected internal dimensions, resolving this issue.

You can also train the model on the whole dataset. For example, download the full SUIM training set from [SUIM dataset](https://irvlab.cs.umn.edu/resources/suim-dataset), preprocess it using one of the scripts above, and then run `train.py`.

## Running Inference

To run inference using your fine-tuned AquaSAM model, use the `AquaSAM_Inference.py` script. This script takes arguments for the data path (containing images to segment), the path to your trained AquaSAM model checkpoint, output directories, and other parameters.

Example usage:
```bash
python AquaSAM_Inference.py \
    -i /path/to/your/test_data_npz_or_image_folders \
    -o /path/to/output_segmentations_npz \
    --seg_png_path /path/to/output_segmentation_pngs \
    --checkpoint /path/to/your/finetuned_aquasam_checkpoint.pth \
    --device cuda:0
```
Refer to `AquaSAM_Inference.py --help` for more details on its command-line arguments. The script will save segmentation masks and can optionally save visualization images.
