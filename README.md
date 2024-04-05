# AquaSAM
the first at-tempt to extend the success of SAM on underwater images with the purpose of creating a versatile tool for underwater image segmentation
## Installation 
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Fine-tune SAM on customized dataset

We provide a step-by-step tutorial with a SUIM dataset to help you quickly start the training process.

### Data preparation and preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip.

In this tutorial, we will fine-tune SAM for foreground underwater image segmentation.

Run pre-processing

```bash
python pre_mbl.py
- split dataset: 80% for training and 20% for testing
- image normalization
- pre-compute image embedding

### Model Training

Please check the step-by-step tutorial: `train.py`.

You can also train the model on the whole dataset. 
Download the training set (https://irvlab.cs.umn.edu/resources/suim-dataset)
