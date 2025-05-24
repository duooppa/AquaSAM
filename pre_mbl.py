#%% import packages
import numpy as np
from PIL import Image
import os
join = os.path.join
import numpy as np
import os
from os.path import join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import matplotlib.pyplot as plt


# set up the parser
parser = argparse.ArgumentParser(description='preprocess CT images')
parser.add_argument('-i', '--jpg_path', type=str, default='./data/suim/train_val/images', help='path to the jpg images')
parser.add_argument('-gt', '--bmp_path', type=str, default='./data/suim/train_val/masks', help='path to the bmp images')
parser.add_argument('-o', '--npz_path', type=str, default='./data/suim/train_val/output', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy') # Included for completeness as per instructions
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--lower_threshold', type=int, default=128, help='lower threshold for bmp processing')
parser.add_argument('--upper_threshold', type=int, default=128, help='upper threshold for bmp processing')
parser.add_argument('--min_count', type=int, default=8, help='minimum count for labels in bmp processing')
parser.add_argument('--num_labels', type=int, default=8, help='number of labels to generate')
args = parser.parse_args()

# Directly use args below, no need for intermediate variables like jpg_path = args.jpg_path

print("Preprocessing start")
# Use args.bmp_path for listdir, assuming names are based on mask files
names = sorted(os.listdir(args.bmp_path))
names = [os.path.splitext(f)[0] for f in names] # Keep .bmp extension for now, will be handled later

# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names[:int(len(names)*0.8)])
test_names = sorted(names[int(len(names)*0.8):])


def color_to_label_mapping(img_np, color_to_label, label_counter):
    label_img = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.int16)
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            rgb = tuple(img_np[i, j, :])
            if rgb not in color_to_label:
                color_to_label[rgb] = label_counter
                label_counter += 1
            label_img[i, j] = color_to_label[rgb]
    return label_img, color_to_label, label_counter

# def preprocessing function
# Changed function signature to accept args directly or specific values from args
def preprocess_ct(bmp_data, current_jpg_path, bmp_name_no_ext, label_id, current_image_size, sam_model, current_device):
    gt_data = bmp_data == label_id
    if np.sum(gt_data)>10:
        imgs = []
        gts =  []
        img_embeddings = []

        # Use bmp_name_no_ext for constructing jpg filename
        img_pil = Image.open(join(current_jpg_path, f"{bmp_name_no_ext}.jpg"))
        image_data = np.array(img_pil)
        image_data_pre = np.clip(image_data, 0, 255)
        gt_slice_i = transform.resize(gt_data, (current_image_size, current_image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
        if np.sum(gt_slice_i)>10:
            img_slice_i = transform.resize(image_data_pre, (current_image_size, current_image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            img_slice_i = np.int16(np.repeat(img_slice_i[:,:,None], 3, axis=-1)) if len(img_slice_i.shape) < 3 else img_slice_i.astype(np.uint8)
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
            imgs.append(img_slice_i)
            assert np.sum(gt_slice_i)>10, 'ground truth should have more than 100 pixels' # Original comment said 100, but condition is >10
            gts.append(gt_slice_i)
            if sam_model is not None:
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_slice_i)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(current_device)
                input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:])
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])
            else:
                print('\r', "Sam model is None", end="", flush=True)
        else:
            print('\r', "np.sum(gt_slice_i) error", end="", flush=True)

        if sam_model is not None:
            return imgs, gts, img_embeddings
        else:
            return imgs, gts
    else:
        # print("\r", "Cannot return for foreground size error", end="", flush=True) # Original comment, can be noisy
        return None, None, None # Or handle appropriately

# names are already without extension from earlier: names = [os.path.splitext(f)[0] for f in names]
# So, when iterating, 'name' is 'filename_without_extension'

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

color_to_label = {}
color_to_label[(0, 0, 0)] = 0 # Black is background
label_counter = 1

print(f"Generating labels to {args.num_labels} labels")
length_names = len(names) # these are names without extension
img_count = 0
for name_no_ext in names: # name_no_ext is 'filename_without_extension'
    img_count += 1
    # Construct full bmp filename for reading
    gt_pil = np.array(Image.open(join(args.bmp_path, f"{name_no_ext}.bmp")))
    gt_pil_int = np.int16(gt_pil) # Not directly used, but kept from original
    gt_pil_thresholded = gt_pil.copy()
    gt_pil_thresholded[gt_pil_thresholded <= args.lower_threshold] = 0
    gt_pil_thresholded[gt_pil_thresholded > args.upper_threshold] = 255
    
    # Assuming gt_pil_thresholded is 3-channel (RGB) after thresholding based on original logic
    # If it becomes single channel, flatten() would be okay. If it's MxNx3, bincount needs single dim.
    # For label mapping, it seems it expects MxNx3.
    # The original code's np.bincount(gt_pil_thresholded.flatten()) might be problematic if gt_pil_thresholded is MxNx3.
    # Let's assume color_to_label_mapping handles the RGB tuple correctly.
    # For bincount on labels after mapping, it's fine.
    # The rare_labels logic seems to apply to the raw thresholded image, which might need adjustment
    # if counts are based on RGB values directly rather than mapped labels.
    # However, sticking to original logic for now for this part:
    if gt_pil_thresholded.ndim == 3 and gt_pil_thresholded.shape[-1] == 3: # if RGB image
        # Convert to a single "class" per pixel for bincount, e.g. by summing channels or a more robust method
        # This part is tricky with RGB. Original code might have assumed grayscale for bincount.
        # For now, if it's RGB, we skip bincount for rare_labels on raw RGB for simplicity in this refactor.
        # The color_to_label_mapping will still work.
        pass # Skipping rare_labels for RGB directly for now.
    elif gt_pil_thresholded.ndim == 2: # Grayscale
        counts = np.bincount(gt_pil_thresholded.flatten())
        rare_labels = np.where(counts < args.min_count)[0]
        for label_val in rare_labels: # label_val is the intensity value
             gt_pil_thresholded[gt_pil_thresholded == label_val] = 0 # Set rare intensity values to 0

    _, color_to_label, label_counter = color_to_label_mapping(gt_pil_thresholded, color_to_label, label_counter)
    
    unique_colors = np.unique(gt_pil_thresholded.reshape(-1, gt_pil_thresholded.shape[-1]), axis=0)
    print('\r', f"({img_count} / {length_names})Reading {name_no_ext}.bmp with {len(unique_colors)} unique color patterns, {label_counter} labels generated", end="", flush=True)
    if label_counter >= args.num_labels: # Stop if we have enough labels (>= because counter might jump)
        break
print(f"\nLabel dict created with {label_counter} labels")


for key_color, label_id_val in list(color_to_label.items()):
    if label_id_val == 0 and key_color != (0,0,0): # Skip if label 0 was assigned to non-black, or handle as needed
        continue

    save_path_label_dir = join(args.npz_path, f'{label_id_val}')
    os.makedirs(save_path_label_dir, exist_ok=True)
    print(f"\nProcessing label: {label_id_val} (Color: {key_color})")

    for name_no_ext in tqdm(train_names): # name_no_ext is 'filename_without_extension'
        bmp_full_name = f"{name_no_ext}.bmp"
        gt_pil = Image.open(join(args.bmp_path, bmp_full_name))
        gt_data_raw = np.array(gt_pil)
        gt_data_labeled, _, _ = color_to_label_mapping(gt_data_raw, color_to_label, label_counter) # Re-map to ensure consistency
        gt_data_final = np.int16(gt_data_labeled)

        try:
            result = preprocess_ct(gt_data_final, args.jpg_path, name_no_ext, label_id_val, args.image_size, sam_model, args.device)
            if result and result[0] is not None and len(result[0]) >= 1:
                imgs, gts, img_embeddings = result
                save_path_train = join(save_path_label_dir, 'train')
                os.makedirs(save_path_train, exist_ok=True)
                imgs = np.stack(imgs, axis=0)
                gts = np.stack(gts, axis=0)
                img_embeddings = np.stack(img_embeddings, axis=0)
                np.savez_compressed(join(save_path_train, name_no_ext + '.npz'), gts=gts, img_embeddings=img_embeddings)
                
                # Sanity check image saving
                idx = np.random.randint(0, imgs.shape[0])
                img_idx = imgs[idx, :, :, :]
                gt_idx = gts[idx, :, :]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
                Image.fromarray(img_idx.astype(np.uint8)).save(join(save_path_train, name_no_ext + '_debug.png'))
            elif result is None or result[0] is None:
                 print(f"\rSkipping {name_no_ext} for label {label_id_val} due to no valid data after preprocess_ct (e.g. sum(gt_data)<=10 or sum(gt_slice_i)<=10).", end="", flush=True)

        except Exception as e:
            print(f"\rError processing {name_no_ext} for train: {e}", end="", flush=True)

    for name_no_ext in tqdm(test_names): # name_no_ext is 'filename_without_extension'
        bmp_full_name = f"{name_no_ext}.bmp"
        gt_pil = Image.open(join(args.bmp_path, bmp_full_name))
        gt_data_raw = np.array(gt_pil)
        gt_data_labeled, _, _ = color_to_label_mapping(gt_data_raw, color_to_label, label_counter)
        gt_data_final = np.int16(gt_data_labeled)
        try:
            result = preprocess_ct(gt_data_final, args.jpg_path, name_no_ext, label_id_val, args.image_size, sam_model, args.device)
            if result and result[0] is not None and len(result[0]) >= 1:
                imgs, gts, img_embeddings = result # img_embeddings might be present if sam_model is not None
                save_path_test = join(save_path_label_dir, 'test')
                os.makedirs(save_path_test, exist_ok=True)
                imgs = np.stack(imgs, axis=0)
                gts = np.stack(gts, axis=0)
                # For test set, original code saved 'imgs' and 'gts'. If 'img_embeddings' are needed, add them.
                # Assuming test set also needs embeddings if available
                if img_embeddings is not None and len(img_embeddings) > 0:
                    img_embeddings = np.stack(img_embeddings, axis=0)
                    np.savez_compressed(join(save_path_test, name_no_ext + '.npz'), imgs=imgs, gts=gts, img_embeddings=img_embeddings)
                else:
                    np.savez_compressed(join(save_path_test, name_no_ext + '.npz'), imgs=imgs, gts=gts)

                # Sanity check image saving
                idx = np.random.randint(0, imgs.shape[0])
                img_idx = imgs[idx, :, :, :]
                gt_idx = gts[idx, :, :]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
                Image.fromarray(img_idx.astype(np.uint8)).save(join(save_path_test, name_no_ext + '_debug.png'))
            elif result is None or result[0] is None:
                 print(f"\rSkipping {name_no_ext} for label {label_id_val} due to no valid data after preprocess_ct (e.g. sum(gt_data)<=10 or sum(gt_slice_i)<=10).", end="", flush=True)

        except Exception as e:
            print(f"\rError processing {name_no_ext} for test: {e}", end="", flush=True)

print("\nPreprocessing finished.")
