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
# parser = argparse.ArgumentParser(description='preprocess CT images')
# parser.add_argument('-i', '--jpg_path', type=str, default='./data/suim/train_val/images', help='path to the jpg images')  # Changed from 'nii_path' to 'jpg_path'
# parser.add_argument('-gt', '--bmp_path', type=str, default='./data/suim/train_val/masks', help='path to the bmp images')  # Changed from 'gt_path' to 'bmp_path'
# parser.add_argument('-o', '--npz_path', type=str, default='./data/suim/train_val/output', help='path to save the npz files')
#
# parser.add_argument('--image_size', type=int, default=256, help='image size')
# parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy')
# parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
# parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
# parser.add_argument('--device', type=str, default='cuda:0', help='device')
# parser.add_argument('--seed', type=int, default=2023, help='random seed')
# args = parser.parse_args()

jpg_path = './data/Npz_files/output/train_val/images'
bmp_path = './data/Npz_files/output/train_val/masks'
npz_path = './data/Npz_files/output'
image_size = 256

anatomy = 'Abd-Gallbladder'
model_type = 'vit_b'
checkpoint = 'work_dir/SAM-ViT-B/sam_vit_b_01ec64.pth'

device = 'cuda:0'

seed = 2023
# Other unchanged arguments...
lower_threshold = 128
upper_threshold = 128
min_count = 8
num_labels = 8


print("Preprocessing start")
names = sorted(os.listdir(jpg_path))  # Changed from 'gt_path' to 'bmp_path'
names = [os.path.splitext(f)[0] for f in names]

# names = [name for name in names if not os.path.exists(join(npz_path, prefix + '_' + name.split('.bmp')[0]+'.npz'))]  # Changed '.nii.gz' to '.bmp'
# names = [name for name in names if os.path.exists(join(jpg_path, name.split('.bmp')[0] + img_name_suffix))]  # Changed 'nii_path' to 'jpg_path' and '.nii.gz' to '.bmp'

# split names into training and testing
np.random.seed(seed)
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
def preprocess_ct(bmp_data, jpg_path, bmp_name, label_id, image_size, sam_model, device):

    gt_data = bmp_data == label_id

    if np.sum(gt_data)>10:
        imgs = []
        gts =  []
        img_embeddings = []

        img_pil = Image.open(join(jpg_path, f"{bmp_name}.jpg"))  # read jpg image
        image_data = np.array(img_pil)
        image_data_pre = np.clip(image_data, 0, 255)  # bmp range
        gt_slice_i = transform.resize(gt_data, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
        if np.sum(gt_slice_i)>10:
            # print("np.sum(gt_slice_i)>100:")
            img_slice_i = transform.resize(image_data_pre, (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            img_slice_i = np.int16(np.repeat(img_slice_i[:,:,None], 3, axis=-1)) if len(img_slice_i.shape) < 3 else img_slice_i.astype(np.uint8)
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
            imgs.append(img_slice_i)
            assert np.sum(gt_slice_i)>10, 'ground truth should have more than 100 pixels'
            gts.append(gt_slice_i)
            if sam_model is not None:
                # print("sam_model is not None:")
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_slice_i)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])
            else:
                print('\r', "Sam model is None", end="", flush=True)
        else:
            print('\r', "np.sum(gt_slice_i) error", end="", flush=True)

        if sam_model is not None:
            # print("imgs: ", imgs, "gts: ", gts, "img_embeddings: ", img_embeddings)
            return imgs, gts, img_embeddings
        else:
            # print("imgs: ", imgs, "gts: ", gts)
            return imgs, gts
    else:
        print("\r", "Cannot return for foreground size error", end="", flush=True)


names = sorted(os.listdir(bmp_path))  # read bmp file names
# names = [name for name in names if
#          not os.path.exists(join(npz_path, prefix + '_' + name.split('.bmp')[0] + '.npz'))]
# names = [name for name in names if os.path.exists(
#     join(jpg_path, name.split('.bmp')[0] + img_name_suffix))]  # check if corresponding jpg file exists

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

color_to_label = {}
# Set the label for black color (RGB = 0,0,0)
color_to_label[(0, 0, 0)] = 0
label_counter = 1

print(f"Generating labels to {num_labels} labels")
length_names = len(names)
img_count = 0 
for name in names:
    img_count += 1
    gt_pil = np.array(Image.open(join(bmp_path, f"{name}")))  # read bmp image
    gt_pil_int = np.int16(gt_pil)
    gt_pil_thresholded = gt_pil.copy()
    gt_pil_thresholded[gt_pil_thresholded <= lower_threshold] = 0
    gt_pil_thresholded[gt_pil_thresholded > upper_threshold] = 255
    counts = np.bincount(gt_pil_thresholded.flatten())
    rare_labels = np.where(counts < min_count)[0]
    _, color_to_label, label_counter = color_to_label_mapping(gt_pil_thresholded, color_to_label, label_counter)
    for label in rare_labels:
        gt_pil_thresholded[gt_pil_thresholded == label] = 0
    unique_colors = np.unique(gt_pil_thresholded.reshape(-1, gt_pil_thresholded.shape[-1]), axis=0)
    print('\r', f"({img_count} / {length_names})Reading {name} with {len(unique_colors)} labels, {label_counter} labels already generated", end="", flush=True)
    if label_counter == num_labels:
        break
print(f"Label dict created with {label_counter} labels")

for key, label in list(color_to_label.items()):

    save_path_label = join(npz_path, f'{label}')
    os.makedirs(save_path_label, exist_ok=True)
    print("\nProcessing label: ", label)
    for name in tqdm(train_names):
        image_name = name.split('.bmp')[0]
        bmp_name = name

        gt_pil = Image.open(join(bmp_path, f"{bmp_name}.bmp"))  # read bmp image
        # print("gt_pil: ", gt_pil)
        gt_data = np.array(gt_pil)
        # print("gt_data: ", gt_data, "sum: ", np.sum(gt_data), gt_data[gt_data!=0])
        gt_data_labeled, _, _ = color_to_label_mapping(gt_data, color_to_label, label_counter)
        gt_data = np.int16(gt_data_labeled)
        try:
            imgs, gts, img_embeddings = preprocess_ct(gt_data, jpg_path, bmp_name, label, image_size, sam_model, device)

            if len(imgs) >= 1:
                save_path_train = join(save_path_label, 'train')
                os.makedirs(save_path_train, exist_ok=True)
                imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
                gts = np.stack(gts, axis=0)  # (n, 256, 256)
                img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
                np.savez_compressed(join(save_path_train, bmp_name.split('.bmp')[0] + '.npz'), gts=gts, img_embeddings=img_embeddings)
                # save an example image for sanity check
                idx = np.random.randint(0, imgs.shape[0])
                img_idx = imgs[idx, :, :, :]
                gt_idx = gts[idx, :, :]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
        except:
            print("\r", "Failed in processing", end="", flush=True)

        # save to npz file
    for name in tqdm(test_names):
        image_name = name.split('.bmp')[0]
        bmp_name = name

        gt_pil = Image.open(join(bmp_path, f"{bmp_name}.bmp"))  # read bmp image
        # print("gt_pil: ", gt_pil)
        gt_data = np.array(gt_pil)
        # print("gt_data: ", gt_data, "sum: ", np.sum(gt_data), gt_data[gt_data!=0])
        gt_data_labeled, _, _ = color_to_label_mapping(gt_data, color_to_label, label_counter)
        gt_data = np.int16(gt_data_labeled)
        try:
            imgs, gts, img_embeddings = preprocess_ct(gt_data, jpg_path, bmp_name, label, image_size, sam_model, device)
            if len(imgs) >= 1:
                save_path_test = join(save_path_label, 'test')
                os.makedirs(save_path_test, exist_ok=True)
                imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
                gts = np.stack(gts, axis=0)  # (n, 256, 256)
                img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
                np.savez_compressed(join(save_path_test, bmp_name.split('.bmp')[0] + '.npz'), imgs=imgs,
                                    gts=gts)
                # save an example image for sanity check
                idx = np.random.randint(0, imgs.shape[0])
                img_idx = imgs[idx, :, :, :]
                gt_idx = gts[idx, :, :]
                bd = segmentation.find_boundaries(gt_idx, mode='inner')
                img_idx[bd, :] = [255, 0, 0]
        except:
            print("\r", "Failed in processing", end="", flush=True)
