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
parser.add_argument('-i', '--jpg_path', type=str, default='./data/suim/train_val/images', help='path to the jpg images')  # Changed from 'nii_path' to 'jpg_path'
parser.add_argument('-gt', '--bmp_path', type=str, default='./data/suim/train_val/masks', help='path to the bmp images')  # Changed from 'gt_path' to 'bmp_path'
parser.add_argument('-o', '--npz_path', type=str, default='./data/suim/train_val/output', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--modality', type=str, default='CT', help='modality')
parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy')
parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='image name suffix')
parser.add_argument('--label_id', type=int, default=255, help='label id') # This is used if color_to_label_mapping is not used for a specific label_id
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')


# Other unchanged arguments...

# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--show_plot', action='store_true', help='Show plot during preprocessing')
args = parser.parse_args()
print("Preprocessing start")
# Assuming names are derived from JPG files, but ground truth is from BMP files.
# The script logic seems to iterate based on files in jpg_path and expects corresponding bmp files.
# Let's stick to os.listdir(args.jpg_path) as per the original code for 'names'
names = sorted(os.listdir(args.jpg_path))
names = [os.path.splitext(f)[0] for f in names] # Get base names

# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names[:int(len(names)*0.8)])
test_names = sorted(names[int(len(names)*0.8):])

def color_to_label_mapping(img_np):
    color_to_label = {}
    # Start label_counter from 1 if 0 is reserved for background, or adjust as needed.
    # Original script implies 0 could be a color.
    label_counter = 0 
    label_img = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint16)

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            rgb = tuple(img_np[i, j, :]) # Assuming img_np is HxWx3
            if rgb not in color_to_label:
                color_to_label[rgb] = label_counter
                label_counter += 1
            label_img[i, j] = color_to_label[rgb]
    return label_img, color_to_label

# def preprocessing function
def preprocess_ct(current_bmp_path, current_jpg_path, base_name, current_image_size, sam_model, current_device, show_plot_flag):
    # base_name is filename without extension, e.g., "image1"
    # Construct full paths
    bmp_file_path = join(current_bmp_path, f"{base_name}.bmp")
    jpg_file_path = join(current_jpg_path, f"{base_name}.jpg")

    if not os.path.exists(bmp_file_path):
        print(f"BMP file not found: {bmp_file_path}")
        return [], [], [] # Return empty lists if files are missing
    if not os.path.exists(jpg_file_path):
        print(f"JPG file not found: {jpg_file_path}")
        return [], [], []


    gt_pil = Image.open(bmp_file_path)
    gt_data_rgb = np.array(gt_pil)
    
    # Apply color_to_label_mapping to convert RGB mask to single-channel label mask
    gt_data_labeled, _ = color_to_label_mapping(gt_data_rgb) # gt_data_labeled is HxW
    gt_data = np.uint16(gt_data_labeled) # Ensure it's integer type for processing

    if show_plot_flag:
        # Displaying the labeled mask (gt_data) might be useful
        plt.imshow(gt_data) 
        plt.title(f"Labeled Mask for {base_name}")
        plt.show()

    # The condition np.sum(gt_data)>100 should apply to the mapped labels.
    # If label_id specific filtering is needed, it should be done here.
    # For now, assume any non-zero label after mapping is of interest.
    # Original script had label_id from args, but color_to_label_mapping implies multiple labels.
    # Let's assume we are interested in slices where any mapped label exists significantly.
    
    if np.sum(gt_data > 0) > 100: # Check if there are significant labeled pixels (not just background)
        imgs = []
        gts =  []
        img_embeddings = []
        
        img_pil = Image.open(jpg_file_path)
        image_data = np.array(img_pil)
        image_data_pre = np.clip(image_data, 0, 255)
        
        gt_slice_i = transform.resize(gt_data, (current_image_size, current_image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=False) # anti_aliasing=False for masks
        
        if np.sum(gt_slice_i > 0) > 100: # Check again after resize
            img_slice_i = transform.resize(image_data_pre, (current_image_size, current_image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            img_slice_i = img_slice_i.astype(np.uint8) # Ensure correct type after resize
            if len(img_slice_i.shape) == 2: # Grayscale to RGB
                img_slice_i = np.repeat(img_slice_i[:,:,None], 3, axis=-1)
            
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
            
            imgs.append(img_slice_i)
            gts.append(gt_slice_i) # gt_slice_i is already the labeled mask
            
            if sam_model is not None:
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_slice_i)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(current_device)
                input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:])
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])
        
        return imgs, gts, img_embeddings # Returns potentially empty lists if inner condition not met
    else:
        # print(f"Skipping {base_name}: Not enough labeled pixels in ground truth (sum <= 100).")
        return [], [], [] # Return empty lists

# The script iterates based on `names` derived from `args.jpg_path`.
# `preprocess_ct` expects `base_name` (filename without extension).

save_path_train = join(args.npz_path, 'train')
save_path_test = join(args.npz_path, 'test')
os.makedirs(save_path_train, exist_ok=True)
os.makedirs(save_path_test, exist_ok=True)

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

# save training data
for name in tqdm(train_names): # `name` is base_name from jpg_path items
    imgs, gts, img_embeddings = [], [], [] # Initialize for current iteration
    try:
        # Pass base_name to preprocess_ct. It will construct full bmp and jpg paths.
        imgs, gts, img_embeddings = preprocess_ct(args.bmp_path, args.jpg_path, name, 
                                                  args.image_size, sam_model, args.device, args.show_plot)
    except Exception as e:
        print(f"\nError processing file {name}.jpg (and corresponding .bmp) for train: {e}")
        continue # Skip to the next file

    if imgs and gts: # Check if lists are not empty
        imgs_stack = np.stack(imgs, axis=0)
        gts_stack = np.stack(gts, axis=0)
        
        save_dict = {'gts': gts_stack}
        if img_embeddings: # SAM model was used
            img_embeddings_stack = np.stack(img_embeddings, axis=0)
            save_dict['img_embeddings'] = img_embeddings_stack
        
        np.savez_compressed(join(save_path_train, name + '.npz'), **save_dict)
        
        # Sanity check image saving
        idx = np.random.randint(0, imgs_stack.shape[0])
        img_idx = imgs_stack[idx, :, :, :]
        gt_idx = gts_stack[idx, :, :] # This is the mapped label mask
        # To visualize boundaries on img_idx, gt_idx needs to be binary or specific label chosen
        # For simplicity, let's assume we visualize boundary of any non-zero label
        bd = segmentation.find_boundaries(gt_idx > 0, mode='inner') 
        img_idx_copy = img_idx.copy() # Avoid modifying the original array
        img_idx_copy[bd, :] = [255, 0, 0] # Draw boundary in red
        # Image.fromarray(img_idx_copy).save(join(save_path_train, name + '_debug.png')) # Using PIL
        # Using io.imsave if preferred, ensure check_contrast=False if not needed.
        io.imsave(join(save_path_train, name + '_debug.png'), img_idx_copy, check_contrast=False)

# save testing data
for name in tqdm(test_names): # `name` is base_name
    imgs, gts, img_embeddings = [], [], []
    try:
        imgs, gts, img_embeddings = preprocess_ct(args.bmp_path, args.jpg_path, name,
                                                  args.image_size, sam_model, args.device, args.show_plot)
    except Exception as e:
        print(f"\nError processing file {name}.jpg (and corresponding .bmp) for test: {e}")
        continue

    if imgs and gts: # Check if lists are not empty
        imgs_stack = np.stack(imgs, axis=0)
        gts_stack = np.stack(gts, axis=0)
        
        save_dict = {'gts': gts_stack, 'imgs': imgs_stack} # Test set usually saves images too
        if img_embeddings: # SAM model was used
            img_embeddings_stack = np.stack(img_embeddings, axis=0)
            save_dict['img_embeddings'] = img_embeddings_stack
            
        np.savez_compressed(join(save_path_test, name + '.npz'), **save_dict)
        
        # Sanity check image saving
        idx = np.random.randint(0, imgs_stack.shape[0])
        img_idx = imgs_stack[idx, :, :, :]
        gt_idx = gts_stack[idx, :, :]
        bd = segmentation.find_boundaries(gt_idx > 0, mode='inner')
        img_idx_copy = img_idx.copy()
        img_idx_copy[bd, :] = [255, 0, 0]
        # Image.fromarray(img_idx_copy).save(join(save_path_test, name + '_debug.png'))
        io.imsave(join(save_path_test, name + '_debug.png'), img_idx_copy, check_contrast=False)

print("\nPreprocessing finished.")
