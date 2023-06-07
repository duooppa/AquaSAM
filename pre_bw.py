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

# set up the parser
parser = argparse.ArgumentParser(description='preprocess CT images')
parser.add_argument('-i', '--jpg_path', type=str, default='./data/suim/train_val/images', help='path to the jpg images')  # Changed from 'nii_path' to 'jpg_path'
parser.add_argument('-gt', '--bmp_path', type=str, default='./data/suim/train_val/masks', help='path to the bmp images')  # Changed from 'gt_path' to 'bmp_path'
parser.add_argument('-o', '--npz_path', type=str, default='./data/suim/train_val/output', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--modality', type=str, default='CT', help='modality')
parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy')
parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='image name suffix')
parser.add_argument('--label_id', type=int, default=255, help='label id')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')


# Other unchanged arguments...

# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()
print("Preprocessing start")
names = sorted(os.listdir(args.jpg_path))  # Changed from 'gt_path' to 'bmp_path'
names = [os.path.splitext(f)[0] for f in names]

# names = [name for name in names if not os.path.exists(join(args.npz_path, prefix + '_' + name.split('.bmp')[0]+'.npz'))]  # Changed '.nii.gz' to '.bmp'
# names = [name for name in names if os.path.exists(join(args.jpg_path, name.split('.bmp')[0] + args.img_name_suffix))]  # Changed 'nii_path' to 'jpg_path' and '.nii.gz' to '.bmp'

# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names[:int(len(names)*0.8)])
test_names = sorted(names[int(len(names)*0.8):])

# def preprocessing function
def preprocess_ct(bmp_path, jpg_path, bmp_name, image_name, label_id, image_size, sam_model, device):
    gt_pil = Image.open(join(bmp_path, f"{bmp_name}.bmp"))  # read bmp image
    # print("gt_pil: ", gt_pil)
    gt_data = np.array(gt_pil)
    # print("gt_data: ", gt_data, "sum: ", np.sum(gt_data), gt_data[gt_data!=0])
    gt_data = np.uint8(gt_data==label_id)
    # print("gt_data: ", gt_data, "sum: ", np.sum(gt_data))
    if np.sum(gt_data)>10:
        imgs = []
        gts =  []
        img_embeddings = []
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
        img_pil = Image.open(join(jpg_path, f"{bmp_name}.jpg"))  # read jpg image
        image_data = np.array(img_pil)
        # print("pil data: ", image_data)
        image_data_pre = np.clip(image_data, 0, 255)  # bmp range
        # print("image_data_pre: ", image_data_pre, type(image_data_pre[0, 0, 0]))
        # print("np.sum(gt_data)>10")
        # resize to image_size
        gt_slice_i = transform.resize(gt_data, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        if np.sum(gt_slice_i)>100:
            # print("np.sum(gt_slice_i)>100:")
            img_slice_i = transform.resize(image_data_pre, (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1)) if len(img_slice_i.shape) < 3 else img_slice_i.astype(np.uint8)
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
            imgs.append(img_slice_i)
            assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
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

        if sam_model is not None:
            # print("imgs: ", imgs, "gts: ", gts, "img_embeddings: ", img_embeddings)
            return imgs, gts, img_embeddings
        else:
            # print("imgs: ", imgs, "gts: ", gts)
            return imgs, gts
    else:
        print("Cannot return for size")


names = sorted(os.listdir(args.bmp_path))  # read bmp file names
# names = [name for name in names if
#          not os.path.exists(join(args.npz_path, prefix + '_' + name.split('.bmp')[0] + '.npz'))]
# names = [name for name in names if os.path.exists(
#     join(args.jpg_path, name.split('.bmp')[0] + args.img_name_suffix))]  # check if corresponding jpg file exists

save_path_train = join(args.npz_path, 'train')
save_path_test = join(args.npz_path, 'test')
os.makedirs(save_path_train, exist_ok=True)
os.makedirs(save_path_test, exist_ok=True)

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

# save training data
for name in tqdm(train_names):
    image_name = name.split('.bmp')[0] + args.img_name_suffix  # Changed from '.nii.gz' to '.bmp'
    bmp_name = name
    # print("path: ", args.bmp_path, "jpg: ", args.jpg_path, "bmp name: ", bmp_name, "img name:", image_name, "label: ", args.label_id, "image_size: ",
    #                                           args.image_size, "model: ", sam_model, "device: ",  args.device)
    imgs, gts, img_embeddings = preprocess_ct(args.bmp_path, args.jpg_path, bmp_name, image_name, args.label_id,
                                              args.image_size, sam_model, args.device)

    # save to npz file
    if len(imgs) >= 1:
        imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0)  # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
        np.savez_compressed(join(save_path_train, bmp_name.split('.bmp')[0] + '.npz'), imgs=imgs, gts=gts,
                            img_embeddings=img_embeddings)
        # save an example image for sanity check
        idx = np.random.randint(0, imgs.shape[0])
        img_idx = imgs[idx, :, :, :]
        gt_idx = gts[idx, :, :]
        bd = segmentation.find_boundaries(gt_idx, mode='inner')
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(join(save_path_train, bmp_name.split('.bmp')[0] + '.png'), img_idx,
                  check_contrast=False)  # save example images

# save testing data
for name in tqdm(test_names):
    image_name = name.split('.bmp')[0] + args.img_name_suffix
    bmp_name = name
    imgs, gts, img_embeddings = preprocess_ct(args.bmp_path, args.jpg_path, bmp_name, image_name, args.label_id,
                                              args.image_size, sam_model, args.device)

    # save to npz file
    if len(imgs) >= 1:
        imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0)  # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
        np.savez_compressed(join(save_path_test, bmp_name.split('.bmp')[0] + '.npz'), imgs=imgs, gts=gts,
                            img_embeddings=img_embeddings)
        # save an example image for sanity check
        idx = np.random.randint(0, imgs.shape[0])
        img_idx = imgs[idx, :, :, :]
        gt_idx = gts[idx, :, :]
        bd = segmentation.find_boundaries(gt_idx, mode='inner')
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(join(save_path_test, bmp_name.split('.bmp')[0] + '.png'), img_idx,
                  check_contrast=False)  # save example images


