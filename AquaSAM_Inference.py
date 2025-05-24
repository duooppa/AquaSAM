# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import argparse
import traceback

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum
def compute_iou(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_miou(seg_masks, gt_masks):
    miou_scores = []
    for seg_mask, gt_mask in zip(seg_masks, gt_masks):
        iou_scores = []
        for seg_class in np.unique(seg_mask):
            if seg_class == 0:  # Skip background class
                continue
            seg_class_mask = seg_mask == seg_class
            gt_class_mask = gt_mask == seg_class
            iou = compute_iou(gt_class_mask, seg_class_mask)
            iou_scores.append(iou)
        miou = np.mean(iou_scores)
        miou_scores.append(miou)
    return np.mean(miou_scores)


def finetune_model_predict(img_np, box_np, sam_trans, sam_model_tune, device='cuda:1'):
    H, W = img_np.shape[:2]
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to('cuda:1')
    input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        aquasam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        aquasam_seg_prob = torch.sigmoid(aquasam_seg_prob)
        # convert soft mask to hard mask
        aquasam_seg_prob = aquasam_seg_prob.cpu().numpy().squeeze()
        aquasam_seg = (aquasam_seg_prob > 0.5).astype(np.uint8)
    return aquasam_seg

#%% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on AquaSAM')
parser.add_argument('-i', '--data_path', type=str, default='data/Npz_files/output/7', help='path to the data folder')
parser.add_argument('-o', '--seg_path_root', type=str, default='data/Test_AquaSAMBaseSeg_sam_7_test', help='path to the segmentation folder')
parser.add_argument('--seg_png_path', type=str, default='data/sanity_test/Test_AquaSAMBase_png_sam_7_test', help='path to the segmentation folder')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('-chk', '--checkpoint', type=str, default='work_dir/SAM-ViT-B/sam_vit_b_01ec64.pth', help='path to the trained model')
args = parser.parse_args()

#% load AquaSAM model
device = args.device
sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

#npz_folders = sorted(os.listdir(args.data_path))
npz_folders = ['test']
os.makedirs(args.seg_png_path, exist_ok=True)
for npz_folder in npz_folders:
    npz_data_path = join(args.data_path, npz_folder)
    save_path = join(args.seg_path_root, npz_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        npz_files = sorted(os.listdir(npz_data_path))
        dice_scores = []
        iou_scores = []
        for npz_file in tqdm(npz_files):
            try:
                npz = np.load(join(npz_data_path, npz_file))
                ori_imgs = npz['imgs']
                ori_gts = npz['gts']
                sam_segs = []
                sam_bboxes = []
                sam_dice_scores = []
                for img_id, ori_img in enumerate(ori_imgs):
                    # get bounding box from mask
                    gt2D = ori_gts[img_id]
                    y_indices, x_indices = np.where(gt2D > 0)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    # add perturbation to bounding box coordinates
                    H, W = gt2D.shape
                    x_min = max(0, x_min - np.random.randint(0, 20))
                    x_max = min(W, x_max + np.random.randint(0, 20))
                    y_min = max(0, y_min - np.random.randint(0, 20))
                    y_max = min(H, y_max + np.random.randint(0, 20))
                    bbox = np.array([x_min, y_min, x_max, y_max])
                    seg_mask = finetune_model_predict(ori_img, bbox, sam_trans, sam_model_tune, device=device)
                    sam_segs.append(seg_mask)
                    sam_bboxes.append(bbox)
                    # The following 2D Dice score is computed slice-by-slice for debugging and per-slice evaluation.
                    # For a comprehensive 3D volumetric assessment (if processing 3D volumes),
                    # one might aggregate these slice scores or compute a Dice score over the entire 3D volume.
                    sam_dice_scores.append(compute_dice(seg_mask>0, gt2D>0))
                # save npz, including sam_segs, sam_bboxes, sam_dice_scores
                dice_scores.append(sam_dice_scores)
                iou_scores.append(compute_miou(sam_segs, ori_gts))
                np.savez_compressed(join(save_path, npz_file), aquasam_segs=sam_segs, gts=ori_gts, sam_bboxes=sam_bboxes)
                # visualize segmentation results
                img_id = np.random.randint(0, len(ori_imgs))
                # show ground truth and segmentation results in two subplots
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(ori_imgs[img_id])
                show_box(sam_bboxes[img_id], axes[0])
                show_mask(ori_gts[img_id], axes[0])
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                axes[1].imshow(ori_imgs[img_id])
                show_box(sam_bboxes[img_id], axes[1])
                show_mask(sam_segs[img_id], axes[1])
                axes[1].set_title('SAM: DSC={:.3f}'.format(sam_dice_scores[img_id]))
                axes[1].axis('off')
                # save figure
                fig.savefig(join(args.seg_png_path, npz_file.split('.npz')[0]+'.png'))
                # close figure
                plt.close(fig)
            except:
                print('error in {}'.format(npz_file))
    mean_dice_scores = np.mean(dice_scores, axis=0)
    mean_iou_scores = np.mean(iou_scores)
    print("Mean Dice Scores:", mean_dice_scores)
    print("Mean IoU Scores:", mean_iou_scores)