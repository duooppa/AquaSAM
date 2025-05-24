# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root, image_size=256):
        self.data_root = data_root
        self.image_size = image_size
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f), allow_pickle = True) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.stack([d['gts'] for d in self.npz_data], axis=0)
        self.img_embeddings = np.stack([d['img_embeddings'] for d in self.npz_data], axis=0)
        print(self.ori_gts.shape, self.img_embeddings.shape)
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        print("img_embed", img_embed.shape)
        img_embed = img_embed.squeeze(0)
        print("img_embed_new", img_embed.shape)
        gt2D = self.ori_gts[index]
        gt2D = gt2D.squeeze(0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))

        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # con(vert img embedding, mask, bounding box to torch tensor
        #print("img_emb", type(torch.tensor(img_embed)))
        #print("gt2D", type(torch.tensor(gt2D)))
        #print("bboxes", type(torch.tensor(bboxes)))

        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:].astype(np.int32)).long(), torch.tensor(bboxes).float()


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--npz_tr_path', type=str, default='data/Npz_files/output/7/train')
parser.add_argument('--task_name', type=str, default='SAM-ViT-B')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM-ViT-B/sam_vit_b_01ec64.pth')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# 
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--show_plot', action='store_true', help="Show plot window during training")
args = parser.parse_args()


# %% set up model for fine-tuning 
device = args.device
model_save_path = join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
sam_model.train()

# Set up the optimizer, hyperparameter tuning will improve performancƒe here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# regress loss for IoU/DSC prediction;
regress_loss = torch.nn.MSELoss(reduction='mean')
#%% train
num_epochs = args.num_epochs
losses = []
best_loss = 1e10
train_dataset = NpzDataset(args.npz_tr_path)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
for epoch in range(num_epochs):
    epoch_loss = 0
    for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        
        # Workaround for a shape mismatch issue in segment_anything.modeling.mask_decoder.py
        # The MaskDecoder internally repeats image_embeddings B times (B=batch_size),
        # making it (B*B, C, H, W). If dense_embeddings are present (B, C, H, W),
        # they need to be similarly expanded to match for element-wise operations.
        current_batch_size = image_embedding.shape[0] # Or whatever variable holds the input image embeddings batch
        if dense_embeddings is not None:
            # Check if dense_embeddings already have the B*B format (e.g. if prompt encoder itself does this sometimes)
            # This is a safeguard, assuming prompt_encoder usually returns (B,C,H,W)
            if dense_embeddings.shape[0] == current_batch_size:
                expanded_dense_embeddings = dense_embeddings.repeat_interleave(current_batch_size, dim=0)
            elif dense_embeddings.shape[0] == current_batch_size * current_batch_size:
                expanded_dense_embeddings = dense_embeddings # Already expanded
            else:
                # This case would be unexpected, raise an error or warning
                print(f"WARNING: dense_embeddings shape {dense_embeddings.shape} is unexpected with batch_size {current_batch_size}. Using as is.")
                expanded_dense_embeddings = dense_embeddings
        else:
            expanded_dense_embeddings = None

        print("image_embeddings", image_embedding.shape)
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=expanded_dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
          )

        # Calculate segmentation loss
        loss_seg = seg_loss(low_res_masks, gt2D.to(device))

        # Calculate target IoU for regression loss
        with torch.no_grad():
            probs = torch.sigmoid(low_res_masks) # (B, num_masks, H, W)
            binary_masks = (probs > 0.5).float() # (B, num_masks, H, W)
            gt2D_float = gt2D.to(device).float() # (B, 1, H, W)

            # Ensure gt2D_float is broadcastable to binary_masks shape for intersection/union calculation
            # This typically means gt2D_float might need to be (B, 1, H, W) if binary_masks is (B, N, H, W)
            # or expanded if necessary. Given DiceCELoss works, gt2D is likely (B, 1, H, W)
            # and broadcasting handles it for element-wise operations with binary_masks (B, N, H, W)

            intersection = torch.sum(binary_masks * gt2D_float, dim=(-2, -1)) # Sum over H, W -> (B, num_masks)
            union = torch.sum(binary_masks, dim=(-2, -1)) + torch.sum(gt2D_float, dim=(-2, -1)) - intersection # Sum over H, W -> (B, num_masks)
            target_iou = intersection / (union + 1e-6) # (B, num_masks)
            # Ensure target_iou has the same shape as iou_predictions (B, num_masks)
            # If iou_predictions is (B, N) and target_iou from above is (B,N), it's fine.

        loss_regress = regress_loss(iou_predictions, target_iou)
        
        total_loss = loss_seg + loss_regress # Simple sum for now

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    epoch_loss /= (step + 1) # ensure step is not zero if dataloader is empty, and average correctly
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Total Loss: {epoch_loss:.4f}, Seg Loss: {loss_seg.item():.4f}, Regress Loss: {loss_regress.item():.4f}')
    # save the model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest_7.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best_7.pth'))

    # %% plot loss
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(join(model_save_path, 'train_loss_7.png'))
    if args.show_plot:
        plt.show()
    plt.close()

