import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
from dataloader import MyData
#from dehazing_model import Dehazing_Model
from dehazing_model_Low_Enhangcement import Dehazing_Model
import torch
from tqdm import tqdm
import piq
import matplotlib.pyplot as plt
import torchvision
from torch.cuda.amp import GradScaler, autocast
import time
from torch.fft import fft2, ifft2
import torch.nn.functional as F
import Feature_Processing
import Feature_Processing
import DWT_Block
from DWT_Block import dwt_init

BATCH_SIZE_TRAIN = 8
LEARNING_RATE = 0.0005
NUM_EPOCHS = 60
IMAGE_SIZE=(512,512)
INPUT_SIZE=IMAGE_SIZE
beta1=0.5
beta2=0.999
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 训练开始前，初始化最优指标和权重保存路径
best_val_psnr = float(0)  # 初始设为无穷大，确保首次损失能触发保存
best_model_path = '../models/OHaze/1.pth'


# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

# Create a path to the Training directory
training_data_path = script_dir  / '..' / 'train_input'  / 'Train' 
validation_patch_data_path = script_dir / '..' / 'train_input' / 'Validation'
validation_original_data_path = script_dir / '..' / 'train_input' / 'Validation_Gen'

# Convert the path to a string and normalize it
training_data_path = os.path.normpath(training_data_path)
validation_original_data_path = os.path.normpath(validation_original_data_path)
validation_patch_data_path = os.path.normpath(validation_patch_data_path)

# Pass the path to the module
training_data = MyData(training_data_path, image_size=IMAGE_SIZE)
validation_patch_data = MyData(validation_patch_data_path, image_size=IMAGE_SIZE)
validation_original_data = MyData(validation_original_data_path, image_size=IMAGE_SIZE)


# Create a DataLoader instance
training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
validation_patch_data_loader = torch.utils.data.DataLoader(validation_patch_data, batch_size=1, shuffle=False)
validation_original_data_loader = torch.utils.data.DataLoader(validation_original_data, batch_size=1, shuffle=False)

#Initialize the model
model=Dehazing_Model()
model=model.to(device)

#Initialize the optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(beta1,beta2))
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00005)

folders = [
        './reports',
        './reports/figures',
        './reports/figures/Validation Generation',
        './models'
    ]

for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


# ---------- 单尺度 DWT loss ----------
def loss_DWT_single(pred_img, label_img):

    pred_low,  pred_high  = dwt_init(pred_img)
    label_low, label_high = dwt_init(label_img)

    pred_low  = torch.abs(pred_low)
    label_low = torch.abs(label_low)

    pred_high  = torch.abs(pred_high)
    label_high = torch.abs(label_high)

    l_low  = F.mse_loss(pred_low,  label_low)
    l_high = F.mse_loss(pred_high, label_high)

    return l_low + l_high


# ---------- 单尺度 FFT loss ----------
# def loss_FFT_single(pred_img, label_img):
#     pred_fft = fft2(pred_img)
#     label_fft = fft2(label_img)
#
#     pred_fft_mag = torch.abs(pred_fft)
#     label_fft_mag = torch.abs(label_fft)
#
#     fft_loss = F.l1_loss(pred_fft_mag, label_fft_mag)
#     return fft_loss
def loss_FFT_single(pred_img, label_img):
    pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
    label_fft = torch.fft.fft2(label_img, dim=(-2, -1))

    pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
    label_fft = torch.stack([label_fft.real, label_fft.imag], dim=-1)

    return F.l1_loss(pred_fft, label_fft)

# ---------- 总 loss，多尺度 ----------
# def loss_fn(pred_list, target):
#
#     # 1/4 尺度
#     target_s1 = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)
#
#     # 1/2 尺度
#     target_s2 = F.interpolate(target, scale_factor=0.50, mode='bilinear', align_corners=False)
#
#     # 原图
#     target_s3 = target
#
#     loss_content = (
#         F.l1_loss(pred_list[0], target_s1) +
#         F.l1_loss(pred_list[1], target_s2) +
#         F.l1_loss(pred_list[2], target_s3)
#     )
#
#     loss_dwt = (
#         loss_DWT_single(pred_list[0], target_s1) +
#         loss_DWT_single(pred_list[1], target_s2) +
#         loss_DWT_single(pred_list[2], target_s3)
#     )
#
#     loss_fft = (
#         loss_FFT_single(pred_list[0], target_s1) +
#         loss_FFT_single(pred_list[1], target_s2) +
#         loss_FFT_single(pred_list[2], target_s3)
#     )
#
#     return loss_content + 3 * loss_dwt + 0.1 * loss_fft
def loss_fn(pred_list, target):
    """
    pred_list: [1/8, 1/4, 1/2, 1]  或 [1/4, 1/2, 1]
    """

    # --------- GT 多尺度 ---------
    target_s1 = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)
    target_s2 = F.interpolate(target, scale_factor=0.50, mode='bilinear', align_corners=False)
    target_s3 = target

    # --------- Content loss ---------
    loss_content = (
        F.l1_loss(pred_list[-3], target_s1) +
        F.l1_loss(pred_list[-2], target_s2) +
        F.l1_loss(pred_list[-1], target_s3)
    )

    # --------- DWT loss ---------
    loss_dwt = (
        loss_DWT_single(pred_list[-3], target_s1) +
        loss_DWT_single(pred_list[-2], target_s2) +
        loss_DWT_single(pred_list[-1], target_s3)
    )

    # --------- FFT loss ---------
    loss_fft = (
        loss_FFT_single(pred_list[-3], target_s1) +
        loss_FFT_single(pred_list[-2], target_s2) +
        loss_FFT_single(pred_list[-1], target_s3)
    )

    # --------- Consistency loss ---------
    loss_consistency = 0.0

    # 1/8 → 1/4
    if len(pred_list) == 4:
        loss_consistency += F.l1_loss(
            F.interpolate(pred_list[0], scale_factor=2, mode='bilinear', align_corners=False),
            pred_list[1]
        )

    # 1/4 → 1/2
    loss_consistency += F.l1_loss(
        F.interpolate(pred_list[-3], scale_factor=2, mode='bilinear', align_corners=False),
        pred_list[-2]
    )

    # 1/2 → 1
    loss_consistency += F.l1_loss(
        F.interpolate(pred_list[-2], scale_factor=2, mode='bilinear', align_corners=False),
        pred_list[-1]
    )

    # --------- Total loss ---------
    loss = (
        loss_content +
        3.0 * loss_dwt +
        0.1 * loss_fft +
        0.5 * loss_consistency   # 推荐权重 0.3 ~ 1.0
    )

    return loss
    # --------- 消融1 实验1 ---------
    # loss = (
    #     7 * loss_dwt +
    #     1 * loss_fft
    # )
    #
    # return loss
    #--------- 消融1 实验2 ---------
    # loss = (
    #     3.0 * loss_dwt +
    #     0.1 * loss_fft +
    #     0.5 * loss_consistency   # 推荐权重 0.3 ~ 1.0
    # )
    # return loss
#List to store Losses and IQA scores
training_loss=[]
validation_patch_loss=[]
validation_whole_loss=[]
validation_patch_psnr=[]
validation_patch_ssim=[]
validation_patch_mse=[]
validation_whole_psnr=[]
validation_whole_ssim=[]
validation_whole_mse=[]
learning_rate_list=[]
#Training Loop

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
    
    # Training
    model.train()
    batch_loss = 0.0
    
    start_time=time.time()
    
    progress_bar = tqdm(enumerate(training_data_loader), total=len(training_data_loader), desc='Training')
    
    for i, batch in progress_bar:
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = Feature_Processing.normalize(inputs)
        targets = Feature_Processing.normalize(targets)
        # Forward Pass
        #outputs = model(inputs)
        #loss = loss_fn(outputs, targets)
        
        # Backward Pass
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs=model(inputs)
            loss=loss_fn(outputs,targets)
            outputs = outputs[-1]

        batch_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()
        
        
        #loss.backward()
        #optimizer.step()
        
        #batch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_description(f"Training Epoch {epoch+1} - Batch {i+1} - Loss: {loss.item():.4f}")
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    learning_rate_list.append(current_lr)
    
    avg_train_loss=batch_loss/len(training_data_loader)
    training_loss.append(avg_train_loss)
    
    # Validation
    model.eval()
    
    validation_batch_patch_loss = 0.0
    validation_batch_whole_loss = 0.0
    validation_batch_patch_psnr=0.0
    validation_batch_patch_ssim=0.0
    validation_batch_patch_mse=0.0
    validation_batch_whole_psnr=0.0
    validation_batch_whole_ssim=0.0
    validation_batch_whole_mse=0.0
    validation_whole_gen_images=[]
    validation_whole_gt_images=[]
    validation_whole_haze_images=[]
    with torch.no_grad():
        # Validation Patch Calculation
        for i, batch in enumerate(validation_patch_data_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = Feature_Processing.normalize(inputs)
            targets = Feature_Processing.normalize(targets)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            outputs = outputs[-1]

            validation_batch_patch_loss += loss.item()
            
            inputs=Feature_Processing.denormalize(inputs)
            outputs=Feature_Processing.denormalize(outputs)
            targets=Feature_Processing.denormalize(targets)
            #Calculate PSNR, SSIM and MSE
            validation_batch_patch_psnr+=piq.psnr(outputs,targets).item()
            validation_batch_patch_ssim+=piq.ssim(outputs,targets,data_range=1.,reduction='mean').item()
            validation_batch_patch_mse+=torch.nn.functional.mse_loss(outputs,targets).item()
        
        validation_patch_loss.append(validation_batch_patch_loss/len(validation_patch_data_loader))
        validation_patch_psnr.append(validation_batch_patch_psnr/len(validation_patch_data_loader))
        validation_patch_ssim.append(validation_batch_patch_ssim/len(validation_patch_data_loader))
        validation_patch_mse.append(validation_batch_patch_mse/len(validation_patch_data_loader))
        
        for i, batch in enumerate(validation_original_data_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs=Feature_Processing.normalize(inputs)
            targets=Feature_Processing.normalize(targets)
            outputs = model(inputs)

            loss=loss_fn(outputs,targets)
            outputs = outputs[-1]

            validation_batch_whole_loss+=loss.item()
            
            inputs=Feature_Processing.denormalize(inputs)
            outputs=Feature_Processing.denormalize(outputs)
            targets=Feature_Processing.denormalize(targets)
            
            validation_whole_gen_images.append(outputs.squeeze(0))
            validation_whole_gt_images.append(targets.squeeze(0))
            validation_whole_haze_images.append(inputs.squeeze(0))
            
            #Calculate PSNR, SSIM and MSE
            validation_batch_whole_psnr+=piq.psnr(outputs,targets).item()
            validation_batch_whole_ssim+=piq.ssim(outputs,targets,data_range=1.,reduction='mean').item()
            validation_batch_whole_mse+=torch.nn.functional.mse_loss(outputs,targets).item()
        
        validation_whole_loss.append(validation_batch_whole_loss/len(validation_original_data_loader))
        validation_whole_psnr.append(validation_batch_whole_psnr/len(validation_original_data_loader))
        validation_whole_ssim.append(validation_batch_whole_ssim/len(validation_original_data_loader))
        validation_whole_mse.append(validation_batch_whole_mse/len(validation_original_data_loader))
    
    #Saving the Validation Images Generated
    images=torch.cat([
        torch.stack(validation_whole_haze_images),
        torch.stack(validation_whole_gen_images),
        torch.stack(validation_whole_gt_images)
        ],dim=0)
    
    grid = torchvision.utils.make_grid(images, nrow=len(validation_whole_gen_images))
    
    #Convert the grid to a numpy array and transpose the dimensions for displaying
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Display the grid
    plt.figure(figsize=(20, 10))
    plt.imshow(grid)
    plt.axis('off')
    
    # Save the grid to a file
    plt.savefig(f'./reports/figures/Validation Generation/image_grid_{epoch+1}.jpg', dpi=300)
    
    # Close the figure to free up memory
    plt.close()
        
    # Plotting the Losses
    epochs= range(1, len(training_loss)+1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_patch_loss, label='Validation Patch Loss')
    plt.plot(epochs, validation_whole_loss, label='Validation Whole Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the figure to a file
    plt.savefig(f'./reports/figures/loss_plot_epoch.jpg')
    
    # Close the figure to free up memory
    plt.close()
    
    # Plotting the Learning Rate
    epochs= range(1, len(learning_rate_list)+1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, learning_rate_list, label='Learning Rate')
    
    plt.title('Learning Rate Each epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    # Save the figure to a file
    plt.savefig(f'./reports/figures/learning_rate_epoch.jpg')
    
    # Close the figure to free up memory
    plt.close()
    
    #Plotting the PSNR
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, validation_whole_psnr, label='Validation Whole PSNR')
    plt.plot(epochs, validation_patch_psnr, label='Validation Patch PSNR')
    plt.title('Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    # Save the figure to a file
    plt.savefig(f'./reports/figures/psnr_plot_epoch.jpg')
    # Close the figure to free up memory
    plt.close()
    
    
    #Plotting the SSIM
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, validation_whole_ssim, label='Validation Whole SSIM')
    plt.plot(epochs, validation_patch_ssim, label='Validation Patch SSIM')
    plt.title('Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    # Save the figure to a file
    plt.savefig(f'./reports/figures/ssim_plot_epoch.jpg')
    # Close the figure to free up memory
    plt.close()
    
    
    #Plotting the MSE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, validation_whole_mse, label='Validation Whole MSE')
    plt.plot(epochs, validation_patch_mse, label='Validation Patch MSE')
    plt.title('Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    # Save the figure to a file
    plt.savefig(f'./reports/figures/mse_plot_epoch.jpg')
    # Close the figure to free up memory
    plt.close()
    
    end_time=time.time()
    print(f'Epoch:{epoch+1} Training Loss: {avg_train_loss:.4f}, Time Taken for epoch: {end_time-start_time:.2f} seconds')

    # 判断是否为最优模型（以验证整体损失为例，也可换validation_patch_loss）
    current_val_psnr = validation_patch_psnr[-1]  # 当前epoch的验证整体损失
    if current_val_psnr > best_val_psnr:
        best_val_psnr = current_val_psnr  # 更新最优损失
        torch.save(model.state_dict(), best_model_path)  # 保存最优权重
        print(f'Best model updated! Epoch: {epoch + 1}, Best Val PSNR: {best_val_psnr:.4f}')

    if epoch == 10:
        torch.save(model.state_dict(), './models/dehazing_model_10.pth')
    
    
    if epoch==30:
        torch.save(model.state_dict(), './models/dehazing_model_30.pth')
        
    if epoch==50:
        torch.save(model.state_dict(), './models/dehazing_model_50.pth')
        

# Save the trained model
torch.save(model.state_dict(), '../models/dehazing_model_final_Low_enhance_100.pth')