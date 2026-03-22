import torch
#import dehazing_model
import dehazing_model_Low_Enhangcement
from pathlib import Path
from dataloader import MyData_Test
import piq
import os
import sys
from pathlib import Path
import Feature_Processing
from matplotlib.pyplot import imsave

sys.path.append(str(Path(__file__).resolve().parent.parent))
# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

#Select size of the input images. Default is 1024x1024
Image_size=(1024,1024)
# Image_size=(512,512)

# Pick the model weights based on the dataset it was trained on. Comment out the other model paths
#model_path='../models/RD_dehazing_model_final.pth'  #For model trained on RESIDE dataset
#model_path='../models/OH-dehazing_model_final.pth'  #For model trained on O-HAZE dataset
#model_path='../models/NH_dehazing_model_final.pth'  #For model trained on NH-HAZE dataset
#model_path='../models/DH_dehazing_model_final.pth'  #For model trained on D-HAZE dataset
#model_path='../models/dehazing_model_final.pth'
#model_path='../models/dehazing_model_best_3_channel.pth'
#model_path='../models/dehazing_model_best_3_channel_Consistencyloss.pth'
#model_path='../models/dehazing_model_best_3_channel_Consistencyloss_架构修正1.pth'
#model_path = '../models/dehazing_model_best_3_channel_Consistencyloss_架构修正完全.pth'
#model_path = '../models/dehazing_model_best_3_channel_Consistencyloss_架构修正完全1.pth'
#model_path='../models/dehazing_model_final_Low_enhance_100.pth'
#model_path = '../models/修改后的输入集/dehazing_model_best_3_channel_Consistencyloss_架构修正完全1.pth'
#model_path = '../models/修改后的输入集/dehazing_model_best_3_channel_Consistencyloss_优化一二完成.pth'
#model_path = '../models/消融2/baseline.pth'
model_path = '../models/Ohaze/1.pth'
#model_path = '../models/DenseHaze/5patchsize_512_test_change.pth'
#model_path = '../models/NHHaze/4_752div_1024input.pth'
#model_path = '../models/NHHaze/dehazing_model_50.pth'
# model_path = '../models/SOTS/3_noNormalize.pth'

input_path = script_dir  / '..' / 'test_input'
input_path = os.path.normpath(input_path) 


def process_images(model, test_data, storage_path, Img_size=(1024,1024)):
    test_data=MyData_Test(test_data, Img_size)
    test_data_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    
    ssim_values=[]
    psnr_values=[]
    
    i=1
    # Loop over the test data
    for hazy_img, gt_img in test_data_dataloader:

        # Input the hazy image to the model
        with torch.no_grad():
            hazy_img=hazy_img.to('cuda')
            gt_img=gt_img.to('cuda')
            hazy_img=Feature_Processing.normalize(hazy_img)
            dehazed_img = model(hazy_img)[-1]
        
        hazy_img=Feature_Processing.denormalize(hazy_img)
        dehazed_img=Feature_Processing.denormalize(dehazed_img)
        # Calculate the MSE, PSNR, and SSIM values for the dehazed image
        ssim_value = piq.ssim(gt_img, dehazed_img).item()
        psnr_value = piq.psnr(gt_img, dehazed_img, data_range=1.).item()
        
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        
        dehazed_img_np = dehazed_img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        imsave(storage_path+ '/dehazed_{}.png'.format(i), dehazed_img_np)
        print(f'Image {i} processed')
        i+=1
    
    average_ssim= sum(ssim_values)/len(ssim_values)
    average_psnr= sum(psnr_values)/len(psnr_values)
    with open(storage_path+'results.txt', 'w') as f:
        f.write(f'Average SSIM: {average_ssim}\n')
        f.write(f'Average PSNR: {average_psnr}\n')



#model = dehazing_model.Dehazing_Model()  # Replace with your model class
model = dehazing_model_Low_Enhangcement.Dehazing_Model()


folder_path="./output/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

model.load_state_dict(torch.load(model_path))  # Replace with your model path
model.to('cuda')
# Set the model to evaluation mode
model.eval()

process_images(model, input_path, folder_path, Image_size)
print("All Images Processed")