import torch
from torch import nn
from torchvision import utils, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

import sys
from Util.face_parsing.BiSeNet import BiSeNet

from .network_util import Convert_Tensor_To_Image
from pathlib import Path
file_path = Path(__file__).parent

def Get_Parsing_Net(device):
    '''
    Usage:
        Obtain the network for parsing and its preprocess method
    '''
    
    PRETRAINED_FILE = (file_path / '''./face_parsing/pretrained_model/79999_iter.pth''').resolve()
    
    n_classes = 19
    parsing_net = BiSeNet(n_classes=n_classes).to(device)
    pretrained_weight = torch.load(PRETRAINED_FILE, map_location=device)
    parsing_net.load_state_dict(pretrained_weight)
    parsing_net.eval();

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return parsing_net, to_tensor


def Extract_Face_Mask(pil_image, parsing_net, to_tensor, device):
    '''
    Usage:
        Extract the face foreground from an pil image
        
    Args:
        pil_image:   (PIL.Image) a single image
        parsing_net: (nn.Module) the network to parse the face images
        to_tensor:   (torchvision.transforms) the image transformation function
        device:      (str) device to place the networks
    '''
    
    with torch.no_grad():
        image = pil_image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = parsing_net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
    return parsing


def Batch_Img_Parsing(img_tensor, parsing_net, device):
    '''
    Usage:
        Parse the image tensor in a batch format
    
    Args:
        img_tensor:  (torch.Tensor) of the image tensor generated from generator in format of [N, C, H, W]
        parsing_net: (nn.Module) of the deep network for parsing
    '''
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]
    PARSING_SIZE = 512
    
    scale_factor = PARSING_SIZE / img_tensor.shape[-1]
    transformed_tensor = ((img_tensor + 1 ) / 2).clamp(0,1) # Rescale tensor to [0,1]
    transformed_tensor = F.interpolate(transformed_tensor, 
                                       scale_factor=scale_factor, 
                                       mode='bilinear', 
                                       align_corners=False) # Scale to 512
    for i in range(transformed_tensor.shape[1]):
        transformed_tensor[:,i,...] = (transformed_tensor[:,i,...] - CHANNEL_MEAN[i]) / CHANNEL_STD[i]
        
    transformed_tensor = transformed_tensor.to(device)
    with torch.no_grad():
        img_parsing = parsing_net(transformed_tensor)[0]
    
    parsing = img_parsing.argmax(1)
    return parsing

def Get_Masked_Tensor(img_tensor, batch_parsing, device, mask_grad=False):
    '''
    Usage:
        To produce the masked img_tensor in a differentiable way
    
    Args:
        img_tensor:    (torch.Tensor) generated 4D tensor of shape [N,C,H,W]
        batch_parsing: (torch.Tensor) the parsing result from SeNet of shape [N,512,512] (the net fixed the parsing to be 512)
        device:        (str) the device to place the tensor
        mask_grad:     (bool) whether requires gradient to flow to the masked tensor or not
    '''
    PARSING_SIZE = 512
    
    mask = (batch_parsing > 0) * (batch_parsing != 16) 
    mask_float = mask.unsqueeze(0).type(torch.FloatTensor) # Make it to a 4D tensor with float for interpolation
    scale_factor = img_tensor.shape[-1] / PARSING_SIZE
    
    resized_mask = F.interpolate(mask_float, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    resized_mask = (resized_mask.squeeze() > 0.5).type(torch.FloatTensor).to(device)
    
    if mask_grad:
        resized_mask.requires_grad = True
    
    masked_img_tensor = torch.zeros_like(img_tensor).to(device)
    for i in range(img_tensor.shape[0]):
        masked_img_tensor[i] = img_tensor[i] * resized_mask[i]
    
    return masked_img_tensor



def vis_parsing_maps(im, parsing_anno, stride):
    import cv2
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im


def Get_Salt_Pepper_Nosiy_Image(img_tensor, mask, prob):
    '''
    Usage:
        Obtain the salt & pepper noisy image 
    
    Args:
        img_tensor: (torch.Tensor) a single generated image from the model
        mask:       (np.array) of type (bool) indicating the fore-/back-ground of the image
        prob:       (float) the probability of salt and pepper noise to appear in foreground
    '''
    img_size = img_tensor.shape[-1]
    salt_pepper_noise = np.random.randint(low=0, high=2,size=(img_size,img_size)) * 2 - 1 # s/p noise will be -1 and 1
        
    noisy_img = img_tensor.clone()
    for h in range(img_size):
        for w in range(img_size):
            if mask[h,w] == True and (np.random.random() < prob):
                noisy_img[:,:,h,w] = salt_pepper_noise[h,w]
    
    return noisy_img


def Get_Weight_Gradient(noisy_img, img_tensor, generator):
    '''
    Usage:
        Obtain the gradients of all filters' weights in the feed-forward path
    
    Args:
        noisy_img:  (torch.Tensor) of the noisy image
        img_tensor: (torch.Tensor) of the original generated image
        generator:  (nn.Module) of the generator
    '''
    loss = torch.sum(torch.abs(noisy_img - img_tensor))
    loss.backward()
    
    if 'module' in list(generator.state_dict().keys())[0]:
        g_module = generator.module
    else:
        g_module = generator
    
    module_list = [g_module.conv1] + list(g_module.convs) + [g_module.to_rgbs[-1]]
    grad_list = [module.conv.weight.grad for module in module_list]
    
    grad_score_list = [(torch.mean(torch.abs(grad), axis=[0,1,3,4])).cpu().numpy() for grad in grad_list]
    return grad_score_list



def Get_Content_Aware_Pruning_Score(generator, n_sample, batch_size, noise_prob, device, info_print=False):
    '''
    Usage:
        Obtain the content aware network score
    
    Args:
        generator:  (nn.Module) of a generator
        n_sample:   (int) of the number of samples for estimation
        batch_size: (int) of the size of the batch
        noise_prob: (float) the density of salt and pepper noise
        device:     (str) the device to place for the operations
    '''
    
    # image parsing network
    parsing_net, to_tensor = Get_Parsing_Net(device)

    # noise and batch setup
    LATENT_DIM = 512 
    n_batch = n_sample // batch_size
    batch_size_list = [batch_size] * (n_batch - 1) + [batch_size + n_sample % batch_size]
    grad_score_list = []
    
    for (idx,batch) in enumerate(batch_size_list):
        if info_print:
            print('Processing Batch: ' + str(idx))
        noise_z = torch.randn(batch, LATENT_DIM).to(device)
        img_tensor = generator(noise_z=[noise_z])
        img_size = img_tensor.shape[-1]
        
        noisy_img_list = []
        for i in range(batch):
            single_img = img_tensor[i:i+1,...] # retain it as a 4D tensor
            
            # Parse an image and get the mask
            pil_single_img = Convert_Tensor_To_Image(single_img)
            parsing = Extract_Face_Mask(pil_single_img, parsing_net, to_tensor, device)
            mask = (parsing > 0) * (parsing != 16)
            resized_mask = np.array(Image.fromarray(mask).resize((img_size, img_size)))
            
            # Get noisy images
            noisy_img = Get_Salt_Pepper_Nosiy_Image(single_img, resized_mask, noise_prob)
            noisy_img_list.append(noisy_img)
        
        # Compute the gradient
        noisy_img_tensor = torch.cat(noisy_img_list)
        grad_score = Get_Weight_Gradient(noisy_img_tensor, img_tensor, generator)
        grad_score_list.append(grad_score)
        generator.zero_grad()
        
    return grad_score_list
