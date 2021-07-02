import math
from typing import Dict, Union

import scipy
import torch
from torch import nn
from lpips import PerceptualLoss
from Evaluation.image_projection import LBFGS
import time
from torchvision import transforms
import torch.nn.functional as F

pretrained_paths = {
    "vgg19_tf": ("http://beacon.corp.adobe.com/beacon/pretrained/tf_vgg19_weights_6ff9fa5a1e5ccd82a73991520502e43b9050b298.pt", "vgg19_tf/"),
}


class VGG(torch.nn.Module):
    def __init__(self, vgg_layers=["block1_conv1", "block1_conv2", "block3_conv2", "block4_conv2"]):
        self.vgg_layers = vgg_layers
        super(VGG, self).__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_dict = {}
        x_dict["block1_conv1"] = F.relu(self.block1_conv1(x))
        x_dict["block1_conv2"] = F.relu(self.block1_conv2(x_dict["block1_conv1"]))
        x_dict["pool1"] = self.pool1(x_dict["block1_conv2"])
        x_dict["block2_conv1"] = F.relu(self.block2_conv1(x_dict["pool1"]))
        x_dict["block2_conv2"] = F.relu(self.block2_conv2(x_dict["block2_conv1"]))
        x_dict["pool2"] = self.pool2(x_dict["block2_conv2"])
        x_dict["block3_conv1"] = F.relu(self.block3_conv1(x_dict["pool2"]))
        x_dict["block3_conv2"] = F.relu(self.block3_conv2(x_dict["block3_conv1"]))
        x_dict["block3_conv3"] = F.relu(self.block3_conv3(x_dict["block3_conv2"]))
        x_dict["block3_conv4"] = F.relu(self.block3_conv4(x_dict["block3_conv3"]))
        x_dict["pool3"] = self.pool3(x_dict["block3_conv4"])
        x_dict["block4_conv1"] = F.relu(self.block4_conv1(x_dict["pool3"]))
        x_dict["block4_conv2"] = F.relu(self.block4_conv2(x_dict["block4_conv1"]))
        x_dict["block4_conv3"] = F.relu(self.block4_conv3(x_dict["block4_conv2"]))
        x_dict["block4_conv4"] = F.relu(self.block4_conv4(x_dict["block4_conv3"]))
        x_dict["pool4"] = self.pool4(x_dict["block4_conv4"])
        x_dict["block5_conv1"] = F.relu(self.block5_conv1(x_dict["pool4"]))
        x_dict["block5_conv2"] = F.relu(self.block5_conv2(x_dict["block5_conv1"]))
        x_dict["block5_conv3"] = F.relu(self.block5_conv3(x_dict["block5_conv2"]))
        x_dict["block5_conv4"] = F.relu(self.block5_conv4(x_dict["block5_conv3"]))
        x_dict["pool5"] = self.pool5(x_dict["block5_conv4"])
        return [x_dict[l] for l in self.vgg_layers]


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self,
                 vgg_layers=["block1_conv1", "block1_conv2", "block3_conv2", "block4_conv2"],
                 perceptual_size=256,
                 device="cuda"):
        super(VGGPerceptualLoss, self).__init__()

        path = download_pretrained_model(*pretrained_paths['vgg19_tf'])
        vgg_weights = torch.load(path, map_location='cpu')
        self.vgg_features = VGG(vgg_layers=vgg_layers)
        self.vgg_features.load_state_dict(vgg_weights)
        self.criterion = torch.nn.MSELoss()
        self.target = None
        self.target_features = None
        self.perceptual_size = perceptual_size

    def forward(self, output, target_kwargs):
        output = torch.clamp(output, -1, 1)
        if output.shape[2] != self.perceptual_size:
            output_vgg = torch.nn.functional.interpolate(output,
                                                         size=self.perceptual_size,
                                                         mode='bilinear',
                                                         align_corners=True)
        output_features = self.vgg_features(output_vgg)

        if self.target_features is None:
            target = target_kwargs['target']
            self.target = torch.clamp(target, -1, 1)
            if target.shape[2] != self.perceptual_size:
                target_vgg = torch.nn.functional.interpolate(target,
                                                             size=self.perceptual_size,
                                                             mode='bilinear',
                                                             align_corners=True)
            with torch.no_grad():
                target_features = self.vgg_features(target_vgg)
            self.target_features = target_features

        vgg_losses = []
        for o, t in zip(output_features, self.target_features):
            vgg_losses.append(self.criterion(o, t))
        mse_loss = self.criterion(output, self.target)
        vgg_loss = sum(vgg_losses)
        loss = mse_loss + vgg_loss
        return loss


def weighted_mse_loss(input, target, weights, weights_sum):
    out = (input - target)**2
    out = out * weights.expand_as(out)
    loss = out.sum() / weights_sum
    return loss


def preprocess_mask(mask, size):
    transform_mask = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    mask = transform_mask(mask)
    mask = mask[0, ...]  # Take the red channel
    assert mask.max() < 1 + 1e-6
    assert -1e-6 < mask.min()
    return mask


class ImageReconstructionLoss(torch.nn.Module):
    def __init__(self, model="net-lin", net="vgg", device='cuda', loss='mse+lpips', pre_cache=True):
        super(ImageReconstructionLoss, self).__init__()
        self.mse = torch.nn.MSELoss().to(device)
        self.loss_type = loss
        if loss == "mse":
            mse_T = 0.0
        elif loss == "mse+lpips+mix":
            mse_T = 0.01
        elif loss == "mse+lpips":
            mse_T = 100
        else:
            raise NotImplementedError('The loss type %s is not implemented' % loss)

        self.mse_T = mse_T
        self.pre_cache = pre_cache
        self.target_cache = None
        self.mask_sum_cache = None

        self.lpips_model = model
        self.lpips_net = net

        if self.mse_T > 0.0:
            tick = time.time()
            self.perceptual = PerceptualLoss(net=self.lpips_net, model=self.lpips_model, use_gpu=True, gpu_ids=[int(device[-1])])  # from LPIPS
            # self.perceptual = VGGPerceptualLoss(vgg_layers=[1, 3], pre_cache=self.pre_cache).to(device)
            self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            self._imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            print("loading time: ", time.time() - tick)

        self.use_lpips = False

    def forward(self, output, targets_kwargs, mse_weight=1, perceptual_weight=1, perceptual_size=256):
        """
        :param output: Image (probably the output of a generator).
        :type output: torch.Tensor
        :param targets_kwargs: Reference image and mask to compare against.
        :type targets_kwargs: dict
        :param mse_weight: Reconstruction error weight.
        :type mse_weight: float
        :param perceptual_weight: Perceptual loss weight.
        :type perceptual_weight: float
        :param perceptual_size: size of image for perceptual model
        :type perceptual size: int
        """
        target = targets_kwargs['target']
        mask = targets_kwargs['mask']

        if self.loss_type == "mse+lpips" and self.lpips_model == "alex":
            perceptual_size = output.shape[2]

        assert output.shape == target.shape, "output {} and target {} do not match".format(output.shape, target.shape)
        if mask is not None:
            assert output.shape[2:] == mask.shape, "output {} and mask {} do not match".format(output.shape[:1], mask.shape[:1])
            if self.mask_sum_cache is None or not self.pre_cache:
                self.mask_sum_cache = mask.sum()  # pre-compute the sum of the mask
            loss = weighted_mse_loss(output, target, mask, self.mask_sum_cache) * mse_weight
        else:
            loss = self.mse(output, target) * mse_weight

        if loss < self.mse_T and not self.use_lpips:
            self.use_lpips = True

        if self.use_lpips:
            output_ = torch.clamp(output, -1., 1.)
            output_ = torch.nn.functional.interpolate(output_, size=perceptual_size, mode='bilinear', align_corners=False)

            if self.target_cache is None or not self.pre_cache:
                print('caculate LPIPS loss at %s resolution' % perceptual_size)
                target_ = torch.clamp(target.detach(), -1., 1.)
                target_ = torch.nn.functional.interpolate(target_, size=perceptual_size, mode='bilinear', align_corners=False)
                self.target_cache = target_

            loss += self.perceptual(output_, self.target_cache, mask).sum()
        return loss


def encode(im, encoder):
    pass


def _adjust_learning_rate(i, iterations, initial_lr, optimizer, rampdown=0.25, rampup=0.05):
    t = i / iterations
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    lr = initial_lr * lr_ramp
    optimizer.param_groups[0]['lr'] = lr


def _print_loss(loss, iteration, iterations, print_iterations):
    if iteration and iteration % print_iterations == 0:
        width = int(math.log10(iterations)) + 1
        iter_str = "[" + str(iteration).rjust(width, " ") + "]"
        loss_str = iter_str + "{:.4f}".format(loss.detach().cpu().numpy())
        print(loss_str)


def optimize(model: torch.nn.Module,
             input_kwargs: Dict,
             targets: torch.Tensor,
             criterion: torch.nn.Module,
             optimizer: Union[torch.optim.Optimizer, LBFGS.FullBatchLBFGS],
             iterations: int,
             print_iterations: int = 10,
             device = None) -> None:
    """
    :param model: generator model to optimize
    :param input_kwargs: dictionary of input arguments to the model
    :param targets:
    :param criterion:
    :param optimizer:
    :param iterations:
    :param print_iterations:
    :return:
    """
    # make sure that all the input variables requires_grad=True
    for key in input_kwargs:
        vars = input_kwargs[key]
        if isinstance(vars, dict):
            for var in vars:
                if torch.is_tensor(var):
                    var.requires_grad = True
        elif torch.is_tensor(vars):
            vars.requires_grad = True

    # thirdparty optimizer
    if isinstance(optimizer, LBFGS.FullBatchLBFGS):
        def closure():
            optimizer.zero_grad()
            outputs = model(**input_kwargs)
            loss = criterion(outputs, targets)
            return loss
        loss = closure()
        loss.backward()
        for i in range(iterations + 1):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, grad, lr, _, _, _, _, _ = optimizer.step(options=options, device=device)
            _print_loss(loss, i, iterations, print_iterations)
    elif isinstance(optimizer, scipy_optim.PyTorchObjective):
        i = [0]

        def callback(x):
            if i[0] % print_iterations == 0:
                with torch.no_grad():
                    outputs = model(**input_kwargs)
                    loss = criterion(outputs, targets)
                    _print_loss(loss, i[0], iterations, print_iterations)
            i[0] += 1
        # import pdb; pdb.set_trace()
        res = scipy.optimize.minimize(optimizer.fun,
                                      optimizer.x0,
                                      method='L-BFGS-B',
                                      callback=callback,
                                      jac=optimizer.jac,
                                      options={'gtol': 1e-6, 'disp': False, 'maxiter': iterations})
        loss = res.fun
        del res
        return loss
    # pytorch optimizers
    elif isinstance(optimizer, torch.optim.Optimizer):
        if isinstance(optimizer, torch.optim.LBFGS):
            for i in range(iterations + 1):
                def closure():
                    optimizer.zero_grad()
                    outputs = model(**input_kwargs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    _print_loss(loss, i, iterations, print_iterations)
                    return loss
                optimizer.step(closure)
        else:  # Adam, SGD, Adagrad, etc.
            initial_lr = optimizer.param_groups[0]['lr']
            for i in range(iterations + 1):
                optimizer.zero_grad()
                _adjust_learning_rate(i, iterations, initial_lr, optimizer)
                outputs = model(**input_kwargs)
                loss = criterion(outputs, targets)
                loss.backward()
                _print_loss(loss, i, iterations, print_iterations)
                optimizer.step()
    else:
        error_str = "Unsupported optimizer. "
        error_str += "Documentation of supported optimizers can be found here:\n"
        error_str += "\nhttps://pytorch.org/docs/stable/optim.html\n"
        raise ValueError(error_str)
    return loss.data.cpu().numpy()
