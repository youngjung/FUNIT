"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default='pretrained/animal149_gen.pt')
parser.add_argument('--dname_ref',
                    type=str,
                    default='images/ref')
parser.add_argument('--dname_src',
                    type=str,
                    default='images/src')
parser.add_argument('--dname_dest',
                    type=str,
                    default='images/qual')
opts = parser.parse_args()
cudnn.benchmark = True
opts.vis = True
config = get_config(opts.config)
config['batch_size'] = 1
config['gpus'] = 1

trainer = Trainer(config)
trainer.cuda()
trainer.load_ckpt(opts.ckpt)
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((128, 128))] + transform_list
transform = transforms.Compose(transform_list)

os.makedirs(opts.dname_dest, exist_ok=True)

fnames_ref = os.listdir(opts.dname_ref)
stylecodes = []
for fname_ref in tqdm(fnames_ref, desc='computing style codes'):
    fname_full = os.path.join(opts.dname_ref, fname_ref)
    img = Image.open(fname_full).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        stylecode = trainer.model.compute_k_style(img_tensor, 1)
        stylecodes.append(stylecode)

fnames_src = os.listdir(opts.dname_src)
for fname_src in tqdm(fnames_src, desc='translating'):
    fname_full = os.path.join(opts.dname_src, fname_src)
    img = Image.open(fname_full).convert('RGB')
    src = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        for i_ref, stylecode in enumerate(stylecodes):
            output_image = trainer.model.translate_simple(src, stylecode)
            image = output_image.detach().cpu().squeeze().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = ((image + 1) * 0.5 * 255.0)
            fname_ref = fnames_ref[i_ref]
            fname_dest_full = os.path.join(opts.dname_dest, '%s_%s.jpg' % (fname_src.strip('.png'), fname_ref.strip('.png')))
            output_img = Image.fromarray(np.uint8(image))
            output_img.save(fname_dest_full, 'JPEG', quality=99)
