import argparse
from collections import namedtuple
from math import ceil

import torch
import numpy as np

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module


def setup(args=None, use_cuda=True, checkpoint_path='./FlowNet2_checkpoint.pth'):
    Args = namedtuple('Args', ['fp16', 'rgb_max'])
    args = Args(False, 255) if args is None else args
    net = FlowNet2(args).cuda() if use_cuda else FlowNet2(args)
    pretrained_dict = torch.load(checkpoint_path)
    net.load_state_dict(pretrained_dict["state_dict"])
    return net


def pad(img, factor=64):
    h, w, c = img.shape
    pad_h = ceil(h / factor) * factor
    pad_w = ceil(w / factor) * factor
    padded_img = np.zeros([pad_h, pad_w, c])
    padded_img[:h, :w, :] = img
    return padded_img


def unpad(img, size):
    h, w = size
    return img[:h, :w, :]


def infer(net, img1, img2):
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    padded_img1, padded_img2 = pad(img1), pad(img2)
    images = [padded_img1, padded_img2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()
    result_data = result.data.cpu().numpy().transpose(1, 2, 0)
    result_data = unpad(result_data, (h, w))
    return result_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str)
    parser.add_argument('--img2', type=str)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    net = setup(checkpoint_path=args.ckpt)
    img1 = read_gen(args.img1)
    img2 = read_gen(args.img2)
    result = infer(net, img1, img2)
    breakpoint()
    print(result)
"""
if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("/home/hjj/PycharmProjects/flownet2_pytorch/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("/home/hjj/flownet2-master/data/FlyingChairs_examples/0000007-img0.ppm")
    pim2 = read_gen("/home/hjj/flownet2-master/data/FlyingChairs_examples/0000007-img1.ppm")
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()

    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("/home/hjj/flownet2-master/data/FlyingChairs_examples/0000007-img.flo", data)

"""
