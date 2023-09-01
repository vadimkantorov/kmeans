import argparse
import matplotlib, matplotlib.animation, matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import kmeans

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i', default = 'astronaut.jpg')
parser.add_argument('--output-path', '-o', default = 'demo.png')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--radius', type = int, default = 5)
parser.add_argument('--num-iter', type = int, default = 50)
parser.add_argument('--K', type = int, default = 3)
parser.add_argument('--mask', choices = ['bil', 'rep'], default = 'rep')
parser.add_argument('--scale-factor', type = float, default = 0.25)
args = parser.parse_args()

img = F.interpolate(torch.as_tensor(plt.imread(args.input_path).copy()).movedim(-1, -3).unsqueeze(0) / 255.0, scale_factor = args.scale_factor)

args.K = img.shape[-2] * img.shape[-1]

I, C, Cyx, history = kmeans.kmeans2d(img, num_iter = args.num_iter, K = args.K, radius = args.radius, mask = args.mask, generator = torch.Generator().manual_seed(args.seed))

colormap = torch.as_tensor(matplotlib.colormaps['jet'].resampled(args.K)(range(args.K)))
colormap = colormap[torch.randperm(len(colormap), generator = torch.Generator().manual_seed(args.seed))]

plt.figure()
plt.subplot(131)
plt.imshow(img[0].movedim(-3, -1))
plt.scatter(*Cyx[0].flip(0), c = colormap[torch.arange(Cyx.shape[-1])].tolist(), marker = 's', s = 100)
plt.subplot(132)
plt.imshow(colormap[I[0]])
plt.subplot(133)
plt.imshow(C[0].transpose(-1, -2)[I[0]])
plt.savefig(args.output_path)
plt.close()

fig = plt.figure(figsize = (5, 5))
fig.set_tight_layout(True)
def update(i, im = []):
    I, C, Cyx = history[i]
    img = colormap[I[0]]

    if not im:
        im.append(plt.imshow(img, animated = True, aspect = 'auto'))
        plt.axis('off')

    im[0].set_array(img)
    plt.suptitle(f'iter: [{i}]')
    return im
matplotlib.animation.FuncAnimation(fig, update, frames = list(range(len(history))), interval = 1000).save(args.output_path + '.gif', dpi = 80)
plt.close()
