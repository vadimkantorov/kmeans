import torch
import matplotlib, matplotlib.pyplot as plt

import kmeans

seed = 42

img = torch.as_tensor(plt.imread('astronaut.jpg').copy())

X = img.unsqueeze(0).movedim(-1, -3) / 255.0

I, C = kmeans.kmeans2d(X, num_iter = 50, K = 3, radius = 5, mask = 'rep', generator = torch.Generator().manual_seed(seed))
I, C = I[0], C[0]

max_num_segments = 1 + int(I.amax())
colormap = torch.as_tensor(matplotlib.colormaps['jet'].resampled(max_num_segments)(range(max_num_segments)))
colormap = colormap[torch.randperm(max_num_segments, generator = torch.Generator().manual_seed(seed))]

plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(colormap[I])
plt.subplot(133)
plt.imshow(C.transpose(-1, -2)[I])
plt.savefig('kmeans.png')
