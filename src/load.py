#%%
import numpy as np
from PIL import Image
#%%
dataset = np.load('dataset/npy/ab1.npy')
#%%
print(dataset.shape)
#%%
half = dataset[0:400]
print(half.shape)
#%%
temp =Image.fromarray(half[2])
temp.save('a.png')
