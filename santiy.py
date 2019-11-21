import glob
from utils import predict, load_images, display_images, scale_up
import numpy as np 


names = sorted(glob.glob('/work/NYUv2/nyu_test_rgb/*.png'))
print(names[:5])
inputs = load_images(names[:5])
print(inputs.shape)
print(inputs[1,50:100,50:100,0])
print(inputs[0,50:100,50:100,0])
print(inputs[0,0:50,0:50,0])

print(np.amax(inputs[1,:,:,0]))
print(np.amin(inputs[1,:,:,0]))