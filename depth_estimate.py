import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, scale_up
from matplotlib import pyplot as plt
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
#parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--input', default='/work/NYUv2/nyu_test_rgb/*.png', type=str, help='Input filename or folder.')  # edit input directory


args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images

names = sorted(glob.glob(args.input))
#print(names[:5])
inputs = load_images(names)
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
#print(inputs.shape)
#print(inputs[1,:,:,0])
#print(np.amax(inputs[1,:,:,0]))
#print(np.amin(inputs[1,:,:,0]))

# Compute results
print("start prediction")
#outputs = predict(model, inputs)
#test = predict(model, inputs, minDepth=10, maxDepth=1000)
#print(test.shape)
outputs = scale_up(2, predict(model, inputs, minDepth=10, maxDepth=1000)[:,:,:,0]) * 10.0
print("about to start printing result")
print(outputs.shape)
print(outputs[0])

outputs = np.float32(outputs)

savefolder = '/work/NYUv2_DE'
np.savez("%s/%s_depth_estimation_densedepth.npz" % (savefolder, "test"), outputs)
print("save successful")

#np.savez("%s/%s_depth_estimation_megadepth.npz" % (savefolder, "test"), test_depth)

#for i in range(outputs.shape[0]):
#    one_sample = outputs[i]
#    print(one_sample.shape)
#    print(one_sample)
#    print(np.amin(one_sample))
#    print(np.amax(one_sample))
#    break


#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
#viz = display_images(outputs.copy(), inputs.copy())
#plt.figure(figsize=(10,5))
#plt.imshow(viz)
#plt.savefig('test.png')
#plt.show()
