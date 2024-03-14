import os
import scipy.io as sio
import numpy as np
from PIL import Image
import hdf5storage

# CH getPseudo
image_file = os.path.join('../../../datasets','Chikusei_raw_mat/HyperspecVNIR_Chikusei_20140729.mat')
label_file = os.path.join('../../../datasets','Chikusei_raw_mat/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat')
image_data = hdf5storage.loadmat(image_file)
label_data = hdf5storage.loadmat(label_file)
data_all = image_data['chikusei'] # data_all:ndarray(2517,2335,128)
label = label_data['GT'][0][0][0] # label:(2517,2335)

img = np.asarray(data_all[:,:,[62,38,27]]) # 按等间隔算出来的45 36 24

img = np.asarray(img)
# Convert hyperspectral data range to rgb data range
img[:,:,0] = img[:,:,0]/np.max(img[:,:,0])*255
img[:,:,1] = img[:,:,1]/np.max(img[:,:,1])*255
img[:,:,2] = img[:,:,2]/np.max(img[:,:,2])*255
img = np.ceil(img)
# convert to PIL image
img = Image.fromarray(np.uint8(img))
img.save("./CH_pseudo.png")

# from scipy import io
# import numpy as np
# from PIL import Image
# # load data
# imgPth = "./Datasets/PaviaU/PaviaU.mat"
# gtPth = './Datasets/PaviaU/PaviaU_gt.mat'
# # read the rgb channels, Only know the rgb channels of PaviaU:[57,34,3].
# # It is difficult to find the rgb channels of other data sets, so I just set them casually.
# img = io.loadmat(imgPth)['paviaU'][:,:,[57,34,3]]
# img = np.asarray(img)
# # Convert hyperspectral data range to rgb data range
# img[:,:,0] = img[:,:,0]/np.max(img[:,:,0])*255
# img[:,:,1] = img[:,:,1]/np.max(img[:,:,1])*255
# img[:,:,2] = img[:,:,2]/np.max(img[:,:,2])*255
# img = np.ceil(img)
# # convert to PIL image
# img = Image.fromarray(np.uint8(img))
# img.save("./PaviaU.png")
