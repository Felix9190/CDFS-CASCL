import os
import scipy.io as sio
import numpy as np
from PIL import Image
import hdf5storage

# SA getPseudo
image_file = os.path.join('../../../datasets', 'WHU-Hi-LongKou/WHU_Hi_LongKou.mat')
label_file = os.path.join('../../../datasets', 'WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')

image_data = sio.loadmat(image_file)
label_data = sio.loadmat(label_file)

data_key = image_file.split('/')[-1].split('.')[0]
label_key = label_file.split('/')[-1].split('.')[0]
data_all = image_data[data_key] # dic-> narray
GroundTruth = label_data[label_key]

# img = np.asarray(data_all[:,:,[90,37,20]])
# img = np.asarray(data_all[:,:,[35,14,4]])
# img = np.asarray(data_all[:,:,[65,30,20]]) # 不错
# img = np.asarray(data_all[:,:,[65,35,20]]) # 不错
# img = np.asarray(data_all[:,:,[65,40,20]]) # 偏蓝
# img = np.asarray(data_all[:,:,[65,40,30]]) # 正好吧
img = np.asarray(data_all[:,:,[135,66,16]]) #

img = np.asarray(img)
# Convert hyperspectral data range to rgb data range
img[:,:,0] = img[:,:,0]/np.max(img[:,:,0])*255
img[:,:,1] = img[:,:,1]/np.max(img[:,:,1])*255
img[:,:,2] = img[:,:,2]/np.max(img[:,:,2])*255
img = np.ceil(img)
# convert to PIL image
img = Image.fromarray(np.uint8(img))
img.save("./LK_pseudo.png")





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
