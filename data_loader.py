import glob
import os
import shutil
from new_algo import main_mod

dirt='/home/aj/Desktop/kiritii/database/right/'+'014/'
print(len(dirt))
import cv2
from orthoprojection_method import orth
from new_algo import main_mod
folders = glob.glob('database/left/*')
import numpy as np

model=main_mod()

for folder in folders:

    print(folder)
    key=int(folder.split('/')[-1])

    for f in glob.glob(folder+'/*'):

        if len(glob.glob('/home/aj/Desktop/kiritii/database/right/'+folder.split('/')[-1]+'/*'))!=len(glob.glob(folder+'/*')):
            pass
        else:
            print('database/right/'+ f.split('/')[-2]+'/'+f.split('/')[-1])
            right_input = cv2.resize(cv2.imread('database/right/'+ f.split('/')[-2]+'/'+f.split('/')[-1]), (224, 224))

            right_image = np.reshape(np.array(right_input, dtype=np.float32), [1, 224, 224, 3]) / 255

            right_feat = model.predict(right_image)
            np.save('database/right/'+ f.split('/')[-2]+'/'+f.split('/')[-1].split('.')[0], right_feat)

            left_input = cv2.resize(cv2.imread(f), (224, 224))

            left_image = np.reshape(np.array(left_input, dtype=np.float32), [1, 224, 224, 3]) / 255

            lef_feat = model.predict(left_image)
            np.save(f.split('.')[0], lef_feat)

            orth_proj=orth(key)
            new_mat = orth_proj.reshape(16, 1)#16X1,1X256 oo=16X256
            leftmat = np.matmul(np.array(new_mat), lef_feat)
            rightmat = np.matmul(np.array(new_mat), right_feat)
            fused_feat=rightmat*leftmat
            print(fused_feat.shape)

            if not os.path.exists('/home/aj/Desktop/kiritii/database/final/' + folder.split('/')[-1]):
                os.mkdir('/home/aj/Desktop/kiritii/database/final/' + folder.split('/')[-1])

            np.save('database/final/'+ f.split('/')[-2]+'/'+f.split('/')[-1].split('.')[0], fused_feat)


        # print('database/right/'+ f.split('/')[-2]+'/'+f.split('/')[-1])
        # right_eye='database/right/'+ f.split('/')[-2]+'/'+f.split('/')[-1]
        # left_eye=f