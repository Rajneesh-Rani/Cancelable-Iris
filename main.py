
import glob
import os
import shutil
from new_algo import main_mod


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from scipy.stats import ortho_group

m = ortho_group.rvs(dim=4)


base_path='/home/aj/Desktop/kiritii/database/'

folders = glob.glob('IITD_database/IITD Database/*')
# imagenames_list=[]
# labels=[]
# imagenames__list = []
count=0
for folder in folders:
    print(folder)
    full_n=1

    for f in glob.glob(folder+'/*'):
        print(f.split('/')[-1])
        eye_P=f.split('/')[-1].split('_')[1].split('.')[0]

        # fullsource=f.split('/')[-1].split('_')[0] + '.bmp'
        # fullsource = str(full_n) + '.bmp'
        # print(fullsource)
        # quit()



        if eye_P is 'R':

            if not os.path.exists(base_path+'right/'+folder.split('/')[-1]):

                os.mkdir(base_path+'right/'+folder.split('/')[-1])
            fullsource=str(len(glob.glob(base_path+'right/'+folder.split('/')[-1]+'/*'))+1)+'.bmp'

            shutil.copyfile(f, base_path + 'right/'+folder.split('/')[-1]+'/'+fullsource)
        else:
            if not os.path.exists( base_path +'left/' + folder.split('/')[-1]):
                # os.mkdir('/home/aj/Desktop/kiritii/right/' + folder.split('/')[-1])
                os.mkdir( base_path +'left/' + folder.split('/')[-1])

            fullsource=str(len(glob.glob( base_path +'left/' + folder.split('/')[-1]+'/*'))+1)+'.bmp'

            shutil.copyfile(f,  base_path + 'left/'+folder.split('/')[-1]+'/'+fullsource)
            # full_n=full_n+1


i = 0
for name in os.listdir("database/left"):
    left_eye = "database/right/{}".format(name)
    if not os.path.isdir(left_eye):
        shutil.rmtree("database/left/{}".format(name))

        i += 1
# print(i)




