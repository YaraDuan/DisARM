import pickle
import os, shutil

path = './data/scannet2/'
train_pkl = path + 'scannet_infos_train.pkl'
val_pkl = path + 'scannet_infos_val.pkl'
train_scene_name = 'scene0362_00'
val_scene_name = 'scene0568_00'
train_txt = path + 'meta_data/scannetv2_train.txt'
val_txt = path + 'meta_data/scannetv2_val.txt'

f=open(train_txt, 'w+')
f.write(train_scene_name+'\n')
f.close()

f=open(val_txt, 'w+')
f.write(val_scene_name+'\n')
f.close()

shutil.copyfile(f'./data/scannet/points/{train_scene_name}.bin', f'./data/scannet2/points/{train_scene_name}.bin')
shutil.copyfile(f'./data/scannet/instance_mask/{train_scene_name}.bin', f'./data/scannet2/instance_mask/{train_scene_name}.bin')
shutil.copyfile(f'./data/scannet/semantic_mask/{train_scene_name}.bin', f'./data/scannet2/semantic_mask/{train_scene_name}.bin')

shutil.copyfile(f'./data/scannet/points/{val_scene_name}.bin', f'./data/scannet2/points/{val_scene_name}.bin')
shutil.copyfile(f'./data/scannet/instance_mask/{val_scene_name}.bin', f'./data/scannet2/instance_mask/{val_scene_name}.bin')
shutil.copyfile(f'./data/scannet/semantic_mask/{val_scene_name}.bin', f'./data/scannet2/semantic_mask/{val_scene_name}.bin')


# create new pkl
# train
file = open(path+'scannet_infos_train_bak.pkl', 'rb')
_new_data = []
data = pickle.load(file)
for d in data:
    if train_scene_name in d['pts_path']:
        _new_data.append(d)
file.close()

f = open(train_pkl, 'wb')
pickle.dump(_new_data, f)
f.close()

# val
file = open(path+'scannet_infos_val_bak.pkl', 'rb')
_new_data = []
data = pickle.load(file)
for d in data:
    if val_scene_name in d['pts_path']:
        _new_data.append(d)
file.close()

f = open(val_pkl, 'wb')
pickle.dump(_new_data, f)
f.close()