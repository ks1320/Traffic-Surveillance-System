import os
import glob
import random

path = '/content/drive/MyDrive/Software/data/plate-detection'
count = 0
train_c = {}
valid_c = {}
total_c = {}
txt_count = 0

for root, dirnames, filenames in os.walk(path):
    all_files = []
    flag = False
    for f_name in filenames:
        if f_name.endswith(('.jpg', '.jpeg', '.png')) and count<500:
            flag = True
            count += 1
            all_files.append(os.path.join(root, f_name))

        elif f_name.endswith('.txt'):
            txt_count += 1

    if flag:
        random.shuffle(all_files)
        split = int(len(all_files)*.8)
        train = all_files[:split]
        valid = all_files[split:]

        total_c[all_files[0].split('/')[-2]] = len(all_files)
        train_c[train[0].split('/')[-2]] = len(train)
        valid_c[valid[0].split('/')[-2]] = len(valid)

        with open("train.txt", "a") as f:
            f.write("\n".join(train))
            f.write("\n")
        with open("valid.txt", "a") as f:
            f.write("\n".join(valid))
            f.write("\n")
        
