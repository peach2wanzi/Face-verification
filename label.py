#coding:utf-8
import os
def label(train=True):
    if(train):
        data_path = "/home/hui/ministl_faces/train"

        dirs = os.listdir(data_path)
        f = open("./train.txt", 'w')
        cnt = 0
        for d in dirs:
            dpath = os.path.join(data_path, d)
            files = os.listdir(dpath)
            for fname in files:
                fpath = os.path.join(dpath, fname)
                line = fpath + ' ' + str(cnt) + '\n'
                f.write(line)
            cnt += 1
    else:
        data_path = "/home/hui/ministl_faces/test"

        dirs = os.listdir(data_path)
        f = open("./test.txt", 'w')
        cnt = 0
        for d in dirs:
            dpath = os.path.join(data_path, d)
            files = os.listdir(dpath)
            for fname in files:
                fpath = os.path.join(dpath, fname)
                line = fpath + ' ' + str(cnt) + '\n'
                f.write(line)
            cnt += 1

label(True)
label(False)
