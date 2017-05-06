#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/user/py/tensorflow-tfrecord-img/images/validation/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((50,50), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)
            print(f + '.jpg')
resize()


dirs = [x[0] for x in os.walk(path)]

for i in dirs:
    path = i + '/'
    dirs = os.listdir(path)
    resize()
    print(i)

path = "/home/user/py/tensorflow-tfrecord-img/images/train/"


dirs = [x[0] for x in os.walk(path)]

for i in dirs:
    path = i + '/'
    dirs = os.listdir(path)
    resize()
    print(i)
