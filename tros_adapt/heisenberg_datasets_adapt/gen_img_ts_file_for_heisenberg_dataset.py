#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import subprocess

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    dataset = sys.argv[1]

    img_folder = os.path.join(dataset, "cam_front")
    img_ts_list = list()
    img_ts_file = img_folder + ".timestamps.txt"
    for img_file in os.listdir(img_folder):
        tmp = img_file.split(".")
        if len(tmp) == 2 and tmp[1] == "jpg":
            img_ts_list.append(tmp[0])
    img_ts_list.sort()
    with open(img_ts_file, "w") as stream:
        stream.write("\n".join(img_ts_list))
        stream.write("\n")
    print("add {}".format(img_ts_file))


