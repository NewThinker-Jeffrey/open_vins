#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import subprocess

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    dataset = sys.argv[1]
    my_dir = os.path.dirname(os.path.abspath(__file__))
    publish_img_script = os.path.join(my_dir, "publish_heisenberg_imgs.py")
    publish_processes = dict()

    bag_without_imgs = os.path.join(dataset, "vio_gt.bag")
    cmd = "ros2 bag play {} -r 2.0".format(bag_without_imgs)
    publish_processes["imu_data"] = subprocess.Popen(cmd, shell=True)

    cam_front_dir = os.path.join(dataset, "cam_front")
    cmd = "{} {}".format(publish_img_script, cam_front_dir)
    publish_processes["cam_front"] = subprocess.Popen(cmd, shell=True)

    topics = ["/hps/robot/walkingMotorStatus",
              "/hps/robot/imu",
              "/hps/robot/gpsStatus",
              "/hps/robot/cam_front",
              ]
    output_bag = os.path.join(dataset, "vio.bag")
    cmd = "ros2 bag record -o {} {}".format(output_bag, " ".join(topics))
    record_process = subprocess.Popen(cmd, shell=True)

    finished_processes = set()
    while len(finished_processes) < len(publish_processes):
        for process_name, process in publish_processes.items():
            if process_name not in finished_processes:
                if subprocess.Popen.poll(process) is not None:
                    print("{} finished.".format(process_name))
                    finished_processes.add(process_name)
        time.sleep(3)


    record_process.send_signal(SIGINT)
    while subprocess.Popen.poll(process) is None:
        print("waiting for recording process to exit ...")
        time.sleep(3)
    print("finished.")




