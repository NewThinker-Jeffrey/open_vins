#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

large_gap_thr = 0.05

def load_imu_times(imu_file):
    imu_times = list()
    with open(imu_file, "r") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("#"):
                continue
            cols = line.split(",")
            assert(len(cols) == 7)
            imu_times.append(int(cols[0]))
    return imu_times

def get_time_gaps(imu_times):
    # if not imu_times:
    #     return list()

    time_gaps = [0.0, ]  # to ensure len(time_gaps) = len(imu_times)
    for i in range(len(imu_times) - 1):
        gap = (imu_times[i+1] - imu_times[i]) * 1e-9
        if gap > large_gap_thr:
            print("large_gap_detected: {} - {} = {}".format(imu_times[i+1], imu_times[i], imu_times[i+1] - imu_times[i]))
        time_gaps.append(gap)

    return time_gaps

def main():
    imu_file = sys.argv[1]
    if os.path.isdir(imu_file):
        imu_file = os.path.join(imu_file, "imu0/data.csv")
    imu_times = load_imu_times(imu_file)
    print("start time is {}".format(imu_times[0] * 1e-9))
    time_gaps = get_time_gaps(imu_times)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_gaps)
    plt.show()



if __name__ == "__main__":
    main()
