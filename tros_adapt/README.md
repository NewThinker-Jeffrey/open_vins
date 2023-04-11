


创建好工作空间 YOUR_WORKSPACE 后, 此 repo 放到 ```YOUR_WORKSPACE/src/``` 目录下.

```hps_3rdparty``` 需要切到 ```isaac/openvins_arm``` 分支，并手动 install ```./build_up_to.sh hps_3rdparty```；

接着执行 ```./prepare_image_transport.sh``` 准备好依赖包 ```image_transport```;

> 如果跑地平线数据集，还需要执行 ```./prepare_image_transport_plugins.sh``` 准备好依赖包 ```image_transport_plugins```，并手动编译一下 
```./build_up_to.sh image_transport_plugins ``` 或者仅仅 ```./build_up_to.sh compressed_image_transport ```

然后执行 ```./build_up_to.sh ov_msckf```

------

跑 heisenberg 数据集

ros2 launch ov_msckf heisenberg_subscribe.launch.py dataset:=/path/to/your/heisenberg/dataset

------

跑 horizon 数据集

ros2 launch ov_msckf horizon_subscribe.launch.py horizon_bag:=/path/to/your/horizon_bag

---