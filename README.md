# HiSilicon_Intelligent-Forestry-UAV-Plant-Protection-System
2024_hisilicon_embedded_competition
概述：
本项目为海思嵌入式大赛的部分相关代码，包括三部分，3516相关代码，3861相关代码和相机标定相关代码
该项目为使用Yolov2对林业害虫进行识别、分类并判断距离，做出喷药决策，并将3516与3861进行通信，完成对药物喷枪控制的功能

使用说明：
先打印项目中的标定纸，使用3516提供的例程store_sample将录制的stream_chn1.h264格式的视频文件，通过安装的FFmpeg将视频截取成图片
ffmpeg –i xxx.mp4 –r 1 –y xxx_%06d.png  # xxx为你视频文件的名字
然后在终端中执行python Camera_Calibration_Distort.py即可完成对3516相机的预标定

3516通过对VI、VPSS、VO、VENC的初始化，加载YOLOV2模型，并加载视频流每帧照片，完成对害虫的识别，拿到害虫的数量信息，通过送入分类网，拿到害虫类别信息，再通过计算检测框中心点坐标到相机光心的距离，拿到害虫深度信息，通过PestDetect_Num_Species_Distance_Flag函数做出喷药决策，并通过串口发送数据包至3861.

3861通过串口获取3516数据包，解析获取害虫数量，种类和距离信息。分别实现对喷药剂量，喷药种类和喷药时机的调控。
其中，3861通过GPIO引脚控制继电器的关闭和导通，从而实现对药泵电机的控制。采用模拟PWM技术控制药泵通电时间，从而实现喷药剂量的控制。
此外，设计了两种喷药种类，通过驱动两个药泵电机实现。
hisignalling_protocol.c为主功能实现文件；app_demo_uart和hal_iot_gpio_ex分别为串口和GPIO驱动文件。



