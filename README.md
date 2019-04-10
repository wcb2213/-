# Lane-line-detection
课题目标：车道线的位置，虚实线检测，根据车辆位置设计碰撞提醒警报。

task1：使用图像阈值的方法均来检测车道线的位置。
-
效果不太理想<rb>
参考代码：https://github.com/ncondo/CarND-Advanced-Lane-Lines

task2： 改用霍夫变换检测车道线并检测虚实线
-
有效区域的canny图需要手动调整<rb>
参考博客：https://blog.csdn.net/weixin_39059031/article/details/82422182

task3： 通过车辆位置信息设计前车预警
-
使用了一个已经训练好的SSD模型得到车辆位置信息<rb>
参考模型：https://github.com/kcg2015/Vehicle-Detection-and-Tracking
