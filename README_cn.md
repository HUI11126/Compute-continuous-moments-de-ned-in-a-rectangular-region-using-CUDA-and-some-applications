A new fast algorithm to compute continuous moments deﬁned in a rectangular region.<br>
图像矩的主要思想是把图像看成一个离散的二维函数，找到一组正交基逼近图像对应的二维函数。这篇论文主要有两个重要的创新点：1、利用最近邻插值把图像矩计算得更准确；2、利用矩阵乘法加快计算图像矩.<br>
两个改进的地方：<br>
1、公式(31)是采用最近邻插值的方法计算每个像素点的积分，为了把积分计算得更准确，换成双线性插值；<br>
2、结合公式(47)和(48)，把不同阶的多项式B和C分别做成矩阵形式，减少计算量。


# 1、图像相关的应用：
1.1、图像重建。<br>
1.2、提取图像边缘信息。根据级数的柯西收敛准则，对于任意ε>0，存在正数N，使得当M1,M2>N时，abs(fM1(x,y)-fM2(x,y))<ε。假设M1>M2，则fM1-M2(x,y)表示原函数f(x,y)很微小部分，即细节部分。Simon Liao的其他论文中也提到了这个应用。<br>
1.3、计算图像梯度。由于基函数在[-1,1]一致连续，可以交换公式(6)和(7)中积分与求和的顺序，分别计算x和y方向的导数，得到梯度。<br>
1.4、图像放缩。重建图像时，把图像x和y方向的勒让德多项式等分成不同的比例即可。<br> 


# 2、点云相关的应用：
点云是离散的三维函数，可以得到类似于图像的结果。<br>
2.1、点云压缩；2.2、提取点云边缘信息；2.3、计算点云中每个点的x、y、z方向的梯度；2.4、点云稠密化，或稀疏化。<br>
第2.3点在自动驾驶中用处比较大，比如计算点云z轴方向的梯度，可以检测障碍物、上坡、下坡、减速带、马路牙子、水坑等，从而获得可行驶区域。<br>
第2.4点可以获得点云在空间中某些固定位置的点云信息，把一帧杂乱无章的点云变成规则的稀疏矩阵。
