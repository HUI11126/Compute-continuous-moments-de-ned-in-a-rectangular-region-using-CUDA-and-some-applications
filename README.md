I am rewriting the code in paper "A new fast algorithm to compute continuous moments deﬁned in a rectangular region" using CUDA, and try to find some applications.<br>  
The main idea of image moments is to treat the image as a discrete two-dimensional function, and find a group of orthogonal basis functions to approach the discrete two-dimensional function. There are two important innovations: 1.Apply the nearest neighbor interpolation to calculate image moments accurately;   2.Apply matrix multiplication to accelerate the calculation.<br>  
We can go three steps further:
1. Replace the nearest neighbor interpolation with bilinear interpolation in formula (31) to calculate the integration more accurately;<br> 
2. Integrate the formula (47) and (48), where the rows of matrix B and C are the orders of Legendre polynomials;<br> 
3. When reconstructing the image, change the formula (47) into subdiagonal triangular matrix, and apply matrix multiplication. The formula (47) would be like
   $$\left[\begin{matrix}
   \lambda_0,_0 & \lambda_0,_1 & ... & \lambda_0,_{n-1}  & \lambda_0,_n \\
   \lambda_1,_0 & \lambda_1,_1 & ... & \lambda_1,_{n-1}  &   0           \\
   \lambda_2,_0 & \lambda_2,_1 & ... &        0          &   0           \\
   .            &       .      & ... &        .          &       .        \\
   .            &       .      & ... &        .          &       .        \\
   \lambda_m,_0 &       0      & ... &        0          &       0         \\
   \end{matrix}\right].$$

# 1.Some image related applications:
1. Image reconstruction. The MSE is 1.8 and PSNR is 45.5<br>
`bash run.sh`<br>
./test_images/lenna-1024_11_1024_recons.tif is the image reconstructions result.
![image](./test_images/lenna-1024_11_1024_recons.tif)<br>
Run `nvprof ./build/rectangular` to see the kernel time.<br> 
    ```
    MSE: 1.80595
    PSNR: 45.5637
    ==760574== Profiling application: ./build/rectangular
    ==760574== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    GPU activities:   88.86%  78.241ms         2  39.121ms  6.6041ms  71.637ms  volta_sgemm_128x128_nn
                        7.42%  6.5349ms         1  6.5349ms  6.5349ms  6.5349ms  inter_liner_k(float*, unsigned char*, int, int, int, int, float)
                        1.62%  1.4307ms         2  715.35us  714.99us  715.72us  volta_sgemm_128x64_nn
                        0.84%  736.59us         4  184.15us  72.894us  297.53us  p_polynomial(float*, int, float, float)
                        0.75%  664.08us         2  332.04us  55.070us  609.01us  matrix_transpose(float*, float const *, int, int)
                        0.21%  182.08us         1  182.08us  182.08us  182.08us  [CUDA memcpy DtoH]
                        0.19%  167.64us         5  33.528us  1.1520us  162.78us  [CUDA memcpy HtoD]
                        0.05%  46.591us         1  46.591us  46.591us  46.591us  recon_img_f2u(unsigned char*, float*, int, int)
                        0.05%  42.943us         1  42.943us  42.943us  42.943us  multiply_coff(float*, float*, int, int)
                        0.00%  1.8560us         2     928ns     832ns  1.0240us  [CUDA memset]
    ```

// TODO<br>
2. Extract image edge information. According to the Cauchy Convergence Theorem, for any ε>0, there exists N, such that M1,M2>N, abs(fM1(x,y)-fM2(x,y))<ε. Assume M1>M2, fM1-M2(x,y) contains very small information of the original image, which is the details or edges of image. Simon Liao also mentioned that in his other papers.<br>
3. Calaulate image gradient in X and Y direction. The basis functions is uniformly continuous, we can exchange the order of integral and summation in formula (6) and (7), calculate the partial derivative in X and Y direction.<br>
4. Image resize. When restructing the image, divide the [-1, 1] into different pieces to get a bigger or smaller image.<br>  

# 2.Point cloud related applications:
Point cloud is a discrete three-dimensional function, we can get similar results as image.<br> 
1. point cloud compress.<br>
2. Extract point cloud edge information.<br>
3. Calculate the gradient in X, Y and Z direction of each point. This may be important in self-driving. For example, we can detects obstacles, uphills, downhills, speed bumps, curbs, puddles, etc. to get a drivable area by calculating the gradient in Z direction.<br>
4. Make point cloud dense or sparse. We can get the point values in fixed positions, and turn a frame of messy point clouds into a regular sparse matrix.<br>  
