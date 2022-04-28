I am rewriting the code in paper "A new fast algorithm to compute continuous moments deﬁned in a rectangular region" using CUDA, and try to find some applications.<br>  
The main idea of image moments is to treat the image as a discrete two-dimensional function, and find a group of orthogonal basis functions to approach the discrete two-dimensional function. There are two important innovations: 1.Apply the nearest neighbor interpolation to calculate image moments accurately;   2.Apply matrix multiplication to accelerate the calculation.<br>  
We can go two steps further:
1. Replace the nearest neighbor interpolation with bilinear interpolation in formula (31) to calculate the integration more accurately;<br> 
2. Integrate the formula (47) and (48), where the rows of matrix B and C are the orders of Legendre polynomials. Currently, I am blocked in the second step.

# 1.Some image related applications:
1. Image compress. Save the coefficients of orthogonal basis in formula (7), instead of the original image. If the coefficients in formula (7) is accurate, it needs very few orthogonal polynomials to reconstruct the image. The number of coefficients would be much less than the number of pixels in the image.<br>
2. Extract image edge information. According to the Cauchy Convergence Theorem, for any ε>0, there exists N, such that M1,M2>N, abs(fM1(x,y)-fM2(x,y))<ε. Assume M1>M2, fM1-M2(x,y) contains very small information of the original image, which is the details or edges of image. Simon Liao also mentioned that in his other papers.<br>
3. Calaulate image gradient in X and Y direction. The basis functions is uniformly continuous, we can exchange the order of integral and summation in formula (6) and (7), calculate the partial derivative in X and Y direction.<br>
4. Image resize. When restructing the image, divide the [-1, 1] into different pieces to get a bigger or smaller image.<br>  

# 2.Point cloud related applications:
Point cloud is a discrete three-dimensional function, we can get similar results as image.<br> 
1. point cloud compress.<br>
2. Extract point cloud edge information.<br>
3. Calculate the gradient in X, Y and Z direction of each point. This may be important in self-driving. For example, we can detects obstacles, uphills, downhills, speed bumps, curbs, puddles, etc. to get a drivable area by calculating the gradient in Z direction.<br>
4. Make point cloud dense or sparse. We can get the point values in fixed positions, and turn a frame of messy point clouds into a regular sparse matrix.<br>  
