#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cuda.h>
#include <time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
#include <chrono>

#define K 5
#define ORD 500  // Legendre polynomial的阶数order
#define TILE_WIDTH 32 // block的宽

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
  if(err!=cudaSuccess)
  {
    fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
    std::cin.get();
    exit(EXIT_FAILURE);
  }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// bilinear interpolation, enlarge k times in x and y direction
__global__ void inter_liner_k(float *dataOut, uchar *dataIn, int imgHeight, int imgWidth, int imgHeight_k, int imgWidth_k, float scale)
{   
  int xIdx = threadIdx.x + blockIdx.x * blockDim.x;	
  int yIdx = threadIdx.y + blockIdx.y * blockDim.y;

  if(xIdx < imgWidth_k && yIdx < imgHeight_k)
  {        
    float fx = (float)((xIdx + 0.5f) * scale - 0.5f);
    int sx = floorf(fx);
    fx -= sx;
    sx = min(sx, imgWidth - 1);
    int sx2 = min(sx + 1, imgWidth - 1);
    if(sx < 0)
      sx2 = 0, sx = 0;

    float2 cbufx;
    cbufx.x = 1.f - fx;
    cbufx.y = fx;

    float fy = (float)((yIdx + 0.5f) * scale - 0.5f);
    int sy = floorf(fy);
    fy -= sy;
    sy = min(sy, imgHeight - 1);
    int sy2 = min(sy + 1, imgHeight - 1);
    if(sy < 0)
        sy2 = 0, sy = 0;

    float2 cbufy;
    cbufy.x = 1.f - fy;
    cbufy.y = fy;

    uchar s11 = dataIn[sy * imgWidth + sx];
    uchar s12 = dataIn[sy * imgWidth + sx2];
    uchar s21 = dataIn[sy2 * imgWidth + sx];
    uchar s22 = dataIn[sy2 * imgWidth + sx2];
       
    float h_rst00x, h_rst01x, h_rst00y, h_rst01y, h_rst00z, h_rst01z;
    h_rst00x = s11 * cbufx.x + s12 * cbufx.y;
    h_rst01x = s21 * cbufx.x + s22 * cbufx.y;

    dataOut[yIdx*imgWidth_k + xIdx] = (h_rst00x * cbufy.x + h_rst01x * cbufy.y) / 127.5f - 1.f; 
  }
}

__constant__ float dj[ORD];
__constant__ float j2_1[ORD]; //2 * j -1
__constant__ float j_1[ORD]; // j - 1
// v:勒让德多项式的值； W：把[-1, 1]等分为W份； div：每一份的长度
// 生成的多项式是(Row) * (Col): ORD * W 的矩阵
__global__ void p_polynomial(float *v, const int W, const float div)
{
  int xIdx = threadIdx.x + blockIdx.x * blockDim.x;	
  
  if(xIdx < W)
  {
    v[xIdx] = 1.0f;
    float temp_x = div * xIdx - 1.0f;
    v[xIdx + W] = temp_x;
    float p0 = 1.0f;
    float p1 = temp_x;
    for(int j=2; j<ORD; j++)
    { 
      float temp_v = v[xIdx + j * W] = (j2_1[j] * temp_x * p1 - j_1[j] * p0) * dj[j];
      p0 = p1;
      p1 = temp_v;
    }
  }
}

// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
__global__ void matrix_transpose(float *odata, const float *idata, int matrixWidth, int matrixHeight)
{
  __shared__ float tile[TILE_WIDTH][TILE_WIDTH+1];

  int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int width = matrixWidth;

  if(x<matrixWidth && y<matrixHeight)
    tile[threadIdx.y][threadIdx.x] = idata[y*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_WIDTH + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_WIDTH + threadIdx.y;
  width = matrixHeight;

  if(x<matrixHeight && y<matrixWidth)
    odata[y*width + x] = tile[threadIdx.x][threadIdx.y];        
}

// 根据公式(7)，重建图像
__global__ void img_recons(uchar *recon_img, float *lambda, float *p_in_Xdir, float *p_in_Ydir_trans, int imgHeight, int imgWidth)
{
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Identify the row and column of the recon_img element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float imgValue = 0.f; //在每个phase中累加图像的值
  if((Row<imgHeight) && (Col<imgWidth))
  {
    for(int i=0; i<ORD; ++i)  // row of lambda
    {
      for(int j=0; j<i; ++j) // col of lambda
      { // X和Y是不是反了？
        // imgValue += lambda[(i-j)*ORD + j] * p_in_Ydir_trans[Row*ORD + i-j] * p_in_Xdir[j*imgWidth + Col] * (float)((2*i+1) * (2*j+1)) / (float)(imgHeight * K * imgWidth * K);
        // 在cublas里面，lambda是按照列存放的
        imgValue += lambda[(i-j)*ORD + j] * p_in_Ydir_trans[Row*ORD + i-j] * p_in_Xdir[j*imgWidth + Col] * (float)((2*(i-j)+1) * (2*j+1)) / (float)(imgHeight * K * imgWidth * K);
        // imgValue += lambda[i*ORD + j] * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i];
      }
    }
    recon_img[Row*imgWidth + Col] = (imgValue + 1.f) * 127.5f;
  }
}

int main(void)
{  
  cv::Mat img_ori = cv::imread("lenna-1024.tif", 0); 
  //////////////////////////////////////////////////// 计算 image moments /////////////////////////////////////////////////////
  // 1、计算X和Y方向的勒让德多项式；
  // 2、把原图的每个像素分成k*k个小方格(图像双线性resize)，使得积分计算更准确
  // 3、转置X方向的勒让德多项式，得到C^T
  // 4、分别计算AC^T和B*(AC^T)，得到图像矩lambda
  float scale = 1.f / (float) K;
  const int imgWidth = img_ori.cols;
  const int imgHeight = img_ori.rows;
  const int imgWidth_k = imgWidth * K;
  const int imgHeight_k = imgHeight * K;
  const int total_ph_X = std::ceil((float)imgWidth_k / (float)TILE_WIDTH);
  const int total_ph_Y = std::ceil((float)imgHeight_k / (float)TILE_WIDTH);

  float *p_in_Xdir_k, *p_in_Ydir_k, *p_in_Xdir, *p_in_Ydir; // 勒让德多项式在X和Y方向的值
  SAFE_CALL(cudaMalloc((void**)&p_in_Xdir_k, ORD * imgWidth_k * sizeof(float)), "cudaMalloc p_in_Xdir_k failed");  // 计算image moments
  SAFE_CALL(cudaMalloc((void**)&p_in_Ydir_k, ORD * imgHeight_k * sizeof(float)), "cudaMalloc p_in_Ydir_k failed");  
  SAFE_CALL(cudaMalloc((void**)&p_in_Xdir, ORD * imgWidth * sizeof(float)), "cudaMalloc p_in_Xdir failed");    // 重建图像
  SAFE_CALL(cudaMalloc((void**)&p_in_Ydir, ORD * imgHeight * sizeof(float)), "cudaMalloc p_in_Ydir failed"); 
  dim3 blockDim_P(32, 1, 1);
  dim3 gridDim_p_X_k((imgWidth_k + blockDim_P.x - 1) / blockDim_P.x, 1, 1);
  dim3 gridDim_p_Y_k((imgHeight_k + blockDim_P.x - 1) / blockDim_P.x, 1, 1);
  dim3 gridDim_p_X((imgWidth + blockDim_P.x - 1) / blockDim_P.x, 1, 1);
  dim3 gridDim_p_Y((imgHeight + blockDim_P.x - 1) / blockDim_P.x, 1, 1);
  
  cudaEvent_t start, stop;
  float runtime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  float *temp_dj = new float[ORD];
  temp_dj[0] = 1.0f;
  temp_dj[1] = 1.0f;
  float *temp_j2_1 = new float[ORD];
  temp_j2_1[0] = 1.0f;
  temp_j2_1[1] = 1.0f;
  float *temp_j_1 = new float[ORD];
  temp_j_1[0] = 1.0f;
  temp_j_1[1] = 1.0f;
  for(int j=2; j<ORD; j++)
  {
    temp_dj[j] = (float)(1) / (float)(j);
    temp_j2_1[j] = (float) (2 * j -1);
    temp_j_1[j] = (float) (j - 1);
  }

  const float dx_k = (float) (2) / (float) (imgWidth_k);
  const float dy_k = (float) (2) / (float) (imgHeight_k);
  const float dx = (float) (2) / (float) (imgWidth);
  const float dy = (float) (2) / (float) (imgHeight);

  SAFE_CALL(cudaMemcpyToSymbol(dj, temp_dj, ORD * sizeof(float)), "cudaMemcpyToSymbol dj failed");
  SAFE_CALL(cudaMemcpyToSymbol(j2_1, temp_j2_1, ORD * sizeof(float)), "cudaMemcpyToSymbol j2_1 failed");
  SAFE_CALL(cudaMemcpyToSymbol(j_1, temp_j_1, ORD * sizeof(float)), "cudaMemcpyToSymbol j_1 failed");
  p_polynomial<<<gridDim_p_X_k, blockDim_P>>>(p_in_Xdir_k, imgWidth_k, dx_k);  // 把X方向转置
  p_polynomial<<<gridDim_p_Y_k, blockDim_P>>>(p_in_Ydir_k, imgHeight_k, dy_k);
  p_polynomial<<<gridDim_p_X, blockDim_P>>>(p_in_Xdir, imgWidth, dx);   // 把X方向转置
  p_polynomial<<<gridDim_p_Y, blockDim_P>>>(p_in_Ydir, imgHeight, dy);

  dim3 blockDim_trans(32, 32);
  float *p_in_Xdir_trans_k;
  SAFE_CALL(cudaMalloc((void**)&p_in_Xdir_trans_k, ORD * imgWidth_k * sizeof(float)), "cudaMalloc p_in_Xdir_trans_k failed");   
  dim3 gridDim_trans_k((imgWidth_k + blockDim_trans.x - 1) / blockDim_trans.x, (ORD + blockDim_trans.y - 1) / blockDim_trans.y);
  matrix_transpose<<<gridDim_trans_k, blockDim_trans>>>(p_in_Xdir_trans_k, p_in_Xdir_k, imgWidth_k, ORD);
  
  uchar *oriImg;
  float *resImg;
  SAFE_CALL(cudaMalloc((void**)&oriImg, imgHeight * imgWidth * sizeof(uchar)), "cudaMalloc oriImg failed");
  SAFE_CALL(cudaMalloc((void**)&resImg, imgHeight_k * imgWidth_k * sizeof(float)), "cudaMalloc resImg failed");
  SAFE_CALL(cudaMemcpy(oriImg, img_ori.data, imgHeight * imgWidth * sizeof(uchar), cudaMemcpyHostToDevice), "oriImg cudaMemcpyHostToDevice failed");

  dim3 blockDim_resize(32, 32);
  dim3 gridDim_resize((imgWidth_k + blockDim_resize.x - 1) / blockDim_resize.x, (imgHeight_k + blockDim_resize.y - 1) / blockDim_resize.y);
  inter_liner_k<<<gridDim_resize, blockDim_resize>>>(resImg, oriImg, imgHeight, imgWidth, imgHeight_k, imgWidth_k, scale);
  
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
      
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      std::cout << "CUBLAS 对象实例化出错" << std::endl;
    }
    getchar();
    return EXIT_FAILURE;
  }

  float *AC;
  SAFE_CALL(cudaMalloc((void**)&AC, imgHeight_k * ORD * sizeof(float)), "cudaMalloc AC failed");
  const float a = 1.0f, b = 0.0f;
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ORD, imgHeight_k, imgWidth_k, &a, 
  //             p_in_Xdir_trans_k, ORD, resImg, imgWidth_k, &b, AC, ORD);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ORD, imgHeight_k, imgWidth_k, &a, 
              p_in_Xdir_trans_k, ORD, resImg, imgWidth_k, &b, AC, ORD);

  float *lambda;
  float *lambda_cpu = new float[ORD * ORD];
  SAFE_CALL(cudaMalloc((void**)&lambda, ORD * ORD * sizeof(float)), "cudaMalloc lambda failed");
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ORD, ORD, imgHeight_k, &a, 
  //             AC, ORD, p_in_Xdir_k, imgHeight_k, &b, lambda, ORD);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ORD, ORD, imgHeight_k, &a, 
              AC, ORD, p_in_Ydir_k, imgHeight_k, &b, lambda, ORD);

  // SAFE_CALL(cudaMemcpy(lambda_cpu, lambda, ORD * ORD * sizeof(float), cudaMemcpyDeviceToHost), "lambda to lambda_cpu failed");
  // for(int i=0; i<ORD; i++)
  // {
  //   for(int j=0; j<i; j++) 
  //   { 
  //     std::cout << lambda_cpu[(i-j)*ORD + j] * (float)((2*(i-j)+1)*(2*j+1)) / (float)(imgHeight_k*imgWidth_k) <<" ";
  //   }
  //   std::cout << std::endl;
  // }

  //////////////////////////////////////////////////// 重建图像 /////////////////////////////////////////////////////
  float *p_in_Ydir_trans;
  SAFE_CALL(cudaMalloc((void**)&p_in_Ydir_trans, ORD * imgHeight * sizeof(float)), "cudaMalloc p_in_Ydir_trans failed"); 
  dim3 gridDim_trans((imgHeight + blockDim_trans.x - 1) / blockDim_trans.x, (ORD + blockDim_trans.y - 1) / blockDim_trans.y);
  matrix_transpose<<<gridDim_trans, blockDim_trans>>>(p_in_Ydir_trans, p_in_Ydir, imgHeight, ORD);

  uchar *recon_img;
  SAFE_CALL(cudaMalloc((void**)&recon_img, imgWidth * imgHeight * sizeof(uchar)), "cudaMalloc recon_img failed");
  dim3 blockDim_recons(32, 32);
  dim3 gridDim_recons((imgWidth + blockDim_recons.x - 1) / blockDim_recons.x, (imgHeight + blockDim_recons.y - 1) / blockDim_recons.y);
  img_recons<<<gridDim_recons, blockDim_recons>>>(recon_img, lambda, p_in_Xdir, p_in_Ydir_trans, imgHeight, imgWidth);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
  } 

  cv::Mat recon_img_cpu(imgHeight, imgWidth, CV_8UC1);
  SAFE_CALL(cudaMemcpy(recon_img_cpu.data, recon_img, imgHeight*imgWidth*sizeof(uchar), cudaMemcpyDeviceToHost), "cudaMemcpy recon_img_cpu.data failed");
  cv::imwrite("lenna-1024_" + std::to_string(K) + "_recons.tif", recon_img_cpu);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&runtime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  std::cout << "cudaEvent_t time: " << runtime * 1000 << " us" << std::endl;
  
  delete[] temp_dj;
  delete[] temp_j_1;
  delete[] temp_j2_1;
  delete[] lambda_cpu;
  // delete[] p_in_Xdir_cpu;
  // delete[] p_in_Ydir_cpu;
  SAFE_CALL(cudaFree(p_in_Xdir_k), "free p_in_Xdir_k failed");
  SAFE_CALL(cudaFree(p_in_Ydir_k), "free p_in_Ydir_k failed");
  SAFE_CALL(cudaFree(p_in_Xdir), "free p_in_Xdir_k failed");
  SAFE_CALL(cudaFree(p_in_Ydir), "free p_in_Ydir_k failed");
  SAFE_CALL(cudaFree(p_in_Xdir_trans_k), "free p_in_Xdir_trans_k failed");
  SAFE_CALL(cudaFree(p_in_Ydir_trans), "free p_in_Xdir_trans_k failed");
  SAFE_CALL(cudaFree(AC), "free AC failed");
  SAFE_CALL(cudaFree(lambda), "free lambda failed");
	SAFE_CALL(cudaFree(oriImg), "free oriImg failed");
  SAFE_CALL(cudaFree(resImg), "free resImg failed");
  return 0;
}
