#include "cuda_runtime.h"
#include <cuda.h>
#include <time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
#include <chrono>

#define K 5
#define ORD 60   // Legendre polynomial的阶数order
#define TILE_WIDTH 32 //TILE_WIDTH即block的宽

inline __device__
float __char_as_float(uchar b8)
{
  return __uint2float_rn(b8) / 127.5f - 1.f;
}

inline __device__
char __float_as_char(float f32)
{
    return (char)__float_as_int(__saturatef(f32));
}

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
__global__ void inter_liner_k(float3 *dataOut, uchar3 *dataIn, int imgHeight, int imgWidth, int imgHeight_k, int imgWidth_k, float scale)
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

    uchar3 s11 = dataIn[sy * imgWidth + sx];
    uchar3 s12 = dataIn[sy * imgWidth + sx2];
    uchar3 s21 = dataIn[sy2 * imgWidth + sx];
    uchar3 s22 = dataIn[sy2 * imgWidth + sx2];
       
    float h_rst00x, h_rst01x, h_rst00y, h_rst01y, h_rst00z, h_rst01z;
    h_rst00x = s11.x * cbufx.x + s12.x * cbufx.y;
    h_rst01x = s21.x * cbufx.x + s22.x * cbufx.y;
    h_rst00y = s11.y * cbufx.x + s12.y * cbufx.y;
    h_rst01y = s21.y * cbufx.x + s22.y * cbufx.y;
    h_rst00z = s11.z * cbufx.x + s12.z * cbufx.y;
    h_rst01z = s21.z * cbufx.x + s22.z * cbufx.y;

    dataOut[yIdx*imgWidth_k + xIdx].x = (h_rst00x * cbufy.x + h_rst01x * cbufy.y) / 127.5f - 1.f; // B
    dataOut[yIdx*imgWidth_k + xIdx].y = (h_rst00y * cbufy.x + h_rst01y * cbufy.y) / 127.5f - 1.f; // G
    dataOut[yIdx*imgWidth_k + xIdx].z = (h_rst00z * cbufy.x + h_rst01z * cbufy.y) / 127.5f - 1.f; // R
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

__global__ void cal_AC(float3 *AC, float3 *Img, float *P, int imgHeight_k, int imgWidth_k, int total_ph)
{
  __shared__ float3 Imgds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Pds[TILE_WIDTH][TILE_WIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Identify the row and column of the AC element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  // https://bbs.csdn.net/topics/360217239
  // https://blog.csdn.net/liuyujie3229166/article/details/106888245
  // result需要初始化，result本来就有值。如果不初始化，每次结果都不一样
  float3 result;
  result.x = 0.f; result.y = 0.f; result.z = 0.f;
  // Loop over the Img and P tiles required to compute AC element
  for(int ph=0; ph<total_ph; ++ph) 
  {
    // Collaborative loading of Img and P tiles into shared memory
    if((Row<imgHeight_k) && (ph*TILE_WIDTH+tx)<imgWidth_k)
      Imgds[ty][tx] = Img[Row*imgWidth_k + ph*TILE_WIDTH + tx];

    if((ph*TILE_WIDTH+ty)<imgWidth_k && Col<ORD)
      Pds[ty][tx] = P[(ph*TILE_WIDTH + ty)*ORD + Col]; 
    
    __syncthreads();

    for(int k=0; k<TILE_WIDTH; ++k)
    {
      result.x += Imgds[ty][k].x * Pds[k][tx];
      result.y += Imgds[ty][k].y * Pds[k][tx];
      result.z += Imgds[ty][k].z * Pds[k][tx];
    }
    __syncthreads();
  }
  if((Row<imgHeight_k) && (Col<ORD))
    AC[Row*ORD + Col] = result; 
}

__constant__ float coff[ORD * ORD];
// 根据公式(48)，计算图像矩
__global__ void cal_lambda(float3 *lambda, float *P, float3 *AC, int imgHeight_k, int imgWidth_k, int total_ph)
{
  __shared__ float Pds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float3 ACds[TILE_WIDTH][TILE_WIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Identify the row and column of the lambda element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float3 moments;
  moments.x = 0.f; moments.y = 0.f; moments.z = 0.f;
  // Loop over the P and AC tiles required to compute lambda element
  for(int ph=0; ph<total_ph; ++ph) 
  {
    // Collaborative loading of Img and P tiles into shared memory
    if((Row<ORD) && (ph*TILE_WIDTH+tx)<imgHeight_k)
      Pds[ty][tx] = P[Row*ORD + ph*TILE_WIDTH+tx];

    if((ph*TILE_WIDTH+ty<imgHeight_k) && Col<ORD)
      ACds[ty][tx] = AC[(ph*TILE_WIDTH+ty)*ORD + Col];
    
    __syncthreads();

    for(int k=0; k<TILE_WIDTH; ++k)
    {
      moments.x += Pds[ty][k] * ACds[k][tx].x;
      moments.y += Pds[ty][k] * ACds[k][tx].y;
      moments.z += Pds[ty][k] * ACds[k][tx].z;
    }
    __syncthreads();
  }
  
  if((Row<ORD) && (Col<ORD))
  {
    // lambda[(Row+Col)*ORD + Col].x = moments.x * coff[(Row+Col)*ORD + Col]; // 写入lambda矩阵的顺序错误，系数也不对
    // lambda[(Row+Col)*ORD + Col].y = moments.y * coff[(Row+Col)*ORD + Col];
    // lambda[(Row+Col)*ORD + Col].z = moments.z * coff[(Row+Col)*ORD + Col];

    // lambda[(Row+Col)*ORD + Col].x = moments.x * coff[Row*ORD + Col]; // 写入lambda矩阵的顺序错误，系数也不对
    // lambda[(Row+Col)*ORD + Col].y = moments.y * coff[Row*ORD + Col];
    // lambda[(Row+Col)*ORD + Col].z = moments.z * coff[Row*ORD + Col];

    lambda[Row*ORD + Col].x = moments.x * coff[Row*ORD + Col]; // 写入lambda矩阵的顺序错误，系数也不对
    lambda[Row*ORD + Col].y = moments.y * coff[Row*ORD + Col];
    lambda[Row*ORD + Col].z = moments.z * coff[Row*ORD + Col];
  }
  // if((Row<ORD) && (Col<ORD) && (Row<Col))
  // {
  //   lambda[Row*ORD + Col].x = 0.f;
  //   lambda[Row*ORD + Col].y = 0.f;
  //   lambda[Row*ORD + Col].z = 0.f;
  // }
}

// // 根据公式(7)，重建图像
// __global__ void img_recons(uchar3 *recon_img, float3 *lambda, float *p_in_Xdir, float *p_in_Ydir_trans, int imgHeight, int imgWidth)
// {
//   // (TILE_WIDTH + 1) * TILE_WIDTH / 2
//   // RTX 2060 shared memory per block中能存放49152 bytes，共12288个float类型的数
//   __shared__ float3 lam_ds[TILE_WIDTH][ORD];
//   __shared__ float PXds[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float PYds[TILE_WIDTH][TILE_WIDTH];

//   int bx = blockIdx.x; int by = blockIdx.y;
//   int tx = threadIdx.x; int ty = threadIdx.y;

//   // Identify the row and column of the recon_img element to work on
//   int Row = by * TILE_WIDTH + ty;
//   int Col = bx * TILE_WIDTH + tx;

//   float3 imgValue; //在每个phase中累加图像的值
//   imgValue.x = 0.f; imgValue.y = 0.f; imgValue.z = 0.f;

//   //根据ORD来计算需要多少ph
//   for(int ph=0; ph< (ORD+TILE_WIDTH-1) / TILE_WIDTH; ++ph)
//   {
//     // X方向
//     if((Col < imgWidth) && (ph*TILE_WIDTH + ty) < ORD)
//       PXds[ty][tx] = p_in_Xdir[(ph*TILE_WIDTH + ty) * imgWidth + Col];
//     // Y方向
//     if((Row < imgHeight) && (ph*TILE_WIDTH + tx) < ORD)
//       PYds[ty][tx] = p_in_Ydir_trans[Row * ORD + ph*TILE_WIDTH + tx];

//     // 把lambda矩阵中X方向的moments全部加载进来
//     for(int ph_lam=0; ph_lam<(ORD+TILE_WIDTH-1) / TILE_WIDTH; ++ph_lam)
//     {
//       if((ph*TILE_WIDTH + ty) < ORD && (ph_lam*TILE_WIDTH + tx) < ORD)
//         lam_ds[ty][ph_lam*TILE_WIDTH + tx] = lambda[(ph*TILE_WIDTH + ty) * ORD + ph_lam*TILE_WIDTH + tx];
//     }
//     __syncthreads();

//     for(int i=0; i<TILE_WIDTH; ++i)
//     {
//       for(int j=0; j<ORD; ++j) //j<=i+ph*TILE_WIDTH
//       {
//         imgValue.x += lam_ds[i-j][j].x * PYds[tx][ty] * PXds[ty][tx];
//         imgValue.y += lam_ds[i-j][j].y * PYds[tx][ty] * PXds[ty][tx];
//         imgValue.z += lam_ds[i-j][j].z * PYds[tx][ty] * PXds[ty][tx];
//       }
//     }
//     __syncthreads();
//   }
//   recon_img[Row*imgWidth + Col].x = __float_as_uint((imgValue.x + 1.f) * 127.5f);
//   recon_img[Row*imgWidth + Col].y = __float_as_uint((imgValue.y + 1.f) * 127.5f);
//   recon_img[Row*imgWidth + Col].z = __float_as_uint((imgValue.z + 1.f) * 127.5f);
// }

// 根据公式(7)，重建图像
__global__ void img_recons(uchar3 *recon_img, float3 *lambda, float *p_in_Xdir, float *p_in_Ydir_trans, int imgHeight, int imgWidth)
{
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Identify the row and column of the recon_img element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float3 imgValue; //在每个phase中累加图像的值
  imgValue.x = 0.f; imgValue.y = 0.f; imgValue.z = 0.f;

  if((Row<imgHeight) && (Col<imgWidth))
  {
    for(int i=0; i<ORD; ++i)  // row of lambda
    {
      for(int j=0; j<i; ++j) // col of lambda
      { // X和Y是不是反了？
        // imgValue.x += lambda[(i-j)*ORD + j].x * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i-j];
        // imgValue.y += lambda[(i-j)*ORD + j].y * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i-j];
        // imgValue.z += lambda[(i-j)*ORD + j].z * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i-j];
        
        // imgValue.x += lambda[(i-j)*ORD + j].x * p_in_Xdir[(i-j)*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + j];
        // imgValue.y += lambda[(i-j)*ORD + j].y * p_in_Xdir[(i-j)*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + j];
        // imgValue.z += lambda[(i-j)*ORD + j].z * p_in_Xdir[(i-j)*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + j];
        
        imgValue.x += lambda[i*ORD + j].x * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i];
        imgValue.y += lambda[i*ORD + j].y * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i];
        imgValue.z += lambda[i*ORD + j].z * p_in_Xdir[j*imgWidth + Col] * p_in_Ydir_trans[Row*ORD + i];
      }
    }
    recon_img[Row*imgWidth + Col].x = (imgValue.x + 1.f) * 127.5f;
    recon_img[Row*imgWidth + Col].y = (imgValue.y + 1.f) * 127.5f;
    recon_img[Row*imgWidth + Col].z = (imgValue.z + 1.f) * 127.5f;  
  }
}

int main(void)
{  
  cv::Mat img_ori = cv::imread("lena.tiff"); 
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
  
  uchar3 *oriImg;
  float3 *resImg;
  SAFE_CALL(cudaMalloc((void**)&oriImg, imgHeight * imgWidth * sizeof(uchar3)), "cudaMalloc oriImg failed");
  SAFE_CALL(cudaMalloc((void**)&resImg, imgHeight_k * imgWidth_k * sizeof(float3)), "cudaMalloc resImg failed");
  SAFE_CALL(cudaMemcpy(oriImg, img_ori.data, imgHeight * imgWidth * sizeof(uchar3), cudaMemcpyHostToDevice), "oriImg cudaMemcpyHostToDevice failed");

  dim3 blockDim_resize(32, 32);
  dim3 gridDim_resize((imgWidth_k + blockDim_resize.x - 1) / blockDim_resize.x, (imgHeight_k + blockDim_resize.y - 1) / blockDim_resize.y);
  inter_liner_k<<<gridDim_resize, blockDim_resize>>>(resImg, oriImg, imgHeight, imgWidth, imgHeight_k, imgWidth_k, scale);
  
  float3 *AC;
  SAFE_CALL(cudaMalloc((void**)&AC, imgHeight_k * ORD * sizeof(float3)), "cudaMalloc AC failed");
  dim3 blockDim_AC(32, 32);
  // 根据AC^T的大小，确定需要block和grid的维数
  dim3 gridDim_AC((ORD + blockDim_AC.x - 1) / blockDim_AC.x, (imgHeight_k + blockDim_AC.y - 1) / blockDim_AC.y); 
  cal_AC<<<gridDim_AC, blockDim_AC>>>(AC, resImg, p_in_Xdir_trans_k, imgHeight_k, imgWidth_k, total_ph_X); 

  float3 *lambda;
  float3 *lambda_cpu = new float3[ORD * ORD];
  SAFE_CALL(cudaMalloc((void**)&lambda, ORD * ORD * sizeof(float3)), "cudaMalloc lambda failed");
  dim3 blockDim_lambda(32, 32);
  dim3 gridDim_lambda((ORD + blockDim_lambda.x - 1) / blockDim_lambda.x, (ORD + blockDim_lambda.y - 1) / blockDim_lambda.y);
  
  float *temp_coff = new float[ORD * ORD];
  for(int i=0; i<ORD; i++)
  {
    for(int j=0; j<ORD; j++)
    {
      temp_coff[i * ORD + j] = (float)((2*i+1) * (2*j+1)) / (float)(imgHeight_k * imgWidth_k);
      // if(j<=i)
      //   temp_coff[i * ORD + j] = (float)((2*i+1) * (2*j+1)) / (float)(imgHeight_k * imgWidth_k);
      // else
      //   temp_coff[i * ORD + j] = 0.f;
    }
  }
  SAFE_CALL(cudaMemcpyToSymbol(coff, temp_coff, ORD * ORD * sizeof(float)), "cudaMemcpyToSymbol coff failed");
  cal_lambda<<<gridDim_lambda, blockDim_lambda>>>(lambda, p_in_Ydir_k, AC, imgHeight_k, imgWidth_k, total_ph_Y);

  // SAFE_CALL(cudaMemcpy(lambda_cpu, lambda, ORD * ORD * sizeof(float3), cudaMemcpyDeviceToHost), "lambda to lambda_cpu failed");
  // for(int i=0; i<ORD; i++)
  // {
  //   for(int j=0; j<ORD; j++) 
  //   { 
  //     std::cout << lambda_cpu[i*ORD + j].x <<" "<<lambda_cpu[i*ORD + j].y <<" "<<lambda_cpu[i*ORD + j].z <<" ";
  //   }
  //   std::cout << std::endl;
  // }

  //////////////////////////////////////////////////// 重建图像 /////////////////////////////////////////////////////
  float *p_in_Ydir_trans;
  SAFE_CALL(cudaMalloc((void**)&p_in_Ydir_trans, ORD * imgHeight * sizeof(float)), "cudaMalloc p_in_Ydir_trans failed"); 
  dim3 gridDim_trans((imgHeight + blockDim_trans.x - 1) / blockDim_trans.x, (ORD + blockDim_trans.y - 1) / blockDim_trans.y);
  matrix_transpose<<<gridDim_trans, blockDim_trans>>>(p_in_Ydir_trans, p_in_Ydir, imgHeight, ORD);

  uchar3 *recon_img;
  SAFE_CALL(cudaMalloc((void**)&recon_img, imgWidth * imgHeight * sizeof(uchar3)), "cudaMalloc recon_img failed");
  dim3 blockDim_recons(32, 32);
  dim3 gridDim_recons((imgWidth + blockDim_recons.x - 1) / blockDim_recons.x, (imgHeight + blockDim_recons.y - 1) / blockDim_recons.y);
  img_recons<<<gridDim_recons, blockDim_recons>>>(recon_img, lambda, p_in_Xdir, p_in_Ydir_trans, imgHeight, imgWidth);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
  } 

  cv::Mat recon_img_cpu(imgHeight, imgWidth, CV_8UC3);
  SAFE_CALL(cudaMemcpy(recon_img_cpu.data, recon_img, imgHeight*imgWidth*sizeof(uchar3), cudaMemcpyDeviceToHost), "cudaMemcpy recon_img_cpu.data failed");
  cv::imwrite("lena_" + std::to_string(K) + "_recons.tiff", recon_img_cpu);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&runtime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  std::cout << "cudaEvent_t time: " << runtime * 1000 << " us" << std::endl;
  
  delete[] temp_dj;
  delete[] temp_j_1;
  delete[] temp_j2_1;
  delete[] temp_coff;
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
