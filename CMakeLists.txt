CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Rectangular_moments)
FIND_PACKAGE(OpenCV REQUIRED)
# ibopencv_imgcodecs
FIND_PACKAGE(CUDA REQUIRED)
CUDA_ADD_EXECUTABLE(rectangular rectangular.cu)
TARGET_LINK_LIBRARIES(rectangular ${OpenCV_LIBS} ${CUDA_cublas_LIBRARY})

# set(CMAKE_CXX_FLAGS "-std=c++11") # 加不加这行都可以
# 直接鼠标右键Run Code不能跑
# mkdir build && cd build
# cd ..
# ./build/display_image
