# vision_core (§5.5). Host build on the Nano, L4T 32.7.x: gcc-7.5, cmake 3.10, CUDA 10.2 (sm_53), TensorRT 8.2,
# Jetson Multimedia API (nvidia-l4t-jetson-multimedia-api). Build: make vision-core
cmake_minimum_required(VERSION 3.10)
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
project(beni_vision_core CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -mcpu=cortex-a57 -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_53,code=sm_53 --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

set(MMAPI /usr/src/jetson_multimedia_api CACHE PATH "Jetson Multimedia API root")
set(TEGRA_LIB /usr/lib/aarch64-linux-gnu/tegra)
if(NOT EXISTS ${MMAPI}/include/NvVideoEncoder.h)
  message(FATAL_ERROR "MMAPI not found at ${MMAPI}: sudo apt install nvidia-l4t-jetson-multimedia-api")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

# Only the MMAPI helper classes this binary uses (samples/common/classes).
set(MMAPI_CLASSES NvBuffer NvElement NvElementProfiler NvLogging NvV4l2Element NvV4l2ElementPlane NvVideoEncoder
    NvVideoDecoder NvJpegEncoder)
set(MMAPI_SRC)
foreach(c ${MMAPI_CLASSES})
  list(APPEND MMAPI_SRC ${MMAPI}/samples/common/classes/${c}.cpp)
endforeach()
add_library(mmapi STATIC ${MMAPI_SRC})
target_include_directories(mmapi PUBLIC ${MMAPI}/include ${MMAPI}/include/libjpeg-8b)
target_compile_options(mmapi PRIVATE -w)

add_executable(vision_core
  src/main.cpp src/argus_cam.cpp src/encoder.cpp src/decoder.cpp src/detector.cpp src/trt_engine.cpp src/bytetrack.cpp
  src/kernels.cu)
target_include_directories(vision_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../../shared
  ${MMAPI}/argus/include /usr/local/cuda/include)
target_compile_options(vision_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
# cmake 3.10 has no target_link_directories: resolve each library to a full path.
set(LIBS)
foreach(l nvbuf_utils nvjpeg nvargus_socketclient v4l2 EGL cuda cudart nvinfer)
  find_library(LIB_${l} ${l} PATHS ${TEGRA_LIB} /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)
  if(NOT LIB_${l})
    message(FATAL_ERROR "lib${l} not found")
  endif()
  list(APPEND LIBS ${LIB_${l}})
endforeach()
target_link_libraries(vision_core PRIVATE mmapi ${LIBS} PkgConfig::ZMQ pthread)
# §14.4: NVTX ranges for Nsight Systems when CUDA's nvToolsExt is present (no-ops otherwise).
find_library(LIB_nvToolsExt nvToolsExt PATHS /usr/local/cuda/lib64)
if(LIB_nvToolsExt)
  target_compile_definitions(vision_core PRIVATE BENI_NVTX)
  target_link_libraries(vision_core PRIVATE ${LIB_nvToolsExt})
endif()
set_target_properties(vision_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
