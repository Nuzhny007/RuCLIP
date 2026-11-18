# CMakeList.txt: проект CMake для RuCLIP
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RuCLIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Бэкенды (Фаза 1: реализован только LibTorch) ---
option(RUCLIP_WITH_TORCH "Собрать бэкенд LibTorch"   ON)
option(RUCLIP_WITH_TRT   "Собрать бэкенд TensorRT"   OFF)
option(RUCLIP_WITH_DNN   "Собрать бэкенд OpenCV DNN" OFF)

if(NOT RUCLIP_WITH_TORCH AND NOT RUCLIP_WITH_TRT AND NOT RUCLIP_WITH_DNN)
  message(FATAL_ERROR "RuCLIP: включите хотя бы один бэкенд (RUCLIP_WITH_TORCH / RUCLIP_WITH_TRT / RUCLIP_WITH_DNN)")
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS "4.6")
  message(FATAL_ERROR "RuCLIP: требуется OpenCV >= 4.6 (найдено ${OpenCV_VERSION})")
endif()
# DNN-бэкенд использует новый graph-движок и нативный LayerNorm — нужен OpenCV 5.x.
if(RUCLIP_WITH_DNN AND OpenCV_VERSION VERSION_LESS "5.0")
  message(FATAL_ERROR "RuCLIP: OpenCV DNN backend требует OpenCV 5.x (найдено ${OpenCV_VERSION})")
endif()

# torch (cu126) ожидает цель CUDA::nvToolsExt, которой нет в CUDA 13
# (NVTX3 заменил nvToolsExt). Библиотека приложению не нужна (см. README),
# поэтому достаточно пустой INTERFACE-цели, чтобы CMake прошёл генерацию.
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

if(RUCLIP_WITH_TORCH)
  find_package(Torch REQUIRED)
endif()

if(RUCLIP_WITH_TRT)
  set(RUCLIP_TRT_ROOT "" CACHE PATH "TensorRT root (содержит include/ и lib/)")
  if(NOT RUCLIP_TRT_ROOT)
    if(DEFINED ENV{TENSORRT_ROOT})
      set(RUCLIP_TRT_ROOT "$ENV{TENSORRT_ROOT}")
    else()
      file(GLOB _trt_candidates "C:/work/libraries/cuda/TensorRT-*")
      list(LENGTH _trt_candidates _trt_count)
      if(_trt_count GREATER 0)
        list(SORT _trt_candidates)
        math(EXPR _trt_last "${_trt_count} - 1")
        list(GET _trt_candidates ${_trt_last} RUCLIP_TRT_ROOT)
      endif()
    endif()
  endif()

  find_path(RUCLIP_NVINFER_INCLUDE_DIR NvInfer.h
            HINTS "${RUCLIP_TRT_ROOT}/include" REQUIRED)
  find_library(RUCLIP_NVINFER_LIB nvinfer_10
               HINTS "${RUCLIP_TRT_ROOT}/lib" REQUIRED)
  find_library(RUCLIP_NVONNXPARSER_LIB nvonnxparser_10
               HINTS "${RUCLIP_TRT_ROOT}/lib" REQUIRED)
  message(STATUS "RuCLIP: TensorRT root = ${RUCLIP_TRT_ROOT}")

  find_package(CUDAToolkit REQUIRED)
endif()

include("CMakeLists.Files.txt")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

source_group("Headers" FILES ${HEADERS})
set(SOURCES ${SOURCES} ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

if(RUCLIP_WITH_TORCH)
  target_compile_definitions(${PROJECT_NAME} PRIVATE RUCLIP_HAS_TORCH=1)
endif()
if(RUCLIP_WITH_TRT)
  target_compile_definitions(${PROJECT_NAME} PRIVATE RUCLIP_HAS_TRT=1)
endif()
if(RUCLIP_WITH_DNN)
  target_compile_definitions(${PROJECT_NAME} PRIVATE RUCLIP_HAS_DNN=1)
endif()

if (MSVC)
  if(RUCLIP_WITH_TORCH)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET RuCLIP
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:RuCLIP>)
  endif()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /openmp")
endif (MSVC)
