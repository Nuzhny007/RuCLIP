# Самостоятельный проект для проверки нейтрального слоя (Фаза 0).
# Не требует LibTorch/OpenCV — только C++17 компилятор.
cmake_minimum_required(VERSION 3.18)
project(ClipMathTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(test_clip_math
	../src/Backend/ClipMath.cpp
	test_clip_math.cpp)

target_include_directories(test_clip_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
