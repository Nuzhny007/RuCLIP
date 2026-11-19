cmake_minimum_required(VERSION 3.18)
project(ruclip_basic_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Единственное, что нужно знать приложению — имя пакета.
# Путь к установленному пакету можно передать как -DruCLIP_DIR=<prefix>/lib/cmake/ruCLIP
# или через CMAKE_PREFIX_PATH=<prefix>.
find_package(ruCLIP CONFIG REQUIRED)

# Пример читает картинку с диска, поэтому дополнительно нужен imgcodecs
# (самой библиотеке хватает core/imgproc, которые тянет ruclip::ruclip).
find_package(OpenCV REQUIRED COMPONENTS imgcodecs)

add_executable(ruclip_basic main.cpp)
target_link_libraries(ruclip_basic PRIVATE ruclip::ruclip opencv_imgcodecs)
