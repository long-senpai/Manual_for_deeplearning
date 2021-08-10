cmake_minimum_required(VERSION 3.10.2)
project(video_stb LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4.5.2 REQUIRED)
message("OpenCV Version ${OpenCV_Version}")
include_directories(include)


add_executable(${PROJECT_NAME} main.cpp include/main.h)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES})
