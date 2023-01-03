cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories(/home/cuidongdong/eigen)
include_directories(/home/cuidongdong/onnxruntime/include)


#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/cuidongdong/anaconda3/lib/python3.9/site-packages/onnxruntime)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})