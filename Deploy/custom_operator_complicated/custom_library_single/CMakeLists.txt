cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
# set(LINK_DIR /home/cuidongdong/install_onnx/usr/local/include/onnxruntime/core/session)

# add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories(/home/cuidongdong/eigen)  # 加载头文件
include_directories(/home/cuidongdong/install_onnx/usr/local/include)
include_directories(/home/cuidongdong/install_onnx/usr/local/include/onnxruntime/core/session)

add_library(custom_op SHARED custom_op.cc custom_op.h)


# add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})


#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/home/cuidongdong/install_onnx/usr/local/lib")
# link_directories(/home/cuidongdong/install_onnx/usr/local/lib)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})