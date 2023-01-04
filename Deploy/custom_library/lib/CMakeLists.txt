set(LIBHELLO_SRC custom_op.cc custom_op.h)

# set(LINK_DIR /home/cuidongdong/install_onnx/usr/local/include/onnxruntime/core/session)

#Include path to header files for Custom Op
include_directories(/home/cuidongdong/eigen)  # 加载头文件
include_directories(/home/cuidongdong/install_onnx/usr/local/include)
include_directories(/home/cuidongdong/install_onnx/usr/local/include/onnxruntime/core/session)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/home/cuidongdong/install_onnx/usr/local/lib")
# link_directories(/home/cuidongdong/install_onnx/usr/local/lib)
# target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})

# add_library(/home/cuidongdong/install_onnx/usr/local/lib)
add_library(customop SHARED ${LIBHELLO_SRC} ${ONNXRUNTIME_LIBRARY})