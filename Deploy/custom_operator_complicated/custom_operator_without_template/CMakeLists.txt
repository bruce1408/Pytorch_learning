cmake_minimum_required(VERSION 3.23)
project (lango)
add_definitions(-std=c++11)

# Configuration
set(ONNXRUNTIME_VERSION 1.13.1)
set(ONNXRUNTIME_ARCH linux-x64)

message(STATUS "download the onnxruntime release...")
# Download an ONNX Runtime release to link against
set(ONNXRUNTIME_RELEASE_NAME onnxruntime-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION})
set(ONNXRUNTIME_RELEASE_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_RELEASE_NAME}.tgz)

# file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz DESTINATION ${CMAKE_BINARY_DIR}/external/)
set(ONNXRUNTIME_DIR ${CMAKE_BINARY_DIR}/external/${ONNXRUNTIME_RELEASE_NAME})
message(STATUS "the onnxruntime dir is:" ${ONNXRUNTIME_DIR})
# Add header files & libraries
include_directories(${ONNXRUNTIME_DIR}/include/)
include_directories(/home/cuidongdong/install_onnx/usr/local/include/)
include_directories(/home/cuidongdong/install_onnx/usr/local/include/onnxruntime/core/session)
link_directories(${ONNXRUNTIME_DIR}/lib/)
# link_directories(/home/cuidongdong/install_onnx/usr/local/lib/)



# Find libraries we're linking against
# find_library(LUA_LIBRARY lua HINTS ${LUA_DIR}/install/lib/)
# find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib/)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/home/cuidongdong/install_onnx/usr/local/lib/")


# Set up target
add_library(lango SHARED lang_ops.cc lang_ops.h custom_op.cc custom_op.h)
target_link_libraries(lango PUBLIC ${ONNXRUNTIME_LIBRARY})
