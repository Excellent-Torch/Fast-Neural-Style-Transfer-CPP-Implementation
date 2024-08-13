# CMakeList.txt : CMake project for FastNeuralStyleTransfer, include source and define
# project specific logic here.

add_executable (FastNeuralStyleTransfer "FastNeuralStyleTransfer.cpp" "FastNeuralStyleTransfer.h"    )


set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
set(Onnx_LIBRARIES "${VCPKG_INSTALLED_DIR}/x64-windows/lib/onnxruntime.lib")


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)





if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET FastNeuralStyleTransfer PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(FastNeuralStyleTransfer ${OpenCV_LIBS} ${CUDA_RUNTIME_LIBRARY} ${Onnx_LIBRARIES} )

