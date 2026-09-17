# 1. Set the minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# 2. Define your project name, version, and language
project(MyProject VERSION 1.0.0 LANGUAGES CXX)

# 3. Enforce the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. Define the executable target and add source files
add_executable(neural_network 
    src/neuralnetwork/neural_net_conv.cpp
    src/matvec/mat.cpp 
    src/matvec/vec.cpp
    src/matvec/tensor.cpp
)



# # 5. Tell CMake where to look for your headers (Adding local raylib/include)
# target_include_directories(neural_network PRIVATE 
#     ${CMAKE_SOURCE_DIR}/src/matvec/include 
#     ${CMAKE_SOURCE_DIR}
#     ${CMAKE_SOURCE_DIR}/raylib/include
# )

# # 6. Link the local pre-compiled static library file directly
# target_link_libraries(neural_network PRIVATE 
#     ${CMAKE_SOURCE_DIR}/raylib/lib/raylib.lib
#     Winmm.lib
#     Gdi32.lib
#     User32.lib
#     Shell32.lib
# )

# 5. Tell CMake where to look for your header files
target_include_directories(neural_network PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/matvec/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/raylib/include
)

# 6. Link libraries with system dependencies
target_link_libraries(neural_network PRIVATE 
    ${CMAKE_SOURCE_DIR}/raylib/lib/raylib.lib
    winmm.lib
    gdi32.lib
    user32.lib
    shell32.lib
    opengl32.lib
)

# 7. Existing definitions and asset generation scripts
target_compile_definitions(neural_network PRIVATE MNIST_DATA_LOCATION="${CMAKE_SOURCE_DIR}/data/")

add_custom_command(TARGET neural_network POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:neural_network>/mnist 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/mnist/train-images-idx3-ubyte $<TARGET_FILE_DIR:neural_network>/mnist/train-images-idx3-ubyte 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/mnist/train-labels-idx1-ubyte $<TARGET_FILE_DIR:neural_network>/mnist/train-labels-idx1-ubyte 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/mnist/t10k-images-idx3-ubyte $<TARGET_FILE_DIR:neural_network>/mnist/t10k-images-idx3-ubyte 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/mnist/t10k-labels-idx1-ubyte $<TARGET_FILE_DIR:neural_network>/mnist/t10k-labels-idx1-ubyte 
)

if (MSVC)
    target_compile_options(neural_network PRIVATE /O2 /arch:AVX2)
endif()

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(neural_network PRIVATE OpenMP::OpenMP_CXX)
endif()