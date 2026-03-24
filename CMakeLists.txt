cmake_minimum_required(VERSION 3.15)
project(StringUKFGNN LANGUAGES CXX)

# ---- C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

# ---- Dependencies
find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)

# ---- Executable
add_executable(sim
    src/main.cpp
    src/app/System.cpp
    src/Renderer.cpp
    src/simulation/PhysicSimulator.cpp
    src/simulation/measurement.cpp
    src/estimation/UKF.cpp
    src/Logs/Logger.cpp
)

set_property(TARGET sim PROPERTY CXX_STANDARD 17)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
target_compile_definitions(sim PRIVATE 
    DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    MODEL_PATH="${CMAKE_SOURCE_DIR}/GNN/model_scripted.pt"
)

# ---- Include directories
target_include_directories(sim PRIVATE
    include
)

# ---- Link libraries
target_link_libraries(sim PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
)