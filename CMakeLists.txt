cmake_minimum_required(VERSION 3.21)

project(gpu_memory_layout_experiments VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHADERS "Compile GLSL shaders during the build" ON)
option(ENABLE_VULKAN_VALIDATION "Enable Vulkan validation by default at runtime" OFF)

find_package(Vulkan REQUIRED)
include(FetchContent)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(cli11)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(gpu_memory_layout_experiments
    src/main.cpp
    src/experiments/dispatch_basics_experiment.cpp
    src/experiments/aos_soa_experiment.cpp
    src/utils/app_options.cpp
    src/utils/json_exporter.cpp
    src/utils/vulkan_compute_utils.cpp
    src/utils/gpu_timestamp_timer.cpp
    src/vulkan_context.cpp
    src/utils/buffer_utils.cpp
    src/benchmark_runner.cpp
)

target_include_directories(gpu_memory_layout_experiments
    PRIVATE
        include
)

target_link_libraries(gpu_memory_layout_experiments
    PRIVATE
        Vulkan::Vulkan
        CLI11::CLI11
        nlohmann_json::nlohmann_json
)

target_compile_definitions(gpu_memory_layout_experiments
    PRIVATE
        $<$<BOOL:${ENABLE_VULKAN_VALIDATION}>:DEFAULT_ENABLE_VULKAN_VALIDATION=1>
)

if(MSVC)
    target_compile_options(gpu_memory_layout_experiments PRIVATE /W4 /permissive-)
    set_target_properties(gpu_memory_layout_experiments PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "--validation --experiment all --iterations 10 --warmup 3 --size 16M --output \"${CMAKE_SOURCE_DIR}/results/tables/all_run\""
    )
else()
    target_compile_options(gpu_memory_layout_experiments PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")

if(BUILD_SHADERS)
    if(DEFINED Vulkan_GLSLC_EXECUTABLE AND EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
        file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
            "${SHADER_SOURCE_DIR}/*.comp"
            "${SHADER_SOURCE_DIR}/*.vert"
            "${SHADER_SOURCE_DIR}/*.frag"
        )

        set(SPIRV_OUTPUTS)
        foreach(SHADER_FILE IN LISTS SHADER_SOURCES)
            get_filename_component(SHADER_NAME "${SHADER_FILE}" NAME)
            set(SPIRV_FILE "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")

            add_custom_command(
                OUTPUT "${SPIRV_FILE}"
                COMMAND "${Vulkan_GLSLC_EXECUTABLE}" "${SHADER_FILE}" -o "${SPIRV_FILE}"
                DEPENDS "${SHADER_FILE}"
                COMMENT "Compiling shader ${SHADER_NAME}"
                VERBATIM
            )

            list(APPEND SPIRV_OUTPUTS "${SPIRV_FILE}")
        endforeach()

        if(SPIRV_OUTPUTS)
            add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
            add_dependencies(gpu_memory_layout_experiments compile_shaders)
        endif()
    else()
        message(WARNING "glslc was not found by CMake. Set Vulkan_GLSLC_EXECUTABLE or disable BUILD_SHADERS.")
    endif()
endif()
