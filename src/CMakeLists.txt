cmake_minimum_required(VERSION 3.20)

cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

project(realcugan-ncnn-vulkan-wrapper)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif ()

find_package(Threads)
find_package(OpenMP)

if (NOT TARGET ncnn)
    find_package(ncnn QUIET)
endif ()

if (NOT TARGET ncnn)
    if (DEFINED ENV{ncnn_DIR})
        find_package(ncnn QUIET PATHS $ENV{ncnn_DIR} NO_DEFAULT_PATH)
    elseif (DEFINED ENV{NCNN_DIR})
        find_package(ncnn QUIET PATHS $ENV{NCNN_DIR} NO_DEFAULT_PATH)
    endif ()
endif ()

if (NOT TARGET ncnn)
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_SYSTEM_GLSLANG OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
    set(NCNN_DISABLE_EXCEPTION ON CACHE BOOL "" FORCE)
    set(NCNN_INT8 OFF CACHE BOOL "" FORCE)

    set(WITH_LAYER_convolution ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_crop ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_deconvolution ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_eltwise ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_flatten ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_innerproduct ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_input ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_pooling ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_reduction ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_relu ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_scale ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_split ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_packing ON CACHE BOOL "" FORCE)
    set(WITH_LAYER_gemm ON CACHE BOOL "" FORCE)

    add_subdirectory(realcugan-ncnn-vulkan/src/ncnn)
endif ()

macro(realcugan_add_shader SHADER_SRC)
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/realcugan-ncnn-vulkan/src/${SHADER_SRC_NAME_WE}.comp.hex.h)

    add_custom_command(
            OUTPUT ${SHADER_COMP_HEADER}
            COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/realcugan-ncnn-vulkan/src/${SHADER_SRC} -DSHADER_COMP_HEADER=${SHADER_COMP_HEADER} -P "${CMAKE_CURRENT_SOURCE_DIR}/realcugan-ncnn-vulkan/src/generate_shader_comp_header.cmake"
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/realcugan-ncnn-vulkan/src/${SHADER_SRC}
            COMMENT "Preprocessing shader source ${SHADER_SRC_NAME_WE}.comp"
            VERBATIM
    )
    set_source_files_properties(${SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_COMP_HEADER})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR}/realcugan-ncnn-vulkan/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/realcugan-ncnn-vulkan/src/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot")
endif ()

# enable global link time optimization
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_supported_output)
if (ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO is not supported: ${ipo_supported_output}")
endif ()

realcugan_add_shader(realcugan_preproc.comp)
realcugan_add_shader(realcugan_postproc.comp)
realcugan_add_shader(realcugan_4x_postproc.comp)
realcugan_add_shader(realcugan_preproc_tta.comp)
realcugan_add_shader(realcugan_postproc_tta.comp)
realcugan_add_shader(realcugan_4x_postproc_tta.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_library(realcugan-ncnn-vulkan-wrapper STATIC realcugan_wrapped.cpp realcugan-ncnn-vulkan/src/realcugan.cpp)
set_property(TARGET realcugan-ncnn-vulkan-wrapper PROPERTY CXX_STANDARD 17)

add_dependencies(realcugan-ncnn-vulkan-wrapper generate-spirv)

target_link_libraries(realcugan-ncnn-vulkan-wrapper ncnn)

install(TARGETS realcugan-ncnn-vulkan-wrapper ARCHIVE DESTINATION lib)
