cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
if (NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif ()

project(realcugan-ncnn-vulkan-wrapper)

cmake_minimum_required(VERSION 3.9)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif ()

find_package(Threads)
find_package(OpenMP)
find_package(Vulkan REQUIRED)
find_package(ncnn REQUIRED)

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

find_package(glslang QUIET)
if (TARGET glslang::glslang AND TARGET glslang::SPIRV)
    message(STATUS "Found glslang via find_package")
    list(APPEND EXTRA_GLSLANG_LIBS glslang::glslang glslang::SPIRV)
else()
    set(GLSLANG_TARGET_DIR "GLSLANG-NOTFOUND" CACHE PATH "Absolute path to glslangTargets.cmake directory")
    if (NOT GLSLANG_TARGET_DIR AND DEFINED ENV{GLSLANG_TARGET_DIR})
        set(GLSLANG_TARGET_DIR $ENV{GLSLANG_TARGET_DIR})
    elseif(NOT GLSLANG_TARGET_DIR AND DEFINED ENV{VULKAN_SDK})
        set(GLSLANG_TARGET_DIR "$ENV{VULKAN_SDK}/lib/cmake")
    endif()

    if (GLSLANG_TARGET_DIR AND EXISTS "${GLSLANG_TARGET_DIR}/glslangTargets.cmake")
        message(STATUS "Using glslang install located at ${GLSLANG_TARGET_DIR}")
        find_package(Threads)
        if(EXISTS "${GLSLANG_TARGET_DIR}/OSDependentTargets.cmake")
            include("${GLSLANG_TARGET_DIR}/OSDependentTargets.cmake")
        endif()
        if(EXISTS "${GLSLANG_TARGET_DIR}/OGLCompilerTargets.cmake")
            include("${GLSLANG_TARGET_DIR}/OGLCompilerTargets.cmake")
        endif()
        if(EXISTS "${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
            include("${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
        endif()
        include("${GLSLANG_TARGET_DIR}/glslangTargets.cmake")
        include("${GLSLANG_TARGET_DIR}/SPIRVTargets.cmake")
    else()
        find_package(glslang REQUIRED)
    endif()
endif()

realcugan_add_shader(realcugan_preproc.comp)
realcugan_add_shader(realcugan_postproc.comp)
realcugan_add_shader(realcugan_4x_postproc.comp)
realcugan_add_shader(realcugan_preproc_tta.comp)
realcugan_add_shader(realcugan_postproc_tta.comp)
realcugan_add_shader(realcugan_4x_postproc_tta.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_library(realcugan-ncnn-vulkan-wrapper STATIC realcugan_wrapped.cpp realcugan-ncnn-vulkan/src/realcugan.cpp)

add_dependencies(realcugan-ncnn-vulkan-wrapper generate-spirv)

set(REALCUGAN_LINK_LIBRARIES ncnn ${Vulkan_LIBRARY} ${EXTRA_GLSLANG_LIBS})

target_link_libraries(realcugan-ncnn-vulkan-wrapper ${REALCUGAN_LINK_LIBRARIES})

install(TARGETS realcugan-ncnn-vulkan-wrapper ARCHIVE DESTINATION lib)