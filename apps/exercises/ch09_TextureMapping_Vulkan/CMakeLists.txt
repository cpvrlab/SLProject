#
# CMake project definition for ch09_TextureMapping with Vulkan Graphics API
#

set(target ch09_TextureMapping_Vulkan)

sl_add_exercise(${target}
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CommandBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorPool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSet.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSetLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Device.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DrawingObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Framebuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/GPUProgram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/GPUShader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Instance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Material.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Node.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Object.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PhysicalDevice.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Pipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RangeManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Rectangle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderPass.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Sampler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderModule.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Sphere.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Statistic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Swapchain.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureImage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/UniformBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/UniformBufferObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vkEnums.h
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanRenderer.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CommandBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorPool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSetLayout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Framebuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GPUProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GPUShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Instance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PhysicalDevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rectangle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RangeManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sphere.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Swapchain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UniformBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UniformBufferObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanRenderer.cpp
    SHADERS
    LIBRARIES
    sl_math
    sl_external
    sl_utils
    sl_cv
    ${glfw_LIBS}
    ${vk_LIBS}
    )

# Generates glsl shader to spirv
if ("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX") #---------------------------------------------------------------------------
elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #---------------------------------------------------------------------
    set(GLSLC "${SL_PROJECT_ROOT}/externals/prebuilt/win64_vulkan_1.2.162.1/Bin32/glslc.exe")
elseif ("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #----------------------------------------------------------------------
    set(GLSLC "${SL_PROJECT_ROOT}/externals/prebuilt/win64_vulkan_1.2.162.1/macOS/bin/glslc")
endif ()

# To convert all shader in folder, replace filename with *.vert / *.frag
file(GLOB SHADER_SOURCE_FILES
        "${SL_PROJECT_ROOT}/data/shaders/vertShader.vert"
        "${SL_PROJECT_ROOT}/data/shaders/fragShader.frag"
        "${SL_PROJECT_ROOT}/data/shaders/vertShaderRed.vert"
        "${SL_PROJECT_ROOT}/data/shaders/fragShaderRed.frag"
        )

foreach (SHADER ${SHADER_SOURCE_FILES})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPIRV_OUTPUT "${SL_PROJECT_ROOT}/data/shaders/${FILE_NAME}.spv")

    add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLC} ${SHADER} -o ${SPIRV_OUTPUT}
            DEPENDS ${SHADER})

    message(STATUS "Compiled for Vulkan: " ${SPIRV_OUTPUT})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV_OUTPUT})
endforeach (SHADER)

add_custom_target(
        Shaders
        DEPENDS ${SPIRV_BINARY_FILES})

add_dependencies(${target} Shaders)
