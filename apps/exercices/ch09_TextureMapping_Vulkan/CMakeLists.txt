#
# CMake project definition for ch09_TextureMapping with Vulkan Graphics API
#

set(target ch09_TextureMapping_Vulkan)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/lib-SLProject/include/CVTypedefs.h
        ${SL_PROJECT_ROOT}/lib-SLProject/include/CVImage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vkUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CommandBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CommandPool.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorPool.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSet.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSetLayout.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Device.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Fence.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Framebuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Image.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageView.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IndexBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Instance.h
        ${CMAKE_CURRENT_SOURCE_DIR}/PhysicalDevice.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Pipeline.h
        ${CMAKE_CURRENT_SOURCE_DIR}/PipelineLayout.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Queue.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RenderPass.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Sampler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Semaphore.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ShaderModule.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Swapchain.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TextureImage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/UniformBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/UniformBufferObject.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/lib-SLProject/source/SL/SL.cpp
        ${SL_PROJECT_ROOT}/lib-SLProject/source/CV/CVImage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vkUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CommandBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CommandPool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorPool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DescriptorSetLayout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Fence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Framebuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageView.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/IndexBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Instance.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PhysicalDevice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PipelineLayout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RenderPass.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Sampler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Semaphore.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ShaderModule.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Swapchain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TextureImage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/UniformBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/UniformBufferObject.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/VertexBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Vertex.cpp
        )

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/vertShader.spv
        ${SL_PROJECT_ROOT}/data/shaders/fragShader.spv
        )

add_executable(${target}
        ${headers}
        ${sources}
        ${shaders}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps/exercices"
        )

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/lib-SLProject/include
        ${SL_PROJECT_ROOT}/lib-SLProject/include/CV
        ${SL_PROJECT_ROOT}/lib-SLProject/include/math
        ${SL_PROJECT_ROOT}/lib-SLProject/include/SL
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w/GL
        ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        ${vk_INCLUDE_DIR}
        PUBLIC
        INTERFACE
        )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
        PRIVATE
        ${PlatformLinkLibs}
        lib-SLExternal
        lib-Utils
        ${OpenCV_LIBS}
        ${glfw_LIBS}
        ${vk_LIBS}
        PUBLIC
        INTERFACE
        )

target_compile_definitions(${target}
        PRIVATE
        ${compile_definitions}
        PUBLIC
        ${DEFAULT_COMPILE_DEFINITIONS}
        INTERFACE
        )

target_compile_options(${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}
        INTERFACE
        )

