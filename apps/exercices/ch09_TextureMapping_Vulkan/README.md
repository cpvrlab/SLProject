# Vulkan Renderer

## Known Issues
* Window resizing does not work (Results in Crash)
* After closing application, sometimes there is an error message in the memory allocator (Access violation)
* Only accepts textures with an alpha channel

## Third party header & libs
* Vulkan Memory Allocator: Used to create/store buffer and textures. It is just one header file. It is inside the source folder\
    https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
