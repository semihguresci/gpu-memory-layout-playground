#pragma once

#include <vulkan/vulkan.h>

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

bool create_buffer_resource(VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, BufferResource& out_resource);

void destroy_buffer_resource(VkDevice device, BufferResource& resource);
