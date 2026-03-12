#pragma once

#include <vulkan/vulkan.h>

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

bool createBufferResource(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    BufferResource& out_resource);

void destroyBufferResource(VkDevice device, BufferResource& resource);
