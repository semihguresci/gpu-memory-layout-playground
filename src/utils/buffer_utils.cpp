#include "utils/buffer_utils.hpp"

#include "vulkan_context.hpp"

#include <cstdint>
#include <iostream>

namespace {

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const bool matches_filter = (type_filter & (1U << i)) != 0U;
        const bool matches_properties = (memory_properties.memoryTypes[i].propertyFlags & properties) == properties;

        if (matches_filter && matches_properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

} // namespace

bool create_buffer_resource(VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, BufferResource& out_resource) {
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};

    if (vkCreateBuffer(device, &buffer_info, nullptr, &out_resource.buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(device, out_resource.buffer, &memory_requirements);

    const uint32_t memory_type = find_memory_type(physical_device, memory_requirements.memoryTypeBits, properties);

    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(device, out_resource.buffer, nullptr);
        out_resource.buffer = VK_NULL_HANDLE;
        return false;
    }

    const VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, memory_requirements.size,
                                          memory_type};

    if (vkAllocateMemory(device, &alloc_info, nullptr, &out_resource.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, out_resource.buffer, nullptr);
        out_resource.buffer = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(device, out_resource.buffer, out_resource.memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, out_resource.memory, nullptr);
        vkDestroyBuffer(device, out_resource.buffer, nullptr);
        out_resource.memory = VK_NULL_HANDLE;
        out_resource.buffer = VK_NULL_HANDLE;
        return false;
    }

    out_resource.size = size;
    return true;
}

bool map_buffer_memory(VulkanContext& context, const BufferResource& buffer, const char* label, void*& mapped_ptr) {
    mapped_ptr = nullptr;
    const VkResult map_result = vkMapMemory(context.device(), buffer.memory, 0U, buffer.size, 0U, &mapped_ptr);
    if (map_result != VK_SUCCESS || mapped_ptr == nullptr) {
        std::cerr << "vkMapMemory failed for " << label << " with VkResult=" << map_result << ".\n";
        mapped_ptr = nullptr;
        return false;
    }

    return true;
}

void destroy_buffer_resource(VkDevice device, BufferResource& resource) {
    if (resource.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, resource.buffer, nullptr);
        resource.buffer = VK_NULL_HANDLE;
    }

    if (resource.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, resource.memory, nullptr);
        resource.memory = VK_NULL_HANDLE;
    }

    resource.size = 0;
}
