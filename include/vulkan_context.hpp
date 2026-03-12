#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vulkan/vulkan.h>

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool initialize(bool enable_validation);
    void shutdown();

    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physical_device_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue computeQueue() const { return compute_queue_; }
    [[nodiscard]] uint32_t computeQueueFamilyIndex() const { return compute_queue_family_index_; }
    [[nodiscard]] bool gpuTimestampsSupported() const { return gpu_timestamps_supported_; }

    [[nodiscard]] std::string selectedDeviceName() const;
    [[nodiscard]] double measureGpuTimeMs(const std::function<void(VkCommandBuffer)>& record_commands);

private:
    bool createInstance(bool enable_validation);
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createCommandResources();
    bool createTimestampQueryPool();
    bool setupDebugMessenger();

    std::optional<uint32_t> findComputeQueueFamily(VkPhysicalDevice physical_device) const;
    bool checkValidationLayerSupport() const;
    bool checkDebugUtilsExtensionSupport() const;
    void destroyDebugMessenger();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence submit_fence_ = VK_NULL_HANDLE;
    VkQueryPool timestamp_query_pool_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index_ = 0;
    float timestamp_period_ = 0.0f;
    bool gpu_timestamps_supported_ = false;
    bool validation_enabled_ = false;
};
