#pragma once

#include <vulkan/vulkan.h>

namespace VulkanComputeUtils {

class GpuTimestampTimer {
  public:
    GpuTimestampTimer() = default;

    bool initialize(VkDevice device, float timestamp_period_ns);
    void shutdown(VkDevice device);

    void record_start(VkCommandBuffer command_buffer) const;
    void record_end(VkCommandBuffer command_buffer) const;

    bool resolve_milliseconds(VkDevice device, double& out_ms) const;
    [[nodiscard]] bool is_ready() const { return query_pool_ != VK_NULL_HANDLE; }

  private:
    static constexpr uint32_t kQueryCount = 2;

    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ns_ = 0.0F;
};

} // namespace VulkanComputeUtils
