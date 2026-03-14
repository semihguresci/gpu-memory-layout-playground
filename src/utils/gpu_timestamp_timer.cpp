#include "utils/gpu_timestamp_timer.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace VulkanComputeUtils {

bool GpuTimestampTimer::initialize(VkDevice device, float timestamp_period_ns) {
    timestamp_period_ns_ = timestamp_period_ns;

    VkQueryPoolCreateInfo query_info{};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = kQueryCount;

    return vkCreateQueryPool(device, &query_info, nullptr, &query_pool_) == VK_SUCCESS;
}

void GpuTimestampTimer::shutdown(VkDevice device) {
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }
    timestamp_period_ns_ = 0.0F;
}

void GpuTimestampTimer::record_start(VkCommandBuffer command_buffer) const {
    vkCmdResetQueryPool(command_buffer, query_pool_, 0, kQueryCount);
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
}

void GpuTimestampTimer::record_end(VkCommandBuffer command_buffer) const {
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
}

bool GpuTimestampTimer::resolve_milliseconds(VkDevice device, double& out_ms) const {
    if (query_pool_ == VK_NULL_HANDLE || timestamp_period_ns_ <= 0.0F) {
        return false;
    }

    std::array<uint64_t, kQueryCount> timestamps{};
    const VkResult result =
        vkGetQueryPoolResults(device, query_pool_, 0, static_cast<uint32_t>(timestamps.size()), sizeof(timestamps),
                              timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS || timestamps[1] < timestamps[0]) {
        return false;
    }

    const uint64_t delta = timestamps[1] - timestamps[0];
    const double ns = static_cast<double>(delta) * static_cast<double>(timestamp_period_ns_);
    out_ms = ns / 1000000.0;

    return !std::isnan(out_ms) && std::isfinite(out_ms);
}

} // namespace VulkanComputeUtils
