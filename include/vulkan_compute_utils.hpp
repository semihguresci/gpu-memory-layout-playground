#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace VulkanComputeUtils {

struct DescriptorBufferBindingUpdate {
    uint32_t binding = 0;
    VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo buffer_info{};
};

bool readBinaryFile(const std::string& path, std::vector<char>& out_data);

bool createShaderModule(
    VkDevice device,
    const std::vector<char>& spirv_code,
    VkShaderModule& out_shader_module);

bool loadShaderModuleFromFile(
    VkDevice device,
    const std::string& path,
    VkShaderModule& out_shader_module);

bool createDescriptorSetLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    VkDescriptorSetLayout& out_layout);

bool createDescriptorPool(
    VkDevice device,
    const std::vector<VkDescriptorPoolSize>& pool_sizes,
    uint32_t max_sets,
    VkDescriptorPool& out_pool);

bool allocateDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptor_pool,
    VkDescriptorSetLayout descriptor_set_layout,
    VkDescriptorSet& out_set);

void updateDescriptorSetBuffers(
    VkDevice device,
    VkDescriptorSet descriptor_set,
    const std::vector<DescriptorBufferBindingUpdate>& updates);

bool createPipelineLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
    VkPipelineLayout& out_pipeline_layout);

bool createComputePipeline(
    VkDevice device,
    VkShaderModule shader_module,
    VkPipelineLayout pipeline_layout,
    const char* entry_point,
    VkPipeline& out_pipeline);

float queryTimestampPeriod(VkPhysicalDevice physical_device);

class GpuTimestampTimer {
public:
    GpuTimestampTimer() = default;

    bool initialize(VkDevice device, float timestamp_period_ns);
    void shutdown(VkDevice device);

    void recordStart(VkCommandBuffer command_buffer) const;
    void recordEnd(VkCommandBuffer command_buffer) const;

    bool resolveMilliseconds(VkDevice device, double& out_ms) const;
    [[nodiscard]] bool isReady() const { return query_pool_ != VK_NULL_HANDLE; }

private:
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ns_ = 0.0f;
};

} // namespace VulkanComputeUtils
