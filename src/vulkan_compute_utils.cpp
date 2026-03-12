#include "vulkan_compute_utils.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace VulkanComputeUtils {

bool readBinaryFile(const std::string& path, std::vector<char>& out_data) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << "\n";
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::cerr << "File is empty or unreadable: " << path << "\n";
        return false;
    }

    out_data.resize(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(out_data.data(), size);
    return file.good();
}

bool createShaderModule(
    VkDevice device,
    const std::vector<char>& spirv_code,
    VkShaderModule& out_shader_module) {
    if (spirv_code.empty() || spirv_code.size() % sizeof(uint32_t) != 0) {
        return false;
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(spirv_code.data());

    return vkCreateShaderModule(device, &create_info, nullptr, &out_shader_module) == VK_SUCCESS;
}

bool loadShaderModuleFromFile(
    VkDevice device,
    const std::string& path,
    VkShaderModule& out_shader_module) {
    std::vector<char> spirv_code;
    if (!readBinaryFile(path, spirv_code)) {
        return false;
    }

    return createShaderModule(device, spirv_code, out_shader_module);
}

bool createDescriptorSetLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    VkDescriptorSetLayout& out_layout) {
    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.bindingCount = static_cast<uint32_t>(bindings.size());
    create_info.pBindings = bindings.empty() ? nullptr : bindings.data();

    return vkCreateDescriptorSetLayout(device, &create_info, nullptr, &out_layout) == VK_SUCCESS;
}

bool createDescriptorPool(
    VkDevice device,
    const std::vector<VkDescriptorPoolSize>& pool_sizes,
    uint32_t max_sets,
    VkDescriptorPool& out_pool) {
    VkDescriptorPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.maxSets = max_sets;
    create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    create_info.pPoolSizes = pool_sizes.empty() ? nullptr : pool_sizes.data();

    return vkCreateDescriptorPool(device, &create_info, nullptr, &out_pool) == VK_SUCCESS;
}

bool allocateDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptor_pool,
    VkDescriptorSetLayout descriptor_set_layout,
    VkDescriptorSet& out_set) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout;

    return vkAllocateDescriptorSets(device, &alloc_info, &out_set) == VK_SUCCESS;
}

void updateDescriptorSetBuffers(
    VkDevice device,
    VkDescriptorSet descriptor_set,
    const std::vector<DescriptorBufferBindingUpdate>& updates) {
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(updates.size());

    for (const auto& update : updates) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = update.binding;
        write.descriptorCount = 1;
        write.descriptorType = update.descriptor_type;
        write.pBufferInfo = &update.buffer_info;
        writes.push_back(write);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool createPipelineLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
    VkPipelineLayout& out_pipeline_layout) {
    VkPipelineLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    create_info.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    create_info.pSetLayouts = descriptor_set_layouts.empty() ? nullptr : descriptor_set_layouts.data();

    return vkCreatePipelineLayout(device, &create_info, nullptr, &out_pipeline_layout) == VK_SUCCESS;
}

bool createComputePipeline(
    VkDevice device,
    VkShaderModule shader_module,
    VkPipelineLayout pipeline_layout,
    const char* entry_point,
    VkPipeline& out_pipeline) {
    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = entry_point;

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;

    return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &out_pipeline) == VK_SUCCESS;
}

float queryTimestampPeriod(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);
    return props.limits.timestampPeriod;
}

bool GpuTimestampTimer::initialize(VkDevice device, float timestamp_period_ns) {
    timestamp_period_ns_ = timestamp_period_ns;

    VkQueryPoolCreateInfo query_info{};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 2;

    return vkCreateQueryPool(device, &query_info, nullptr, &query_pool_) == VK_SUCCESS;
}

void GpuTimestampTimer::shutdown(VkDevice device) {
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }
    timestamp_period_ns_ = 0.0f;
}

void GpuTimestampTimer::recordStart(VkCommandBuffer command_buffer) const {
    vkCmdResetQueryPool(command_buffer, query_pool_, 0, 2);
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
}

void GpuTimestampTimer::recordEnd(VkCommandBuffer command_buffer) const {
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
}

bool GpuTimestampTimer::resolveMilliseconds(VkDevice device, double& out_ms) const {
    if (query_pool_ == VK_NULL_HANDLE || timestamp_period_ns_ <= 0.0f) {
        return false;
    }

    std::array<uint64_t, 2> timestamps{};
    const VkResult result = vkGetQueryPoolResults(
        device,
        query_pool_,
        0,
        static_cast<uint32_t>(timestamps.size()),
        sizeof(timestamps),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS || timestamps[1] < timestamps[0]) {
        return false;
    }

    const uint64_t delta = timestamps[1] - timestamps[0];
    const double ns = static_cast<double>(delta) * static_cast<double>(timestamp_period_ns_);
    out_ms = ns / 1000000.0;

    return !std::isnan(out_ms) && std::isfinite(out_ms);
}

} // namespace VulkanComputeUtils
