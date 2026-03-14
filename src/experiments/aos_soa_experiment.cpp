#include "experiments/aos_soa_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/layout_assert.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <ranges>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kFloatArraysForSoa = 8;
constexpr uint32_t kFloatsPerParticle = 8;

struct ParticleAosStd430 {
    float px;
    float py;
    float pz;
    float mass;
    float vx;
    float vy;
    float vz;
    float dt;
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleAosStd430);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleAosStd430);
GPU_LAYOUT_ASSERT_SIZE(ParticleAosStd430, 32);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleAosStd430, 4);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, px, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, py, 4);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, pz, 8);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, mass, 12);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vx, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vy, 20);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vz, 24);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, dt, 28);

std::vector<uint32_t> make_particle_counts(uint32_t max_particles) {
    const std::array<uint32_t, 3> base_counts = {1000000, 5000000, 10000000};
    std::vector<uint32_t> output;
    output.reserve(base_counts.size() + 3);

    for (uint32_t value : base_counts) {
        if (value <= max_particles) {
            output.push_back(value);
        }
    }

    if (output.empty()) {
        const std::array<uint32_t, 4> fallback = {131072, 262144, 524288, 1048576};
        for (uint32_t value : fallback) {
            if (value <= max_particles) {
                output.push_back(value);
            }
        }
    }

    if (output.empty() && max_particles > 0) {
        output.push_back(max_particles);
    }

    std::ranges::sort(output);
    output.erase(std::ranges::unique(output).begin(), output.end());
    return output;
}

bool create_aos_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                   VkShaderModule& shader_module, VkDescriptorSetLayout& descriptor_set_layout,
                                   VkDescriptorPool& descriptor_pool, VkDescriptorSet& descriptor_set,
                                   VkPipelineLayout& pipeline_layout, VkPipeline& pipeline,
                                   BufferResource& storage_buffer, VkDeviceSize buffer_size) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storage_buffer)) {
        return false;
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, shader_module)) {
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings, descriptor_set_layout)) {
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};

    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1, descriptor_pool)) {
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), descriptor_pool, descriptor_set_layout,
                                                     descriptor_set)) {
        return false;
    }

    const VkDescriptorBufferInfo buffer_info{storage_buffer.buffer, 0, storage_buffer.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {VulkanComputeUtils::DescriptorBufferBindingUpdate{
            .binding = 0, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = buffer_info}});

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)}};

    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {descriptor_set_layout}, push_constant_ranges,
                                                    pipeline_layout)) {
        return false;
    }

    return VulkanComputeUtils::create_compute_pipeline(context.device(), shader_module, pipeline_layout, "main",
                                                       pipeline);
}

bool create_soa_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                   VkShaderModule& shader_module, VkDescriptorSetLayout& descriptor_set_layout,
                                   VkDescriptorPool& descriptor_pool, VkDescriptorSet& descriptor_set,
                                   VkPipelineLayout& pipeline_layout, VkPipeline& pipeline,
                                   std::vector<BufferResource>& storage_buffers, VkDeviceSize buffer_size_per_array) {
    storage_buffers.assign(kFloatArraysForSoa, BufferResource{});

    for (auto& buffer : storage_buffers) {
        if (!create_buffer_resource(
                context.physical_device(), context.device(), buffer_size_per_array, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer)) {
            return false;
        }
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, shader_module)) {
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(kFloatArraysForSoa);
    for (uint32_t binding = 0; binding < kFloatArraysForSoa; ++binding) {
        bindings.push_back(VkDescriptorSetLayoutBinding{binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    }

    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings, descriptor_set_layout)) {
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFloatArraysForSoa}};

    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1, descriptor_pool)) {
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), descriptor_pool, descriptor_set_layout,
                                                     descriptor_set)) {
        return false;
    }

    std::vector<VulkanComputeUtils::DescriptorBufferBindingUpdate> updates;
    updates.reserve(kFloatArraysForSoa);
    for (uint32_t binding = 0; binding < kFloatArraysForSoa; ++binding) {
        const VkDescriptorBufferInfo info{storage_buffers[binding].buffer, 0, storage_buffers[binding].size};
        updates.push_back(VulkanComputeUtils::DescriptorBufferBindingUpdate{
            .binding = binding, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = info});
    }

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set, updates);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)}};

    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {descriptor_set_layout}, push_constant_ranges,
                                                    pipeline_layout)) {
        return false;
    }

    return VulkanComputeUtils::create_compute_pipeline(context.device(), shader_module, pipeline_layout, "main",
                                                       pipeline);
}

void cleanup_aos_resources(VulkanContext& context, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
                           VkDescriptorPool descriptor_pool, VkDescriptorSetLayout descriptor_set_layout,
                           VkShaderModule shader_module, BufferResource& storage_buffer) {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), pipeline, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), pipeline_layout, nullptr);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device(), descriptor_pool, nullptr);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device(), descriptor_set_layout, nullptr);
    }
    if (shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), shader_module, nullptr);
    }
    destroy_buffer_resource(context.device(), storage_buffer);
}

void cleanup_soa_resources(VulkanContext& context, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
                           VkDescriptorPool descriptor_pool, VkDescriptorSetLayout descriptor_set_layout,
                           VkShaderModule shader_module, std::vector<BufferResource>& storage_buffers) {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), pipeline, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), pipeline_layout, nullptr);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device(), descriptor_pool, nullptr);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device(), descriptor_set_layout, nullptr);
    }
    if (shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), shader_module, nullptr);
    }

    for (auto& buffer : storage_buffers) {
        destroy_buffer_resource(context.device(), buffer);
    }
    storage_buffers.clear();
}

BenchmarkResult run_kernel_case(VulkanContext& context, const BenchmarkRunner& runner, const std::string& name,
                                uint32_t particles, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
                                VkDescriptorSet descriptor_set) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);

    return runner.run_timed(name, [&]() {
        const double ms = context.measure_gpu_time_ms([&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0,
                                    nullptr);
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &particles);
            vkCmdDispatch(cmd, group_count, 1, 1);
        });

        if (!std::isfinite(ms)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return ms;
    });
}

} // namespace

std::vector<BenchmarkResult> run_aos_soa_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const AosSoaExperimentConfig& config) {
    std::vector<BenchmarkResult> results;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "AoS vs SoA experiment requires GPU timestamp support.\n";
        return results;
    }

    const std::string aos_shader = VulkanComputeUtils::resolve_shader_path(config.aos_shader_path, "06_aos.comp.spv");
    const std::string soa_shader = VulkanComputeUtils::resolve_shader_path(config.soa_shader_path, "06_soa.comp.spv");
    if (aos_shader.empty() || soa_shader.empty()) {
        std::cerr << "Could not locate SPIR-V shader(s) for AoS vs SoA experiment.\n";
        return results;
    }

    const auto max_particles = static_cast<uint32_t>(config.max_buffer_bytes / (kFloatsPerParticle * sizeof(float)));
    const std::vector<uint32_t> particle_counts = make_particle_counts(max_particles);
    if (particle_counts.empty()) {
        std::cerr << "Scratch buffer too small for AoS vs SoA experiment.\n";
        return results;
    }

    VkShaderModule aos_shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout aos_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool aos_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet aos_descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout aos_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline aos_pipeline = VK_NULL_HANDLE;
    BufferResource aos_buffer{};

    const VkDeviceSize aos_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kFloatsPerParticle * sizeof(float);
    if (!create_aos_pipeline_resources(context, aos_shader, aos_shader_module, aos_descriptor_layout,
                                       aos_descriptor_pool, aos_descriptor_set, aos_pipeline_layout, aos_pipeline,
                                       aos_buffer, aos_buffer_size)) {
        std::cerr << "Failed to set up AoS resources.\n";
        cleanup_aos_resources(context, aos_pipeline, aos_pipeline_layout, aos_descriptor_pool, aos_descriptor_layout,
                              aos_shader_module, aos_buffer);
        return results;
    }

    VkShaderModule soa_shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout soa_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool soa_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet soa_descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout soa_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline soa_pipeline = VK_NULL_HANDLE;
    std::vector<BufferResource> soa_buffers;

    const VkDeviceSize soa_array_size = static_cast<VkDeviceSize>(particle_counts.back()) * sizeof(float);
    if (!create_soa_pipeline_resources(context, soa_shader, soa_shader_module, soa_descriptor_layout,
                                       soa_descriptor_pool, soa_descriptor_set, soa_pipeline_layout, soa_pipeline,
                                       soa_buffers, soa_array_size)) {
        std::cerr << "Failed to set up SoA resources.\n";
        cleanup_aos_resources(context, aos_pipeline, aos_pipeline_layout, aos_descriptor_pool, aos_descriptor_layout,
                              aos_shader_module, aos_buffer);
        cleanup_soa_resources(context, soa_pipeline, soa_pipeline_layout, soa_descriptor_pool, soa_descriptor_layout,
                              soa_shader_module, soa_buffers);
        return results;
    }

    for (uint32_t particles : particle_counts) {
        results.push_back(run_kernel_case(context, runner, "06_aos_particles_" + std::to_string(particles), particles,
                                          aos_pipeline, aos_pipeline_layout, aos_descriptor_set));

        results.push_back(run_kernel_case(context, runner, "06_soa_particles_" + std::to_string(particles), particles,
                                          soa_pipeline, soa_pipeline_layout, soa_descriptor_set));
    }

    cleanup_aos_resources(context, aos_pipeline, aos_pipeline_layout, aos_descriptor_pool, aos_descriptor_layout,
                          aos_shader_module, aos_buffer);
    cleanup_soa_resources(context, soa_pipeline, soa_pipeline_layout, soa_descriptor_pool, soa_descriptor_layout,
                          soa_shader_module, soa_buffers);

    return results;
}
