#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace ScalarTypeWidthUtils {

enum class WidthVariant : uint8_t {
    kFp32 = 0U,
    kFp16Storage = 1U,
    kU32 = 2U,
    kU16 = 3U,
    kU8 = 4U,
};

uint32_t storage_units_for_variant(WidthVariant variant, uint32_t elements);

VkDeviceSize buffer_size_for_variant(WidthVariant variant, uint32_t elements);

double storage_bytes_per_element(WidthVariant variant);

float clamp_unit(float value);

float update_scalar(float value);

float make_seed_scalar(uint32_t index);

uint16_t float_to_half_bits(float value);

float half_bits_to_float(uint16_t bits);

uint16_t quantize_u16(float value);

float dequantize_u16(uint16_t value);

uint8_t quantize_u8(float value);

float dequantize_u8(uint8_t value);

uint16_t read_u16_lane(uint32_t word, uint32_t lane);

void write_u16_lane(uint32_t& word, uint32_t lane, uint16_t value);

uint8_t read_u8_lane(uint32_t word, uint32_t lane);

void write_u8_lane(uint32_t& word, uint32_t lane, uint8_t value);

float expected_variant_value(WidthVariant variant, uint32_t index);

float validation_tolerance(WidthVariant variant);

} // namespace ScalarTypeWidthUtils
