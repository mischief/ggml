#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// A shader for a custom op (ggml_custom_4d). The shader binds the op's
// sources in order, then dst, as storage buffers 0..n. Its push constants
// are the push_size bytes that setup writes, then one uint per binding:
// the byte offset of the tensor in its binding (a view can start off the
// binding alignment). setup also writes the element counts of the
// dispatch; the workgroup counts are those divided by wg_denoms.
struct ggml_vk_custom_op {
    const void * spv;
    size_t       spv_size;
    uint32_t     push_size;
    uint32_t     wg_denoms[3];
    void      (* setup)(const struct ggml_tensor * node, void * push, uint32_t elements[3]);
};

// Run the custom ops of fun with a shader on Vulkan devices; the CPU runs
// fun as before. Register before the first graph that holds such an op.
GGML_BACKEND_API void ggml_backend_vk_register_custom(ggml_custom_op_t fun, const struct ggml_vk_custom_op * op);

// Pinned host memory that ops on a Vulkan device read in place, over PCIe;
// the scheduler keeps ops on such tensors on the device. Other pinned host
// buffers stay CPU memory.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_read_buffer_type(void);

#ifdef  __cplusplus
}
#endif
