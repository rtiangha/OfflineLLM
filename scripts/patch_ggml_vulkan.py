#!/usr/bin/env python3
"""Patch llama.cpp's ggml-vulkan.cpp so it can start Vulkan from a libadrenotools
custom driver (e.g. Mesa Turnip) instead of the system libvulkan.so.

Usage: python3 scripts/patch_ggml_vulkan.py llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp

If the env vars GGML_VK_CUSTOM_DRIVER_DIR, GGML_VK_CUSTOM_DRIVER_NAME,
GGML_VK_CUSTOM_TMP_DIR and GGML_VK_CUSTOM_HOOK_DIR are all set (the app sets them via
SmolLM.setCustomDriver), Vulkan is initialised from the custom driver's
vkGetInstanceProcAddr. Otherwise behaviour is unchanged.

The patch does four things:
  1. dispatcher .init(vkGetInstanceProcAddr)       -> .init(ggml_vk_resolve_gipa())
  2. raw vkGetInstanceProcAddr(...) calls           -> ggml_vk_active_gipa(...)
  3. every other raw vkXxx(...) call (vkGetPhysicalDeviceFeatures2, vkGetDeviceProcAddr,
     vkCmdCopyBuffer, ...)                          -> VULKAN_HPP_DEFAULT_DISPATCHER.vkXxx(...)
Step 3 is also applied to the sibling ggml-vulkan-*.cpp files in the same directory.
  4. supports_op hook: if env GGML_VK_UNSUPPORTED_TYPES is a comma-separated list of ggml
     type names (e.g. "Q6_K" or "q6_K,q5_K"), the Vulkan backend reports MUL_MAT,
     MUL_MAT_ID and GET_ROWS on those weight types as unsupported, so llama.cpp keeps those
     weights in CPU memory and runs those ops on the CPU. Works around driver/shader bugs
     (e.g. Q6_K on Turnip Adreno 660) while everything else stays on the GPU.
  5. BDA is now OFF by default (set OFFLINELLM_VK_ENABLE_BDA to turn it on): it crashed both the
     stock Adreno driver and Turnip on an Adreno 660, and is only an optional optimisation.
  5b. API filter: physical devices below Vulkan 1.2 are dropped (logged as "physical device N:
     Vulkan x.y.z -> kept|dropped"), so the app falls back to the CPU instead of crashing.
     Opt out: OFFLINELLM_VK_ALLOW_OLD_API.
  6. diagnostics: after the dispatcher is initialised, logs (tag ggml-vulkan) which key
     Vulkan entry points are null; and guards the vkGetBufferDeviceAddress call in
     ggml_vk_create_buffer against a null entry point.
  7. tracing: logs "buffer_device_address=<0|1> (OFFLINELLM_VK_ENABLE_BDA set|unset)" when
     the device is created, and, with env OFFLINELLM_VK_TRACE set, breadcrumb lines before
     the map / bind / buffer-address / log steps of ggml_vk_create_buffer (the last
     breadcrumb before a crash shows which step died).
  8. per-device entry points: after the logical device is created, the dispatcher is
     re-initialised with its device (init(device) = vkGetDeviceProcAddr). Some drivers
     (Qualcomm on Adreno 6xx) leave Vulkan 1.2 core entries such as vkWaitSemaphores and
     vkGetBufferDeviceAddress empty in the loader's table although the trampolines
     returned for the instance exist, which crashes with a null call; init(device) fetches the
     driver's own pointers and falls back to the KHR names. Opt out: OFFLINELLM_VK_NO_DEVICE_INIT.
(2-3 matter because the linked system loader must never be handed objects that belong
to the custom driver.) Idempotent; exits non-zero if the init anchor is missing.
"""
import glob
import os
import re
import sys

MARKER = "GGML_VK_ADRENOTOOLS_PATCH"
NOT_PREFIXED = r'(?<![A-Za-z0-9_.:>"])'  # skip PFN_vk..., obj.vk..., ns::vk..., "vk..."

HELPER = r'''
// ---- GGML_VK_ADRENOTOOLS_PATCH (added by scripts/patch_ggml_vulkan.py) ----
#if defined(__ANDROID__)
#include <dlfcn.h>
#include <cstdlib>
#include <android/log.h>
extern "C" {
#include <adrenotools/driver.h>
}
#endif

static PFN_vkGetInstanceProcAddr g_ggml_vk_gipa = vkGetInstanceProcAddr;

static PFN_vkGetInstanceProcAddr ggml_vk_resolve_gipa() {
#if defined(__ANDROID__)
    const char * drv_dir  = getenv("GGML_VK_CUSTOM_DRIVER_DIR");   // must end with '/'
    const char * drv_name = getenv("GGML_VK_CUSTOM_DRIVER_NAME");  // e.g. vulkan.ad07XX.so
    const char * tmp_dir  = getenv("GGML_VK_CUSTOM_TMP_DIR");
    const char * hook_dir = getenv("GGML_VK_CUSTOM_HOOK_DIR");     // nativeLibraryDir
    if (drv_dir && drv_name && tmp_dir && hook_dir && *drv_dir && *drv_name && *tmp_dir && *hook_dir) {
        void * handle = adrenotools_open_libvulkan(
            RTLD_NOW | RTLD_LOCAL, ADRENOTOOLS_DRIVER_CUSTOM,
            tmp_dir, hook_dir, drv_dir, drv_name, nullptr, nullptr);
        if (handle != nullptr) {
            auto gipa = (PFN_vkGetInstanceProcAddr) dlsym(handle, "vkGetInstanceProcAddr");
            if (gipa != nullptr) {
                __android_log_print(ANDROID_LOG_INFO, "ggml-vulkan", "using custom Vulkan driver %s%s", drv_dir, drv_name);
                g_ggml_vk_gipa = gipa; // handle intentionally never closed
                return gipa;
            }
        }
        __android_log_print(ANDROID_LOG_ERROR, "ggml-vulkan", "custom driver %s%s failed to load; using system Vulkan", drv_dir, drv_name);
    }
#endif
    g_ggml_vk_gipa = vkGetInstanceProcAddr;
    return g_ggml_vk_gipa;
}

static PFN_vkVoidFunction ggml_vk_active_gipa(VkInstance instance, const char * name) {
    return g_ggml_vk_gipa(instance, name);
}

#include <string>
#include <cctype>
#include <cstdlib>
#include <mutex>

// True if the user asked (GGML_VK_UNSUPPORTED_TYPES) for this op's weight type to stay off the GPU.
static bool ggml_vk_type_rejected(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID && op->op != GGML_OP_GET_ROWS) {
        return false;
    }
    if (op->src[0] == nullptr) {
        return false;
    }
    // Re-read the variable on every call (cheap) so the app can add types at run time,
    // e.g. after its automatic Q6_K check.
    static std::mutex mtx;
    static std::string last_raw = "\x01";
    static std::string list;
    const char * e = getenv("GGML_VK_UNSUPPORTED_TYPES");
    const std::string raw = e != nullptr ? e : "";
    std::string name = ggml_type_name(op->src[0]->type);
    for (char & ch : name) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (raw != last_raw) {
        last_raw = raw;
        std::string v = raw;
        for (char & ch : v) {
            ch = (char) std::tolower((unsigned char) ch);
        }
        list = "," + v + ",";
    }
    if (list.size() <= 2) {
        return false;
    }
    return list.find("," + name + ",") != std::string::npos;
}

static void ggml_vk_log_dispatcher_state() {
#if defined(__ANDROID__)
    auto & d = VULKAN_HPP_DEFAULT_DISPATCHER;
    __android_log_print(ANDROID_LOG_INFO, "ggml-vulkan",
        "dispatcher non-null: CreateBuffer=%d MemReq=%d AllocMem=%d MapMem=%d BindBufMem=%d GetBDA=%d GetBDAKHR=%d WaitSem=%d WaitSemKHR=%d SemCount=%d",
        d.vkCreateBuffer != nullptr, d.vkGetBufferMemoryRequirements != nullptr,
        d.vkAllocateMemory != nullptr, d.vkMapMemory != nullptr, d.vkBindBufferMemory != nullptr,
        d.vkGetBufferDeviceAddress != nullptr, d.vkGetBufferDeviceAddressKHR != nullptr,
        d.vkWaitSemaphores != nullptr, d.vkWaitSemaphoresKHR != nullptr, d.vkGetSemaphoreCounterValue != nullptr);
#endif
}
// ---- end GGML_VK_ADRENOTOOLS_PATCH ----
'''

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    src = open(path, encoding="utf-8").read()
    if MARKER in src:
        print("already patched")
        return

    init_pat = re.compile(r"\.init\(\s*vkGetInstanceProcAddr\s*\)")
    n_init = len(init_pat.findall(src))
    if n_init == 0:
        print("ERROR: anchor '.init(vkGetInstanceProcAddr)' not found.")
        print("Loader/dispatcher lines present in this llama.cpp revision:")
        for i, line in enumerate(src.splitlines(), 1):
            if any(k in line for k in ("DISPATCHER.init", ".init(", "DynamicLoader", "getProcAddress", "vkGetInstanceProcAddr")):
                print(f"  {i}: {line.strip()}")
        sys.exit(1)
    src = init_pat.sub(".init(ggml_vk_resolve_gipa())", src)

    RAW = re.compile(NOT_PREFIXED + r"(vk[A-Z][A-Za-z0-9]*)\(")
    src, n_gipa = re.subn(NOT_PREFIXED + r"vkGetInstanceProcAddr\(", "ggml_vk_active_gipa(", src)
    src, n_raw = RAW.subn(lambda m: "VULKAN_HPP_DEFAULT_DISPATCHER." + m.group(1) + "(", src)
    counts = {"vkGetInstanceProcAddr": n_gipa, "other raw vk calls": n_raw}

    sup_pat = re.compile(r"(static bool ggml_backend_vk_device_supports_op\(ggml_backend_dev_t dev, const ggml_tensor \* op\) \{\n)")
    src, n_sup = sup_pat.subn(lambda m: m.group(1) + "    if (ggml_vk_type_rejected(op)) {\n        return false;\n    }\n", src, count=1)
    if n_sup == 0:
        print("WARNING: ggml_backend_vk_device_supports_op not found; GGML_VK_UNSUPPORTED_TYPES will have no effect")

    disp_old = "VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);"
    if src.count(disp_old) == 1:
        src = src.replace(disp_old, disp_old + "\n    ggml_vk_log_dispatcher_state();", 1)
        counts["dispatcher log"] = 1
    else:
        print("WARNING: dispatcher init(instance) anchor not found; no dispatcher diagnostics")

    dev_old = "device->device = device->physical_device.createDevice(device_create_info);"
    if src.count(dev_old) == 1:
        src = src.replace(dev_old, dev_old + "\n"
            "        if (getenv(\"OFFLINELLM_VK_NO_DEVICE_INIT\") == nullptr) {\n"
            "            VULKAN_HPP_DEFAULT_DISPATCHER.init(device->device);\n"
            "            ggml_vk_log_dispatcher_state();\n"
            "        }", 1)
        counts["device init"] = 1
    else:
        print("WARNING: createDevice anchor not found once; per-device entry points not initialised")

    bda_old = "device->buffer_device_address = vk12_features.bufferDeviceAddress;"
    n_bda = src.count(bda_old)
    if n_bda == 1:
        src = src.replace(bda_old,
            "device->buffer_device_address = vk12_features.bufferDeviceAddress && getenv(\"OFFLINELLM_VK_ENABLE_BDA\") != nullptr;\n"
            "#if defined(__ANDROID__)\n"
            "        __android_log_print(ANDROID_LOG_INFO, \"ggml-vulkan\", \"buffer_device_address=%d (driver reports %d, OFFLINELLM_VK_ENABLE_BDA %s)\", (int) device->buffer_device_address, (int) vk12_features.bufferDeviceAddress, getenv(\"OFFLINELLM_VK_ENABLE_BDA\") != nullptr ? \"set\" : \"unset\");\n"
            "#endif", 1)
    else:
        print(f"WARNING: expected 1 buffer_device_address assignment, found {n_bda}; BDA stays enabled")
    counts["BDA default off"] = n_bda

    # Drop physical devices below Vulkan 1.2: this ggml needs vkWaitSemaphores and friends as
    # core 1.2 entry points, which a 1.1 driver (stock Adreno 660) does not provide -> null call.
    found_old = "GGML_LOG_DEBUG(\"ggml_vulkan: Found %zu Vulkan devices:\\n\", vk_instance.device_indices.size());"
    if src.count(found_old) == 1:
        src = src.replace(found_old,
            "// GGML_VK_ADRENOTOOLS_PATCH: keep only devices that expose Vulkan 1.2 (see scripts/patch_ggml_vulkan.py).\n"
            "    if (getenv(\"OFFLINELLM_VK_ALLOW_OLD_API\") == nullptr) {\n"
            "        std::vector<size_t> kept;\n"
            "        for (size_t k : vk_instance.device_indices) {\n"
            "            const uint32_t v = devices[k].getProperties().apiVersion;\n"
            "#if defined(__ANDROID__)\n"
            "            __android_log_print(ANDROID_LOG_INFO, \"ggml-vulkan\", \"physical device %zu: Vulkan %u.%u.%u -> %s\", k, VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v), v >= VK_API_VERSION_1_2 ? \"kept\" : \"dropped (needs 1.2)\");\n"
            "#endif\n"
            "            if (v >= VK_API_VERSION_1_2) kept.push_back(k);\n"
            "        }\n"
            "        vk_instance.device_indices = kept;\n"
            "        if (vk_instance.device_indices.empty()) {\n"
            "            GGML_LOG_INFO(\"ggml_vulkan: No devices with Vulkan 1.2 found.\\n\");\n"
            "            return;\n"
            "        }\n"
            "    }\n    " + found_old, 1)
        counts["API>=1.2 filter"] = 1
    else:
        print("WARNING: 'Found %zu Vulkan devices' anchor not found; no API version filter")

    # Helper goes after the last #include of the leading include block.
    lines = src.split("\n")
    last_inc = -1
    for i, line in enumerate(lines[:250]):
        if line.startswith("#include"):
            last_inc = i
    if last_inc < 0:
        print("ERROR: no #include lines found near top of file")
        sys.exit(1)
    lines.insert(last_inc + 1, HELPER)
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print(f"patched: init x{n_init}, " + ", ".join(f"{k} x{v}" for k, v in counts.items()) + f", supports_op hook x{n_sup}")

    # Sibling translation units (ggml-vulkan-buffers.cpp etc.): same dispatcher rewrite.
    d = os.path.dirname(os.path.abspath(path))
    for other in sorted(glob.glob(os.path.join(d, "ggml-vulkan*.cpp"))):
        if other == os.path.abspath(path):
            continue
        t = open(other, encoding="utf-8").read()
        if re.search(NOT_PREFIXED + r"vkGetInstanceProcAddr\(", t):
            print(f"WARNING: {os.path.basename(other)} calls vkGetInstanceProcAddr directly; not patched")
        t2, k = RAW.subn(lambda m: "VULKAN_HPP_DEFAULT_DISPATCHER." + m.group(1) + "(", t)
        bda_site = "buf->bda_addr = device->device.getBufferAddress(addressInfo);"
        if bda_site in t2:
            t2 = t2.replace(
                bda_site,
                "if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferDeviceAddress != nullptr) {\n"
                "            " + bda_site + "\n"
                "        } else {\n"
                "            GGML_LOG_WARN(\"ggml_vulkan: vkGetBufferDeviceAddress entry point is null; skipping\\n\");\n"
                "        }", 1)
            print(f"guarded vkGetBufferDeviceAddress in {os.path.basename(other)}")
        if "ggml_vk_create_buffer(" in t2 and "OFFLINELLM_TRACE" not in t2:
            macro = (
                "#include \"ggml-vulkan-common.h\"\n"
                "#if defined(__ANDROID__)\n#include <android/log.h>\n#include <cstdlib>\n"
                "#define OFFLINELLM_TRACE(label) do { if (getenv(\"OFFLINELLM_VK_TRACE\") != nullptr) "
                "__android_log_print(ANDROID_LOG_INFO, \"ggml-vulkan\", \"create_buffer: %s size=%zu\", label, size); } while (0)\n"
                "#else\n#define OFFLINELLM_TRACE(label) do {} while (0)\n#endif\n")
            if t2.count('#include "ggml-vulkan-common.h"\n') >= 1:
                t2 = t2.replace('#include "ggml-vulkan-common.h"\n', macro, 1)
            for stmt, label in (
                ("buf->ptr = device->device.mapMemory(", "before mapMemory"),
                ("device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);", "before bindBufferMemory"),
                ("if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferDeviceAddress != nullptr) {", "before getBufferAddress"),
                ("device->memory_logger->log_allocation(buf, size);", "before log_allocation"),
            ):
                if t2.count(stmt) == 1:
                    t2 = t2.replace(stmt, 'OFFLINELLM_TRACE("' + label + '");\n        ' + stmt, 1)
                else:
                    print(f"WARNING: trace anchor not found once: {stmt[:50]}")
        if t2 != t:
            open(other, "w", encoding="utf-8").write(t2)
            print(f"patched {os.path.basename(other)}: {k} raw vk call(s)")

if __name__ == "__main__":
    main()
