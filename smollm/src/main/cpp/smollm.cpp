#include "LLMInference.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

// Tells the (patched) ggml-vulkan backend which libadrenotools custom driver to use.
// MUST be called before initBackends(): the Vulkan instance is created once per
// process, so a different driver only takes effect after an app restart.
// Pass empty/null strings to use the system Vulkan driver.
static void setOrClearEnv(JNIEnv* env, const char* name, jstring value) {
    const char* c = value != nullptr ? env->GetStringUTFChars(value, nullptr) : nullptr;
    if (c != nullptr && c[0] != '\0') {
        setenv(name, c, 1);
    } else {
        unsetenv(name);
    }
    if (c != nullptr) env->ReleaseStringUTFChars(value, c);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_setCustomDriver(JNIEnv* env, jobject thiz,
        jstring tmpDir, jstring hookDir, jstring driverDir, jstring driverName) {
    setOrClearEnv(env, "GGML_VK_CUSTOM_TMP_DIR", tmpDir);
    setOrClearEnv(env, "GGML_VK_CUSTOM_HOOK_DIR", hookDir);
    setOrClearEnv(env, "GGML_VK_CUSTOM_DRIVER_DIR", driverDir);
    setOrClearEnv(env, "GGML_VK_CUSTOM_DRIVER_NAME", driverName);
}

// Loads the ggml backend plugins (libggml-cpu-android_*.so variants + libggml-vulkan.so)
// from the app's nativeLibraryDir. ggml scores each CPU variant against the device's
// actual features (dotprod/fp16/i8mm/SVE...) and keeps only the best one. Idempotent.
extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_initBackends(JNIEnv* env, jobject thiz, jstring nativeLibDir) {
    static std::atomic_bool initialized{false};
    if (initialized.exchange(true)) {
        return;
    }
    jboolean isCopy = true;
    const char* dirCstr = env->GetStringUTFChars(nativeLibDir, &isCopy);
    if (dirCstr != nullptr && dirCstr[0] != '\0') {
        __android_log_print(ANDROID_LOG_INFO, "smollm-backends",
                            "loading backend plugins from %s", dirCstr);
        ggml_backend_load_all_from_path(dirCstr);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "smollm-backends",
                            "nativeLibraryDir EMPTY — falling back to ggml_backend_load_all() "
                            "(likely loads nothing on Android)");
        ggml_backend_load_all();
    }
    env->ReleaseStringUTFChars(nativeLibDir, dirCstr);

    // Which plugins actually won the load-time scoring. Seven ggml-cpu variants
    // ship in the APK and exactly one is kept; without this there is no way to
    // tell a device running the dotprod/i8mm kernels from one that silently fell
    // back to the armv8.0 baseline, and no way to see whether Vulkan registered.
    const size_t devCount = ggml_backend_dev_count();
    __android_log_print(ANDROID_LOG_INFO, "smollm-backends",
                        "registry after load: %zu device(s)", devCount);
    for (size_t i = 0; i < devCount; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);
        const char* desc = ggml_backend_dev_description(dev);
        const char* type;
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   type = "CPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type = "IGPU";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "ACCEL"; break;
            default:                             type = "OTHER"; break;
        }
        __android_log_print(ANDROID_LOG_INFO, "smollm-backends",
                            "  device %zu: [%s] name=%s desc=%s",
                            i, type, name ? name : "?", desc ? desc : "?");
    }
}

// Returns the description of the first GPU-type ggml backend device (e.g. the
// Vulkan device name like "Adreno (TM) 640"), or "" when no GPU backend
// registered — either not compiled in or Vulkan init failed on this device.
extern "C" JNIEXPORT jstring JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_getGpuDeviceName(JNIEnv* env, jobject thiz) {
    std::string result;
    try {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                const char* desc = ggml_backend_dev_description(dev);
                if (desc != nullptr) {
                    result = desc;
                }
                break;
            }
        }
    } catch (...) {
        result.clear();
    }
    return env->NewStringUTF(result.c_str());
}

// Which backends actually registered, and which ggml-cpu variant won the runtime
// scoring. GGML_CPU_ALL_VARIANTS picks one plugin per device silently, and ggml
// only logs the scores under NDEBUG-disabled GGML_LOG_DEBUG — so on a release
// build a device that fell back to the armv8.0 baseline (no DOTPROD/MATMUL_INT8,
// several times slower on quantized models) is indistinguishable from one that
// picked the right variant. This surfaces the feature list so "it feels slow"
// reports can be checked instead of guessed at.
extern "C" JNIEXPORT jstring JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_getBackendReport(JNIEnv* env, jobject thiz) {
    std::string out;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        const char* regName = ggml_backend_reg_name(reg);
        out += regName != nullptr ? regName : "(unnamed)";

        auto getFeatures = (ggml_backend_get_features_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (getFeatures != nullptr) {
            out += " [";
            bool first = true;
            for (ggml_backend_feature* f = getFeatures(reg); f != nullptr && f->name != nullptr; f++) {
                if (!first) {
                    out += ' ';
                }
                out += f->name;
                // Flags report "1"; only value-carrying entries (SVE_CNT, ...) need it spelled out.
                if (f->value != nullptr && strcmp(f->value, "1") != 0) {
                    out += '=';
                    out += f->value;
                }
                first = false;
            }
            out += ']';
        }
        out += '\n';
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* devName = ggml_backend_dev_name(dev);
        const char* devDesc = ggml_backend_dev_description(dev);
        const char* devType;
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   devType = "CPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   devType = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  devType = "iGPU";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: devType = "ACCEL"; break;
            default:                             devType = "?";     break;
        }
        out += "device ";
        out += devName != nullptr ? devName : "(unnamed)";
        out += " (";
        out += devType;
        out += "): ";
        out += devDesc != nullptr ? devDesc : "";
        out += '\n';
    }

    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jfloat minP,
                                            jfloat temperature, jfloat topP, jint topK, jfloat repeatPenalty,
                                            jboolean storeChats, jlong contextSize,
                                            jstring chatTemplate, jint nThreads, jboolean useMmap, jboolean useMlock,
                                            jint nGpuLayers, jint nThreadsBatch, jboolean kvCacheQ8) {
    jboolean    isCopy           = true;
    const char* modelPathCstr    = env->GetStringUTFChars(modelPath, &isCopy);
    auto*       llmInference     = new LLMInference();
    const char* chatTemplateCstr = env->GetStringUTFChars(chatTemplate, &isCopy);

    try {
        llmInference->loadModel(modelPathCstr, minP, temperature, topP, topK, repeatPenalty,
                                storeChats, contextSize, chatTemplateCstr, nThreads, useMmap, useMlock,
                                nGpuLayers, nThreadsBatch, kvCacheQ8);
    } catch (std::exception& error) {
        env->ReleaseStringUTFChars(modelPath, modelPathCstr);
        env->ReleaseStringUTFChars(chatTemplate, chatTemplateCstr);
        delete llmInference;
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
        return 0;
    }

    env->ReleaseStringUTFChars(modelPath, modelPathCstr);
    env->ReleaseStringUTFChars(chatTemplate, chatTemplateCstr);
    return reinterpret_cast<jlong>(llmInference);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_addChatMessage(JNIEnv* env, jobject thiz, jlong modelPtr, jstring message,
                                                 jstring role) {
    jboolean    isCopy       = true;
    const char* messageCstr  = env->GetStringUTFChars(message, &isCopy);
    const char* roleCstr     = env->GetStringUTFChars(role, &isCopy);
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->addChatMessage(messageCstr, roleCstr);
    env->ReleaseStringUTFChars(message, messageCstr);
    env->ReleaseStringUTFChars(role, roleCstr);
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_getResponseGenerationSpeed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getResponseGenerationTime();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_getContextSizeUsed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getContextSizeUsed();
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_close(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    delete llmInference;
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_startCompletion(JNIEnv* env, jobject thiz, jlong modelPtr, jstring prompt) {
    jboolean    isCopy       = true;
    const char* promptCstr   = env->GetStringUTFChars(prompt, &isCopy);
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    try {
        llmInference->startCompletion(promptCstr);
    } catch (std::exception& error) {
        env->ReleaseStringUTFChars(prompt, promptCstr);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
        return;
    }
    env->ReleaseStringUTFChars(prompt, promptCstr);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_completionLoop(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    try {
        std::string response = llmInference->completionLoop();
        return env->NewStringUTF(response.c_str());
    } catch (std::exception& error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_stopCompletion(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->stopCompletion();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_benchModel(JNIEnv* env, jobject /*unused*/, jlong modelPtr, jint pp, jint tg, jint pl,
                                             jint nr) {
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    std::string result       = llmInference->benchModel(pp, tg, pl, nr);
    return env->NewStringUTF(result.c_str());
}

// ---------------------------------------------------------------------------
// GPU self-test: runs a few ggml ops on the CPU backend and on the GPU backend
// with identical deterministic inputs and reports the normalised error of the GPU
// result against the CPU reference. Pinpoints which op/type a broken Vulkan
// driver (or shader) gets wrong, without needing logcat.
// ---------------------------------------------------------------------------
namespace {

struct SelfTestCase {
    std::string name;
    std::function<ggml_tensor*(ggml_context*, std::vector<ggml_tensor*>&)> build;
};

void fillTensorDeterministic(ggml_tensor* t, uint32_t seed) {
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float> f(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        f[i] = (float) ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; // [-1, 1)
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) n);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

// Returns 1 = computed, 0 = failed, -1 = op not supported by this backend.
int runSelfTestCase(ggml_backend_t backend, const SelfTestCase& c, std::vector<float>& out) {
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) return 0;
    std::vector<ggml_tensor*> inputs;
    ggml_tensor* result = c.build(ctx, inputs);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    int status = 0;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf != nullptr) {
        if (!ggml_backend_supports_op(backend, result)) {
            status = -1;
        } else {
            for (size_t i = 0; i < inputs.size(); i++) {
                fillTensorDeterministic(inputs[i], (uint32_t) (1 + i));
            }
            if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
                out.resize((size_t) ggml_nelements(result));
                ggml_backend_tensor_get(result, out.data(), 0, out.size() * sizeof(float));
                status = 1;
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return status;
}

std::vector<SelfTestCase> makeSelfTestCases() {
    std::vector<SelfTestCase> cases;
    auto unary = [&](const char* name, std::function<ggml_tensor*(ggml_context*, ggml_tensor*)> op) {
        cases.push_back({name, [op](ggml_context* ctx, std::vector<ggml_tensor*>& in) {
            ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 8);
            in = {a};
            return op(ctx, a);
        }});
    };
    auto binary = [&](const char* name, std::function<ggml_tensor*(ggml_context*, ggml_tensor*, ggml_tensor*)> op) {
        cases.push_back({name, [op](ggml_context* ctx, std::vector<ggml_tensor*>& in) {
            ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 8);
            ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 8);
            in = {a, b};
            return op(ctx, a, b);
        }});
    };
    binary("add F32", [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_add(c, a, b); });
    binary("mul F32", [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_mul(c, a, b); });
    unary("silu F32", [](ggml_context* c, ggml_tensor* a) { return ggml_silu(c, a); });
    unary("rms_norm F32", [](ggml_context* c, ggml_tensor* a) { return ggml_rms_norm(c, a, 1e-5f); });
    unary("soft_max F32", [](ggml_context* c, ggml_tensor* a) { return ggml_soft_max(c, a); });

    struct TypeName { ggml_type type; const char* name; };
    const TypeName types[] = {
        {GGML_TYPE_F32, "F32"}, {GGML_TYPE_F16, "F16"}, {GGML_TYPE_Q4_0, "Q4_0"},
        {GGML_TYPE_Q8_0, "Q8_0"}, {GGML_TYPE_Q4_K, "Q4_K"}, {GGML_TYPE_Q6_K, "Q6_K"},
    };
    const int64_t batches[] = {1, 32};  // 1 = mat-vec shaders (generation), 32 = mat-mat (prompt)
    for (const TypeName& tn : types) {
        for (int64_t n : batches) {
            const ggml_type type = tn.type;
            std::string name = std::string("mul_mat ") + tn.name + " N=" + std::to_string(n);
            cases.push_back({name, [type, n](ggml_context* ctx, std::vector<ggml_tensor*>& in) {
                ggml_tensor* w = ggml_new_tensor_2d(ctx, type, 512, 128);        // weights [K=512, M=128]
                ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, n);  // activations [K, N]
                in = {w, x};
                return ggml_mul_mat(ctx, w, x);
            }});
        }
    }
    return cases;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_runGpuSelfTest(JNIEnv* env, jobject thiz) {
    std::string report;
    try {
        ggml_backend_dev_t gpu = nullptr;
        ggml_backend_dev_t cpu = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (gpu == nullptr && (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)) gpu = dev;
            if (cpu == nullptr && type == GGML_BACKEND_DEVICE_TYPE_CPU) cpu = dev;
        }
        if (gpu == nullptr || cpu == nullptr) {
            report = "No GPU (or CPU) backend device is registered.";
        } else {
            char line[256];
            const char* desc = ggml_backend_dev_description(gpu);
            report += std::string("GPU: ") + (desc != nullptr ? desc : "?") + "\n";
            report += "Normalised error of GPU vs CPU (OK < 0.002):\n";
            ggml_backend_t beCpu = ggml_backend_dev_init(cpu, nullptr);
            ggml_backend_t beGpu = ggml_backend_dev_init(gpu, nullptr);
            int nOk = 0, nWarn = 0, nBad = 0, nSkip = 0;
            if (beCpu == nullptr || beGpu == nullptr) {
                report += "Could not initialise a backend.\n";
            } else {
                for (const SelfTestCase& c : makeSelfTestCases()) {
                    std::vector<float> ref, got;
                    const int rc = runSelfTestCase(beCpu, c, ref);
                    const int gc = runSelfTestCase(beGpu, c, got);
                    if (gc < 0 || rc <= 0) {
                        snprintf(line, sizeof(line), "%s: skipped (%s)\n", c.name.c_str(),
                                 gc < 0 ? "GPU does not support this op" : "CPU reference failed");
                        nSkip++;
                    } else if (gc == 0 || got.size() != ref.size()) {
                        snprintf(line, sizeof(line), "%s: BAD (GPU compute failed)\n", c.name.c_str());
                        nBad++;
                    } else {
                        double num = 0.0, den = 0.0;
                        size_t nonFinite = 0;
                        for (size_t i = 0; i < ref.size(); i++) {
                            if (!std::isfinite(got[i])) { nonFinite++; continue; }
                            const double d = (double) got[i] - (double) ref[i];
                            num += d * d;
                            den += (double) ref[i] * (double) ref[i];
                        }
                        const double nmse = den > 0.0 ? num / den : num;
                        const char* verdict = (nonFinite > 0 || nmse > 5e-2) ? "BAD" : (nmse > 2e-3 ? "WARN" : "OK");
                        if (verdict[0] == 'B') nBad++; else if (verdict[0] == 'W') nWarn++; else nOk++;
                        snprintf(line, sizeof(line), "%s: %s (err %.2e%s)\n", c.name.c_str(), verdict, nmse,
                                 nonFinite > 0 ? ", NaN/Inf present" : "");
                    }
                    report += line;
                }
                snprintf(line, sizeof(line), "Summary: %d OK, %d WARN, %d BAD, %d skipped\n", nOk, nWarn, nBad, nSkip);
                report += line;
            }
            if (beGpu != nullptr) ggml_backend_free(beGpu);
            if (beCpu != nullptr) ggml_backend_free(beCpu);
        }
    } catch (const std::exception& e) {
        report += std::string("Self-test aborted: ") + e.what() + "\n";
    } catch (...) {
        report += "Self-test aborted by an unknown error.\n";
    }
    return env->NewStringUTF(report.c_str());
}

// Quick GPU-vs-CPU check of Q6_K matrix multiplication (some drivers, e.g. Turnip on Adreno 6xx,
// compute it wrongly). Returns 1 = correct, 0 = wrong results, -1 = inconclusive (no GPU,
// op unsupported, or compute failed). Used by the app to decide automatically whether Q6_K
// should stay on the CPU.
extern "C" JNIEXPORT jint JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_runQ6kCheck(JNIEnv* env, jobject thiz) {
    int verdict = -1;
    try {
        ggml_backend_dev_t gpu = nullptr;
        ggml_backend_dev_t cpu = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (gpu == nullptr && (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)) gpu = dev;
            if (cpu == nullptr && type == GGML_BACKEND_DEVICE_TYPE_CPU) cpu = dev;
        }
        if (gpu != nullptr && cpu != nullptr) {
            ggml_backend_t beCpu = ggml_backend_dev_init(cpu, nullptr);
            ggml_backend_t beGpu = ggml_backend_dev_init(gpu, nullptr);
            if (beCpu != nullptr && beGpu != nullptr) {
                verdict = 1;
                const int64_t batches[] = {1, 32};  // mat-vec (generation) and mat-mat (prompt) shaders
                for (int64_t n : batches) {
                    SelfTestCase c{"q6k", [n](ggml_context* ctx, std::vector<ggml_tensor*>& in) {
                        ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 512, 128);
                        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, n);
                        in = {w, x};
                        return ggml_mul_mat(ctx, w, x);
                    }};
                    std::vector<float> ref, got;
                    const int rc = runSelfTestCase(beCpu, c, ref);
                    const int gc = runSelfTestCase(beGpu, c, got);
                    if (rc <= 0 || gc <= 0 || got.size() != ref.size()) { verdict = -1; break; }
                    double num = 0.0, den = 0.0;
                    size_t nonFinite = 0;
                    for (size_t i = 0; i < ref.size(); i++) {
                        if (!std::isfinite(got[i])) { nonFinite++; continue; }
                        const double d = (double) got[i] - (double) ref[i];
                        num += d * d;
                        den += (double) ref[i] * (double) ref[i];
                    }
                    const double nmse = den > 0.0 ? num / den : num;
                    if (nonFinite > 0 || nmse > 5e-2) { verdict = 0; break; }
                }
            }
            if (beGpu != nullptr) ggml_backend_free(beGpu);
            if (beCpu != nullptr) ggml_backend_free(beCpu);
        }
    } catch (...) {
        verdict = -1;
    }
    return verdict;
}

// Calls a function the (patched) Vulkan backend publishes through its proc-address table.
static void callVulkanBackendHook(const char* name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        void* fn = reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
        if (fn != nullptr) {
            ((void (*)()) fn)();
            return;
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_nativeSavePipelineCache(JNIEnv* env, jobject thiz) {
    try { callVulkanBackendHook("offlinellm_pipeline_cache_save"); } catch (...) {}
}

extern "C" JNIEXPORT void JNICALL
Java_com_jegly_offlineLLM_smollm_SmolLM_nativeClearPipelineCache(JNIEnv* env, jobject thiz) {
    try { callVulkanBackendHook("offlinellm_pipeline_cache_clear"); } catch (...) {}
}
