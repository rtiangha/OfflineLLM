package com.jegly.offlineLLM.ai

import com.jegly.offlineLLM.smollm.SmolLM
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.time.measureTime

class InferenceEngine {

    private val instance = SmolLM()
    private val stateLock = ReentrantLock()

    @Volatile
    private var generationJob: Job? = null

    @Volatile
    private var loadJob: Job? = null

    val isModelLoaded = AtomicBoolean(false)

    @Volatile
    var isGenerating = false

    /** What the loaded model is actually running on. */
    enum class BackendMode { NONE, CPU, GPU, CPU_FALLBACK, CPU_NO_GPU }

    data class BackendStatus(
        val mode: BackendMode = BackendMode.NONE,
        val gpuLayers: Int = 0,
        val detail: String = "",
    )

    private val _backendStatus = kotlinx.coroutines.flow.MutableStateFlow(BackendStatus())
    val backendStatus: kotlinx.coroutines.flow.StateFlow<BackendStatus> = _backendStatus

    data class GenerationResult(
        val response: String,
        val tokensPerSecond: Float,
        val durationSeconds: Int,
        val contextLengthUsed: Int,
    )

    /** Loads ggml backend plugins (best CPU variant + Vulkan) from the app's lib dir. */
    fun initBackends(nativeLibDir: String) = instance.loadBackends(nativeLibDir)

    /** Picks the custom Vulkan driver (or none) for the GPU backend; call before [initBackends]. */
    fun configureCustomDriver(tmpDir: String?, hookDir: String?, driverDir: String?, driverName: String?) =
        instance.configureCustomDriver(tmpDir, hookDir, driverDir, driverName)

    /** GPU device description from the ggml backend registry, or "" if none. */
    fun getGpuDeviceInfo(): String = instance.getGpuDeviceInfo()

    /** Registered backends and the CPU feature variant selected for this device. */
    fun getBackendInfo(): String = instance.getBackendInfo()

    /** CPU-vs-GPU op comparison report (diagnostic). */
    fun gpuSelfTest(): String = instance.gpuSelfTest()

    /** 1 = Q6_K correct on the GPU, 0 = wrong, -1 = inconclusive. */
    fun q6kGpuCheck(): Int = instance.q6kGpuCheck()

    /** Writes the Vulkan pipeline cache to disk if it changed. */
    fun savePipelineCache() = instance.savePipelineCache()

    /** Deletes the Vulkan pipeline cache file and stops saving until the app restarts. */
    fun clearPipelineCache() = instance.clearPipelineCache()

    fun loadModel(
        modelPath: String,
        params: SmolLM.InferenceParams = SmolLM.InferenceParams(),
        systemPrompt: String = "",
        conversationHistory: List<Pair<String, String>> = emptyList(),
        onSuccess: () -> Unit,
        onError: (Exception) -> Unit,
    ) {
        stateLock.withLock {
            loadJob?.cancel()

            loadJob = CoroutineScope(Dispatchers.Default).launch {
                try {
                    var gpuFallbackReason: String? = null
                    try {
                        instance.load(modelPath, params)
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Exception) {
                        // GPU offload can fail on flaky Vulkan drivers or when VRAM
                        // allocation is refused — retry once fully on CPU before
                        // giving up, so the app stays usable with GPU toggled on.
                        if (params.nGpuLayers > 0) {
                            android.util.Log.w(
                                "InferenceEngine",
                                "GPU-accelerated load failed (${e.message}); retrying on CPU"
                            )
                            gpuFallbackReason = e.message ?: "unknown error"
                            instance.load(modelPath, params.copy(nGpuLayers = 0))
                        } else {
                            throw e
                        }
                    }

                    if (systemPrompt.isNotBlank()) {
                        instance.addSystemPrompt(systemPrompt)
                    }

                    for ((role, content) in conversationHistory) {
                        instance.addChatMessage(role, content)
                    }

                    _backendStatus.value = when {
                        gpuFallbackReason != null ->
                            BackendStatus(BackendMode.CPU_FALLBACK, 0, gpuFallbackReason)
                        params.nGpuLayers <= 0 -> BackendStatus(BackendMode.CPU)
                        instance.getGpuDeviceInfo().isEmpty() -> BackendStatus(BackendMode.CPU_NO_GPU)
                        else -> BackendStatus(BackendMode.GPU, params.nGpuLayers)
                    }

                    withContext(Dispatchers.Main) {
                        isModelLoaded.set(true)
                        onSuccess()
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        onError(e)
                    }
                }
            }
        }
    }

    suspend fun unloadModel() {
        val jobToJoin: Job?
        stateLock.withLock {
            loadJob?.cancel()
            jobToJoin = generationJob
            isModelLoaded.set(false)
            _backendStatus.value = BackendStatus()
            isGenerating = false
        }
        // Signal the native side to stop first — this makes the blocking
        // completionLoop() JNI call return promptly instead of waiting for
        // the next token, so the coroutine cancellation can take effect.
        try { instance.stop() } catch (_: Exception) {}
        jobToJoin?.cancel()
        // Now wait for the generation coroutine to fully exit before freeing
        // native memory. 5 s timeout as a safeguard against a stuck JNI call.
        withTimeoutOrNull(5_000) {
            try { jobToJoin?.join() } catch (_: Exception) {}
        }
        instance.savePipelineCache()
        try { instance.close() } catch (_: Exception) {}
    }

    fun generateResponse(
        query: String,
        onToken: (String) -> Unit,
        onComplete: (GenerationResult) -> Unit,
        onCancelled: () -> Unit,
        onError: (Exception) -> Unit,
    ) {
        stateLock.withLock {
            if (!isModelLoaded.get()) {
                onError(IllegalStateException("Model not loaded"))
                return
            }

            generationJob?.cancel()

            generationJob = CoroutineScope(Dispatchers.Default).launch {
                try {
                    isGenerating = true
                    val fullResponse = StringBuilder()
                    var stopRequested = false
                    var tokensSinceLastEmit = 0

                    val duration = measureTime {
                        instance.getResponseAsFlow(query).collect { piece ->
                            if (stopRequested) return@collect

                            fullResponse.append(piece)

                            // Only scan the tail for stop sequences — avoids full-string scan every token
                            val tail = if (fullResponse.length > 200) fullResponse.substring(fullResponse.length - 200) else fullResponse.toString()
                            if (containsStopSequence(tail)) {
                                stopRequested = true
                                return@collect
                            }

                            // Batch UI updates every 3 tokens to reduce Main thread dispatches
                            // and Compose recompositions without visible latency
                            tokensSinceLastEmit++
                            if (tokensSinceLastEmit >= 3) {
                                tokensSinceLastEmit = 0
                                val displayResponse = cleanModelOutput(fullResponse.toString(), isFinal = false)
                                if (displayResponse.isNotBlank() || fullResponse.isEmpty()) {
                                    withContext(Dispatchers.Main) {
                                        onToken(displayResponse)
                                    }
                                }
                            }
                        }
                    }

                    // Final cleanup
                    val fullResponseStr = fullResponse.toString()
                    val finalResponse = cleanModelOutput(fullResponseStr, isFinal = true).let {
                        if (it.isBlank()) {
                            if (fullResponseStr.isNotBlank()) "(No visible content produced)" else "(Empty response)"
                        } else it
                    }

                    // Pipelines compile on first use, so persist them now that a response has finished.
                    instance.savePipelineCache()

                    withContext(Dispatchers.Main) {
                        isGenerating = false
                        onComplete(
                            GenerationResult(
                                response = finalResponse,
                                tokensPerSecond = instance.getResponseGenerationSpeed(),
                                durationSeconds = duration.inWholeSeconds.toInt(),
                                contextLengthUsed = instance.getContextLengthUsed(),
                            )
                        )
                    }
                } catch (e: CancellationException) {
                    isGenerating = false
                    withContext(Dispatchers.Main) { onCancelled() }
                } catch (e: Exception) {
                    isGenerating = false
                    withContext(Dispatchers.Main) { onError(e) }
                }
            }
        }
    }

    private fun containsStopSequence(text: String): Boolean {
        val stops = listOf("<turn|", "<|turn_end|>", "<turn_end|>", "<start_of_turn>", "<end_of_turn>")
        return stops.any { text.contains(it, ignoreCase = true) }
    }

    private fun cleanModelOutput(raw: String, isFinal: Boolean): String {
        var cleaned = raw
            .replace(Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL), "")
            .replace(Regex("<think>.*", RegexOption.DOT_MATCHES_ALL), "")
            .replace(Regex("<turn\\|.*?\\|>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<\\|turn_end\\|>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<turn_end\\|>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<start_of_turn>.*?\\n", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<end_of_turn>", RegexOption.IGNORE_CASE), "")
            .replace("System instruction:", "")

        if (!isFinal) {
            // Buffer to prevent tag fragments from flickering
            val potentialTagStart = listOf("<turn", "<|turn", "<|", "<start", "<")
            for (p in potentialTagStart) {
                if (cleaned.endsWith(p, ignoreCase = true)) {
                    return cleaned.substring(0, cleaned.length - p.length).trim()
                }
            }
        } else {
            // Final pass to remove any trailing tag fragments
            cleaned = cleaned.replace(Regex("<.*$", RegexOption.IGNORE_CASE), "")
        }

        return cleaned.trim()
    }

    fun stopGeneration() {
        try { instance.stop() } catch (_: Exception) {}
        stateLock.withLock {
            generationJob?.cancel()
            isGenerating = false
        }
    }

    fun isReady(): Boolean = isModelLoaded.get() && !isGenerating
}
