package com.jegly.offlineLLM.ui.components

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.text.font.FontWeight
import com.jegly.offlineLLM.ai.GpuDriverManager
import com.jegly.offlineLLM.ai.InferenceEngine

/**
 * Small label showing what the loaded model is actually running on: the GPU (Vulkan) or the CPU,
 * and why when the CPU is used although the GPU switch is on. Draws nothing while no model is loaded.
 */
@Composable
fun BackendIndicator(status: InferenceEngine.BackendStatus) {
    val scheme = MaterialTheme.colorScheme
    val (text, color) = when (status.mode) {
        InferenceEngine.BackendMode.GPU -> gpuLabel(status) to scheme.primary
        InferenceEngine.BackendMode.CPU -> "CPU" to scheme.onSurfaceVariant
        InferenceEngine.BackendMode.CPU_FALLBACK -> "CPU (GPU load failed)" to scheme.error
        InferenceEngine.BackendMode.CPU_NO_GPU -> "CPU (no usable GPU)" to scheme.tertiary
        InferenceEngine.BackendMode.NONE -> return
    }
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.Bold,
        color = color,
    )
}

private fun gpuLabel(status: InferenceEngine.BackendStatus): String = buildString {
    append("Vulkan")
    if (!GpuDriverManager.appliedDriverId.isNullOrEmpty()) append(" \u00B7 custom driver")
    if (status.gpuLayers in 1..98) append(" \u00B7 ${status.gpuLayers} layers")
}
