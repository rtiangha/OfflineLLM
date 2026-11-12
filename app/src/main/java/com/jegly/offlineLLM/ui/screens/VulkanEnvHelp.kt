package com.jegly.offlineLLM.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/** Human-readable size for the Vulkan cache indicator. */
fun formatCacheSize(bytes: Long): String = when {
    bytes < 1024L -> "$bytes B"
    bytes < 1024L * 1024L -> String.format(java.util.Locale.US, "%.1f KB", bytes / 1024.0)
    else -> String.format(java.util.Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0))
}

/** One variable shown in the help dialog. [exampleValue] is what tapping it inserts. */
data class EnvVarHelp(
    val name: String,
    val description: String,
    val takesValue: Boolean = false,
    val exampleValue: String = "1",
)

data class EnvVarGroup(val title: String, val items: List<EnvVarHelp>)

/** Offline help for the Vulkan environment variables box (kept in sync with scripts/patch_ggml_vulkan.py by hand). */
object VulkanEnvHelp {

    val groups: List<EnvVarGroup> = listOf(
        EnvVarGroup(
            "Added For This App",
            listOf(
                EnvVarHelp("GGML_VK_UNSUPPORTED_TYPES", "Comma-separated weight types that must stay on the CPU, e.g. Q6_K. Fixes Turnip's wrong Q6_K results; this is what Safe Preset sets.", takesValue = true, exampleValue = "Q6_K"),
                EnvVarHelp("OFFLINELLM_VK_ENABLE_BDA", "Turns the buffer device address feature on. Off by default because it crashed the system driver and Turnip."),
                EnvVarHelp("OFFLINELLM_VK_ALLOW_OLD_API", "Stops skipping GPUs below Vulkan 1.2. By default they are skipped so the app falls back to the CPU instead of crashing."),
                EnvVarHelp("OFFLINELLM_VK_NO_DEVICE_INIT", "Turns off loading Vulkan function pointers per device after the device is created."),
                EnvVarHelp("OFFLINELLM_VK_TRACE", "Logs breadcrumbs while buffers are created (adb logcat tag ggml-vulkan) to find which step crashes."),
                EnvVarHelp("OFFLINELLM_VK_NO_PIPELINE_CACHE", "Forces the saved pipeline cache off, even when the Vulkan Cache switch is on."),
                EnvVarHelp("OFFLINELLM_NO_AUTO_QUIRKS", "Skips the automatic check that moves Q6_K to the CPU when the GPU driver computes it incorrectly."),
                EnvVarHelp("OFFLINELLM_DISABLE_FLASH_ATTN", "Disables flash attention and the Q8 KV cache. Try it if the GPU output is garbage."),
            ),
        ),
        EnvVarGroup(
            "Likely Useful On A Phone",
            listOf(
                EnvVarHelp("GGML_VK_DISABLE_F16", "Avoid FP16 shaders."),
                EnvVarHelp("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "Avoid integer dot-product shaders."),
                EnvVarHelp("GGML_VK_DISABLE_MMVQ", "Turn off the quantized matrix-vector path."),
                EnvVarHelp("GGML_VK_FORCE_MMVQ", "Force the quantized matrix-vector path on."),
                EnvVarHelp("GGML_VK_DISABLE_FUSION", "No operation fusion."),
                EnvVarHelp("GGML_VK_DISABLE_MULTI_ADD", "No fused multi-add."),
                EnvVarHelp("GGML_VK_DISABLE_GRAPH_OPTIMIZE", "No graph reordering."),
                EnvVarHelp("GGML_VK_DISABLE_ASYNC", "Turn off asynchronous work."),
                EnvVarHelp("GGML_VK_DISABLE_DESCRIPTOR_REUSE", "Do not reuse descriptor sets."),
                EnvVarHelp("GGML_VK_SERIALIZE_SUBMISSIONS", "Wait for each GPU submission to finish before the next one (for debugging sync problems)."),
                EnvVarHelp("GGML_VK_MAX_NODES_PER_SUBMIT", "Graph nodes per GPU submission. Default is 100.", takesValue = true, exampleValue = "100"),
                EnvVarHelp("GGML_VK_PREFER_HOST_MEMORY", "Allocate in system memory instead of device-local memory."),
                EnvVarHelp("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "Do not use device memory that the CPU can see."),
                EnvVarHelp("GGML_VK_ALLOW_SYSMEM_FALLBACK", "Allow falling back to system memory when GPU memory runs out."),
                EnvVarHelp("GGML_VK_FORCE_MAX_ALLOCATION_SIZE", "Caps a single memory allocation, in bytes.", takesValue = true, exampleValue = "1073741824"),
                EnvVarHelp("GGML_VK_FORCE_MAX_BUFFER_SIZE", "Caps a single buffer size, in bytes.", takesValue = true, exampleValue = "1073741824"),
                EnvVarHelp("GGML_VK_SUBALLOCATION_BLOCK_SIZE", "Size of sub-allocated memory blocks, in bytes.", takesValue = true, exampleValue = "268435456"),
                EnvVarHelp("GGML_VK_ENABLE_MEMORY_PRIORITY", "Use the memory-priority extension."),
                EnvVarHelp("GGML_VK_ALLOW_GRAPHICS_QUEUE", "Use a graphics-capable queue for compute (avoided by default)."),
                EnvVarHelp("GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "Use a dedicated transfer queue for async copies."),
            ),
        ),
        EnvVarGroup(
            "Probably Irrelevant On Adreno",
            listOf(
                EnvVarHelp("GGML_VK_DISABLE_COOPMAT", "Turn off cooperative-matrix shaders."),
                EnvVarHelp("GGML_VK_DISABLE_COOPMAT2", "Turn off the second-generation cooperative-matrix shaders."),
                EnvVarHelp("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR", "Turn off the cooperative-matrix decode-vector extension."),
                EnvVarHelp("GGML_VK_DISABLE_BFLOAT16", "Turn off bf16 shaders."),
                EnvVarHelp("GGML_VK_DISABLE_DOT2", "Turn off the mixed-float dot-product extension."),
                EnvVarHelp("GGML_VK_DISABLE_OCP_FP4", "Turn off FP4 microscaling."),
                EnvVarHelp("GGML_VK_FA_SPARSE_DISABLE", "Turn off sparse masking in flash attention."),
                EnvVarHelp("GGML_VK_VISIBLE_DEVICES", "Comma-separated device indices to use.", takesValue = true, exampleValue = "0"),
                EnvVarHelp("GGML_VK_DISABLE_ALLREDUCE", "Multi-GPU: turn off all-reduce."),
                EnvVarHelp("GGML_VK_ALLREDUCE_SPIN_US", "Multi-GPU: spin time in microseconds.", takesValue = true, exampleValue = "100"),
            ),
        ),
        EnvVarGroup(
            "Diagnostics",
            listOf(
                EnvVarHelp("GGML_VK_PERF_LOGGER", "Per-operation timing."),
                EnvVarHelp("GGML_VK_PERF_LOGGER_CONCURRENT", "Per-operation timing, including concurrent work."),
                EnvVarHelp("GGML_VK_PERF_LOGGER_FREQUENCY", "How often the timing is reported.", takesValue = true, exampleValue = "10"),
                EnvVarHelp("GGML_VK_MEMORY_LOGGER", "Log memory allocations."),
                EnvVarHelp("GGML_VK_SYNC_LOGGER", "Log synchronisation."),
                EnvVarHelp("GGML_VK_PIPELINE_STATS", "Pipeline statistics for pipelines whose name matches the filter.", takesValue = true, exampleValue = "mul_mat"),
                EnvVarHelp("GGML_VK_DEBUG_MARKERS", "Add debug markers for GPU debugging tools."),
            ),
        ),
    )

    const val FOOTER =
        "These logs go through ggml's logger and may not appear in adb logcat. " +
            "Names that don't start with GGML_VK_ or OFFLINELLM_ (such as GGML_VULKAN_SKIP_CHECKS) are rejected by the app. " +
            "Changes apply after an app restart."

    /** Adds NAME=value to [text], replacing an existing line for the same name. */
    fun insertLine(text: String, name: String, value: String): String {
        val lines = text.lines().filter { it.isNotBlank() }.toMutableList()
        val index = lines.indexOfFirst { it.substringBefore('=').trim() == name }
        val line = "$name=$value"
        if (index >= 0) lines[index] = line else lines.add(line)
        return lines.joinToString("\n")
    }
}

@Composable
fun VulkanEnvHelpDialog(
    envText: String,
    onInsert: (name: String, value: String) -> Unit,
    onDismiss: () -> Unit,
) {
    val present = envText.lines().map { it.substringBefore('=').trim() }.toSet()
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Vulkan Environment Variables") },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                Text(
                    "Tap a variable to add it to the box. Variables that take a value are added with an example value you can edit.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                VulkanEnvHelp.groups.forEach { group ->
                    Text(
                        group.title,
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
                    )
                    group.items.forEach { item ->
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { onInsert(item.name, item.exampleValue) }
                                .padding(vertical = 6.dp),
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    item.name,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontFamily = FontFamily.Monospace,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f, fill = false),
                                )
                                if (item.name in present) {
                                    Text("  ✓ added", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                                }
                            }
                            val extra = if (item.takesValue) " Takes a value (example: ${item.exampleValue})." else ""
                            Text(
                                item.description + extra,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
                Text(
                    VulkanEnvHelp.FOOTER,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 16.dp),
                )
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
    )
}
