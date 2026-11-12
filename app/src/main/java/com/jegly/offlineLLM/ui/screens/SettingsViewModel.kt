package com.jegly.offlineLLM.ui.screens

import android.app.Application
import android.net.Uri
import android.widget.Toast
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.jegly.offlineLLM.ai.GpuDriverInfo
import com.jegly.offlineLLM.ai.GpuDriverManager
import com.jegly.offlineLLM.ai.ModelManager
import com.jegly.offlineLLM.data.local.entities.ModelInfo
import com.jegly.offlineLLM.data.repository.ChatRepository
import com.jegly.offlineLLM.data.repository.ExportData
import com.jegly.offlineLLM.data.repository.ExportedChat
import com.jegly.offlineLLM.data.repository.ExportedMessage
import com.jegly.offlineLLM.data.repository.SettingsRepository
import com.jegly.offlineLLM.ui.theme.ThemeMode
import com.jegly.offlineLLM.utils.FileUtils
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import javax.inject.Inject

data class SettingsUiState(
    val models: List<ModelInfo> = emptyList(),
    val activeModel: ModelInfo? = null,
    val temperature: Float = 0.7f,
    val maxTokens: Int = 2048,
    val contextSize: Int = 4096,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val minP: Float = 0.1f,
    val repeatPenalty: Float = 1.1f,
    val biometricLock: Boolean = false,
    val autoLockOnBackground: Boolean = false,
    val screenshotProtectionEnabled: Boolean = false,
    val tapjackingProtectionEnabled: Boolean = true,
    val sensitiveDataAccessibilityEnabled: Boolean = true,
    val secureStorageBackend: String = "Unknown",
    val systemPromptKey: String = "default",
    val customSystemPrompt: String = "",
    val themeMode: String = "PTYXIS",
    val accentColor: String = "dynamic",
    val disableThinking: Boolean = true,
    val mathLatexHints: Boolean = false,
    val translatorFrom: String = "en",
    val translatorTo: String = "es",
    val catppuccinAccent: String = "mauve",
    val draculaAccent: String = "purple",
    val catppuccinFlavor: String = "mocha",
    val ptyxisPalette: String = "cobalt_neon",
    val monochromeAccents: Boolean = false,
    val appFont: String = "turret_road",
    val fontScale: Float = 1.0f,
    val gpuLayers: Int = 99,
    val useGpu: Boolean = false,
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
    val kvCacheQ8: Boolean = false,
    val numThreads: Int = 4,
    val gpuDeviceName: String = "",
    val backendInfo: String = "",
    val isImportingModel: Boolean = false,
    // Custom Vulkan drivers (libadrenotools). "" = system driver.
    val gpuDrivers: List<GpuDriverInfo> = emptyList(),
    val selectedGpuDriver: String = "",
    // Driver the Vulkan backend actually started with in this process; null = not started yet.
    val appliedGpuDriver: String? = null,
    val gpuDriverAutoDisabled: String? = null,
    val isImportingDriver: Boolean = false,
    // GGML_VK_* debug switches (raw text) and whether they differ from what's running.
    val vulkanEnvVars: String = "",
    val envNeedsRestart: Boolean = false,
    // True when the GPU Acceleration switch differs from its value at app start.
    val gpuToggleNeedsRestart: Boolean = false,
    // Weight types the automatic GPU check moved to the CPU ("" = none).
    val autoUnsupportedTypes: String = "",
    val selfTestRunning: Boolean = false,
    val selfTestReport: String = "",
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val application: Application,
    private val settingsRepository: SettingsRepository,
    private val chatRepository: ChatRepository,
    private val modelManager: ModelManager,
) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(loadState())
    val uiState: StateFlow<SettingsUiState> = _uiState

    init {
        viewModelScope.launch {
            chatRepository.getAllModels().collect { models ->
                val activeId = settingsRepository.activeModelId
                _uiState.update {
                    it.copy(
                        models = models,
                        activeModel = models.find { m -> m.id == activeId }
                    )
                }
            }
        }
        // Off the main thread: first call may initialise the Vulkan instance.
        viewModelScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            val gpuName = modelManager.getGpuDeviceInfo()
            // Same backend registry, already loaded by the call above.
            val backends = modelManager.getBackendInfo()
            val drivers = GpuDriverManager(application).list()
            _uiState.update {
                it.copy(
                    gpuDeviceName = gpuName,
                    backendInfo = backends,
                    gpuDrivers = drivers,
                    selectedGpuDriver = settingsRepository.selectedGpuDriver,
                    appliedGpuDriver = GpuDriverManager.appliedDriverId,
                    gpuDriverAutoDisabled = GpuDriverManager.autoDisabledDriver,
                    envNeedsRestart = envNeedsRestart(),
                    autoUnsupportedTypes = GpuDriverManager.autoUnsupportedTypes ?: "",
                )
            }
        }
    }

    private fun loadState(): SettingsUiState {
        return SettingsUiState(
            temperature = settingsRepository.temperature,
            maxTokens = settingsRepository.maxTokens,
            contextSize = settingsRepository.contextSize,
            topP = settingsRepository.topP,
            topK = settingsRepository.topK,
            minP = settingsRepository.minP,
            repeatPenalty = settingsRepository.repeatPenalty,
            biometricLock = settingsRepository.biometricLock,
            autoLockOnBackground = settingsRepository.autoLockOnBackgroundEnabled,
            screenshotProtectionEnabled = settingsRepository.screenshotProtectionEnabled,
            tapjackingProtectionEnabled = settingsRepository.tapjackingProtectionEnabled,
            sensitiveDataAccessibilityEnabled = settingsRepository.sensitiveDataAccessibilityEnabled,
            secureStorageBackend = settingsRepository.secureStorageBackend,
            systemPromptKey = settingsRepository.systemPromptKey,
            customSystemPrompt = settingsRepository.customSystemPrompt,
            themeMode = settingsRepository.themeMode,
            accentColor = settingsRepository.accentColor,
            disableThinking = settingsRepository.disableThinking,
            mathLatexHints = settingsRepository.mathLatexHints,
            translatorFrom = settingsRepository.translatorFrom,
            translatorTo = settingsRepository.translatorTo,
            catppuccinAccent = settingsRepository.catppuccinAccent,
            draculaAccent = settingsRepository.draculaAccent,
            catppuccinFlavor = settingsRepository.catppuccinFlavor,
            ptyxisPalette = settingsRepository.ptyxisPalette,
            monochromeAccents = settingsRepository.monochromeAccents,
            appFont = settingsRepository.appFont,
            fontScale = settingsRepository.fontScale,
            gpuLayers = settingsRepository.gpuLayers,
            useGpu = settingsRepository.useGpu,
            gpuToggleNeedsRestart = gpuToggleNeedsRestart(settingsRepository.useGpu),
            selectedGpuDriver = settingsRepository.selectedGpuDriver,
            vulkanEnvVars = settingsRepository.vulkanEnvVars,
            useMmap = settingsRepository.useMmap,
            useMlock = settingsRepository.useMlock,
            kvCacheQ8 = settingsRepository.kvCacheQ8,
            numThreads = settingsRepository.numThreads,
        )
    }

    fun setTemperature(value: Float) {
        settingsRepository.temperature = value
        _uiState.update { it.copy(temperature = value) }
    }

    fun setMaxTokens(value: Int) {
        settingsRepository.maxTokens = value
        _uiState.update { it.copy(maxTokens = value) }
    }

    fun setContextSize(value: Int) {
        settingsRepository.contextSize = value
        _uiState.update { it.copy(contextSize = value) }
    }

    fun setTopP(value: Float) {
        settingsRepository.topP = value
        _uiState.update { it.copy(topP = value) }
    }

    fun setTopK(value: Int) {
        settingsRepository.topK = value
        _uiState.update { it.copy(topK = value) }
    }

    fun setMinP(value: Float) {
        settingsRepository.minP = value
        _uiState.update { it.copy(minP = value) }
    }

    fun setRepeatPenalty(value: Float) {
        settingsRepository.repeatPenalty = value
        _uiState.update { it.copy(repeatPenalty = value) }
    }

    fun setBiometricLock(enabled: Boolean) {
        settingsRepository.biometricLock = enabled
        _uiState.update { it.copy(biometricLock = enabled) }
    }

    fun setAutoLockOnBackground(enabled: Boolean) {
        settingsRepository.autoLockOnBackgroundEnabled = enabled
        _uiState.update { it.copy(autoLockOnBackground = enabled) }
    }

    fun setScreenshotProtectionEnabled(enabled: Boolean) {
        settingsRepository.screenshotProtectionEnabled = enabled
        _uiState.update { it.copy(screenshotProtectionEnabled = enabled) }
    }

    fun setTapjackingProtectionEnabled(enabled: Boolean) {
        settingsRepository.tapjackingProtectionEnabled = enabled
        _uiState.update { it.copy(tapjackingProtectionEnabled = enabled) }
    }

    fun setSensitiveDataAccessibilityEnabled(enabled: Boolean) {
        settingsRepository.sensitiveDataAccessibilityEnabled = enabled
        _uiState.update { it.copy(sensitiveDataAccessibilityEnabled = enabled) }
    }

    fun setSystemPrompt(key: String) {
        settingsRepository.systemPromptKey = key
        _uiState.update { it.copy(systemPromptKey = key) }
    }

    fun setCustomSystemPrompt(value: String) {
        settingsRepository.customSystemPrompt = value
        _uiState.update { it.copy(customSystemPrompt = value) }
    }

    fun setTheme(mode: ThemeMode) {
        settingsRepository.themeMode = mode.name
        _uiState.update { it.copy(themeMode = mode.name) }
    }

    fun setAccentColor(key: String) {
        settingsRepository.accentColor = key
        _uiState.update { it.copy(accentColor = key) }
    }

    fun setDisableThinking(disabled: Boolean) {
        settingsRepository.disableThinking = disabled
        _uiState.update { it.copy(disableThinking = disabled) }
    }

    fun setMathLatexHints(enabled: Boolean) {
        settingsRepository.mathLatexHints = enabled
        _uiState.update { it.copy(mathLatexHints = enabled) }
    }

    fun setTranslatorFrom(code: String) {
        settingsRepository.translatorFrom = code
        _uiState.update { it.copy(translatorFrom = code) }
    }

    fun setTranslatorTo(code: String) {
        settingsRepository.translatorTo = code
        _uiState.update { it.copy(translatorTo = code) }
    }

    fun setCatppuccinAccent(key: String) {
        settingsRepository.catppuccinAccent = key
        _uiState.update { it.copy(catppuccinAccent = key) }
    }

    fun setDraculaAccent(key: String) {
        settingsRepository.draculaAccent = key
        _uiState.update { it.copy(draculaAccent = key) }
    }

    fun setCatppuccinFlavor(key: String) {
        settingsRepository.catppuccinFlavor = key
        _uiState.update { it.copy(catppuccinFlavor = key) }
    }

    fun setPtyxisPalette(key: String) {
        settingsRepository.ptyxisPalette = key
        _uiState.update { it.copy(ptyxisPalette = key) }
    }

    fun setMonochromeAccents(enabled: Boolean) {
        settingsRepository.monochromeAccents = enabled
        _uiState.update { it.copy(monochromeAccents = enabled) }
    }

    fun setAppFont(key: String) {
        settingsRepository.appFont = key
        _uiState.update { it.copy(appFont = key) }
    }

    fun setFontScale(value: Float) {
        settingsRepository.fontScale = value
        _uiState.update { it.copy(fontScale = value) }
    }

    fun setGpuLayers(value: Int) {
        settingsRepository.gpuLayers = value
        _uiState.update { it.copy(gpuLayers = value) }
    }

    fun setUseGpu(enabled: Boolean) {
        settingsRepository.useGpu = enabled
        _uiState.update { it.copy(useGpu = enabled, gpuToggleNeedsRestart = gpuToggleNeedsRestart(enabled)) }
    }

    fun importGpuDriver(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(isImportingDriver = true) }
            val result = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                GpuDriverManager(application).importDriver(uri)
            }
            val drivers = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                GpuDriverManager(application).list()
            }
            _uiState.update { it.copy(isImportingDriver = false, gpuDrivers = drivers) }
            result.fold(
                onSuccess = { driver ->
                    Toast.makeText(application, "Driver imported: ${driver.name}", Toast.LENGTH_SHORT).show()
                },
                onFailure = { e ->
                    Toast.makeText(application, "Driver import failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            )
        }
    }

    /** Takes effect after an app restart: Vulkan starts once per process. */
    fun selectGpuDriver(id: String) {
        settingsRepository.selectedGpuDriver = id
        settingsRepository.setGpuDriverTrial("")
        _uiState.update { it.copy(selectedGpuDriver = id, gpuDriverAutoDisabled = null) }
    }

    fun deleteGpuDriver(id: String) {
        viewModelScope.launch {
            val manager = GpuDriverManager(application)
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { manager.delete(id) }
            if (settingsRepository.selectedGpuDriver == id) selectGpuDriver(GpuDriverManager.SYSTEM_DRIVER)
            val drivers = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { manager.list() }
            _uiState.update { it.copy(gpuDrivers = drivers) }
        }
    }

    private fun gpuToggleNeedsRestart(enabled: Boolean): Boolean =
        GpuDriverManager.appliedUseGpu?.let { it != enabled } ?: false

    private fun envNeedsRestart(): Boolean {
        val applied = GpuDriverManager.appliedEnv ?: return false
        return applied != GpuDriverManager.normalizeEnv(settingsRepository.vulkanEnvVars)
    }

    /** GGML_VK_* debug switches; take effect after an app restart. */
    fun setVulkanEnvVars(text: String) {
        settingsRepository.vulkanEnvVars = text
        _uiState.update { it.copy(vulkanEnvVars = text, envNeedsRestart = envNeedsRestart()) }
    }

    fun runGpuSelfTest() {
        if (_uiState.value.selfTestRunning) return
        _uiState.update { it.copy(selfTestRunning = true, selfTestReport = "") }
        viewModelScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            val report = modelManager.runGpuSelfTest()
            _uiState.update { it.copy(selfTestRunning = false, selfTestReport = report) }
        }
    }

    fun restartApp() {
        val launch = application.packageManager.getLaunchIntentForPackage(application.packageName) ?: return
        val component = launch.component ?: return
        application.startActivity(android.content.Intent.makeRestartActivityTask(component))
        Runtime.getRuntime().exit(0)
    }

    fun setUseMmap(enabled: Boolean) {
        settingsRepository.useMmap = enabled
        _uiState.update { it.copy(useMmap = enabled) }
    }

    fun setUseMlock(enabled: Boolean) {
        settingsRepository.useMlock = enabled
        _uiState.update { it.copy(useMlock = enabled) }
    }

    fun setKvCacheQ8(enabled: Boolean) {
        settingsRepository.kvCacheQ8 = enabled
        _uiState.update { it.copy(kvCacheQ8 = enabled) }
    }

    fun setNumThreads(value: Int) {
        settingsRepository.numThreads = value
        _uiState.update { it.copy(numThreads = value) }
    }

    fun selectModel(modelId: Long) {
        settingsRepository.activeModelId = modelId
        _uiState.update { state ->
            state.copy(activeModel = state.models.find { it.id == modelId })
        }
    }

    fun deleteModel(modelId: Long) {
        viewModelScope.launch {
            modelManager.deleteModel(modelId)
        }
    }

    fun importModel(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(isImportingModel = true) }
            val result = modelManager.importModel(uri)
            _uiState.update { it.copy(isImportingModel = false) }
            result.fold(
                onSuccess = { model ->
                    Toast.makeText(application, "Model imported: ${model.name}", Toast.LENGTH_SHORT).show()
                },
                onFailure = { e ->
                    Toast.makeText(application, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            )
        }
    }

    fun exportChats(uri: Uri) {
        viewModelScope.launch {
            try {
                val json = Json { prettyPrint = true }
                val exportData = ExportData(chats = emptyList())
                val jsonString = json.encodeToString(exportData)
                FileUtils.writeTextToUri(application, uri, jsonString)
                Toast.makeText(application, "Chats exported", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(application, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    fun importChats(uri: Uri) {
        viewModelScope.launch {
            try {
                val jsonString = FileUtils.readTextFromUri(application, uri)
                if (jsonString != null) {
                    val result = chatRepository.importChatsFromJson(jsonString)
                    result.fold(
                        onSuccess = { count ->
                            Toast.makeText(application, "Imported $count chats", Toast.LENGTH_SHORT).show()
                        },
                        onFailure = { e ->
                            Toast.makeText(application, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    )
                }
            } catch (e: Exception) {
                Toast.makeText(application, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    fun clearAllChats() {
        viewModelScope.launch {
            chatRepository.deleteAllConversations()
            Toast.makeText(application, "All chats cleared", Toast.LENGTH_SHORT).show()
        }
    }
}
