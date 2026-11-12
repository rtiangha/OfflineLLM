package com.jegly.offlineLLM.ai

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

/** One imported driver package (Adrenotools "ADPKG" format: meta.json + .so files). */
data class GpuDriverInfo(
    val id: String,
    val name: String,
    val version: String,
    val vendor: String,
    val libraryName: String,
)

/**
 * Imports, lists, resolves and deletes custom Vulkan driver packages (e.g. Mesa Turnip
 * builds) stored under filesDir/gpu_drivers/<id>/. The directory must stay in the app's
 * private storage: libadrenotools refuses to dlopen drivers from shared storage.
 */
class GpuDriverManager(private val context: Context) {

    /** A driver ready to hand to libadrenotools. [dirPath] always ends with '/'. */
    data class Selection(val dirPath: String, val libraryName: String)

    private val root = File(context.filesDir, "gpu_drivers")

    fun list(): List<GpuDriverInfo> =
        root.listFiles { f -> f.isDirectory && ID_REGEX.matches(f.name) }
            ?.mapNotNull { readInfo(it) }
            ?.sortedBy { it.name.lowercase() }
            ?: emptyList()

    /** Returns null if the id is malformed, the package is gone, or its main library is missing. */
    fun resolve(id: String): Selection? {
        if (!ID_REGEX.matches(id)) return null
        val dir = File(root, id)
        val info = readInfo(dir) ?: return null
        if (!File(dir, info.libraryName).isFile) return null
        return Selection(dir.absolutePath + "/", info.libraryName)
    }

    fun delete(id: String): Boolean {
        if (!ID_REGEX.matches(id)) return false
        val dir = File(root, id)
        return dir.isDirectory && dir.deleteRecursively()
    }

    /** Imports a driver .zip chosen through the system file picker. */
    fun importDriver(uri: Uri): Result<GpuDriverInfo> = runCatching {
        root.mkdirs()
        val tmp = File(root, ".tmp-${System.nanoTime()}")
        check(tmp.mkdirs()) { "Could not create a temporary folder" }
        try {
            val input = context.contentResolver.openInputStream(uri)
                ?: error("Could not read the selected file")
            var files = 0
            var total = 0L
            ZipInputStream(input.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) continue
                    // Flatten: keep only the file name, so no entry can escape the folder.
                    val name = entry.name.substringAfterLast('/')
                    if (name != "meta.json" && !name.endsWith(".so")) continue
                    require(FILE_NAME_REGEX.matches(name)) { "Unexpected file name in package: $name" }
                    require(++files <= MAX_FILES) { "Package has too many files" }
                    FileOutputStream(File(tmp, name)).use { out ->
                        val buf = ByteArray(64 * 1024)
                        while (true) {
                            val n = zip.read(buf)
                            if (n < 0) break
                            total += n
                            require(total <= MAX_TOTAL_BYTES) { "Package is too large" }
                            out.write(buf, 0, n)
                        }
                    }
                }
            }

            val meta = File(tmp, "meta.json")
            require(meta.isFile) { "Not a driver package: meta.json not found" }
            val info = parseMeta(meta, id = "")
                ?: error("meta.json is invalid or has no libraryName")
            val lib = File(tmp, info.libraryName)
            require(lib.isFile) { "Driver library ${info.libraryName} is missing from the package" }
            require(isElf(lib)) { "${info.libraryName} is not a valid shared library" }

            val id = makeId(info)
            val dest = File(root, id)
            if (dest.exists()) dest.deleteRecursively()
            check(tmp.renameTo(dest)) { "Could not install the driver" }
            dest.listFiles()?.forEach { if (it.name.endsWith(".so")) it.setReadOnly() }
            info.copy(id = id)
        } finally {
            if (tmp.exists()) tmp.deleteRecursively()
        }
    }

    private fun readInfo(dir: File): GpuDriverInfo? =
        File(dir, "meta.json").takeIf { it.isFile }?.let { parseMeta(it, dir.name) }

    private fun parseMeta(file: File, id: String): GpuDriverInfo? = try {
        val json = JSONObject(file.readText())
        val lib = json.optString("libraryName", "")
        if (!FILE_NAME_REGEX.matches(lib) || !lib.endsWith(".so")) {
            null
        } else {
            GpuDriverInfo(
                id = id,
                name = json.optString("name", lib).ifBlank { lib },
                version = json.optString("driverVersion", ""),
                vendor = json.optString("vendor", ""),
                libraryName = lib,
            )
        }
    } catch (e: Exception) {
        null
    }

    private fun makeId(info: GpuDriverInfo): String {
        val slug = "${info.name}-${info.version}".lowercase()
            .replace(Regex("[^a-z0-9._-]+"), "-")
            .trim('-', '.')
            .take(80)
        return if (ID_REGEX.matches(slug)) slug else "driver-${info.libraryName.hashCode().toUInt()}"
    }

    private fun isElf(file: File): Boolean = try {
        val magic = ByteArray(4)
        file.inputStream().use { it.read(magic) } == 4 &&
            magic[0] == 0x7F.toByte() && magic[1] == 'E'.code.toByte() &&
            magic[2] == 'L'.code.toByte() && magic[3] == 'F'.code.toByte()
    } catch (e: Exception) {
        false
    }

    companion object {
        /** Value of the "selected driver" setting that means "use the device's own driver". */
        const val SYSTEM_DRIVER = ""

        private val ID_REGEX = Regex("[a-z0-9][a-z0-9._-]{0,79}")
        private val FILE_NAME_REGEX = Regex("[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
        private const val MAX_FILES = 64
        private const val MAX_TOTAL_BYTES = 400L * 1024 * 1024

        /**
         * Driver id the native Vulkan backend was started with in this process
         * ("" = system driver); null until the backends have been initialised.
         * Vulkan starts once per process, so changing the driver needs an app restart.
         */
        @Volatile var appliedDriverId: String? = null

        /**
         * Normalised GGML_VK_* environment text the Vulkan backend was started with in this
         * process; null until the backends have been initialised.
         */
        @Volatile var appliedEnv: String? = null

        /**
         * Value of the "GPU Acceleration (Vulkan)" setting when this process started
         * (null until ModelManager has been created); Settings compares against it to offer a restart.
         */
        @Volatile var appliedUseGpu: Boolean? = null

        /** Value of the "Vulkan Cache" setting when this process started (null until ModelManager exists). */
        @Volatile var appliedUseVkCache: Boolean? = null

        /**
         * Weight types the automatic GPU check moved to the CPU in this process ("" = none,
         * "Q6_K" = Q6_K); null until the check has had its chance to run.
         */
        @Volatile var autoUnsupportedTypes: String? = null

        private val ENV_NAME_REGEX = Regex("(GGML_VK|OFFLINELLM)_[A-Z0-9_]{1,60}")

        /**
         * Parses user text (one NAME=VALUE per line; blank lines and #comments ignored) into
         * pairs. Only GGML_VK_* names are accepted, so this can't touch unrelated process state.
         */
        fun parseEnv(text: String): List<Pair<String, String>> =
            text.lines().mapNotNull { raw ->
                val line = raw.trim()
                if (line.isEmpty() || line.startsWith("#")) return@mapNotNull null
                val eq = line.indexOf('=')
                if (eq <= 0) return@mapNotNull null
                val name = line.substring(0, eq).trim()
                val value = line.substring(eq + 1).trim()
                if (ENV_NAME_REGEX.matches(name)) name to value else null
            }

        fun normalizeEnv(text: String): String =
            parseEnv(text).joinToString("\n") { "${it.first}=${it.second}" }

        /** Set when a driver was switched off because the previous launch died while loading it. */
        @Volatile var autoDisabledDriver: String? = null
    }
}
