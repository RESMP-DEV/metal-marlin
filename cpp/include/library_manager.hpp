#pragma once

#include "metal_device.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace metal_marlin {

/**
 * @brief Exception raised when metallib file does not exist.
 */
class MetallibNotFoundError : public std::runtime_error {
public:
    explicit MetallibNotFoundError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Exception raised when metallib fails to load.
 */
class MetallibLoadError : public std::runtime_error {
public:
    explicit MetallibLoadError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Version information extracted from a metallib file.
 */
struct MetallibVersion {
    std::string path;
    double build_date;         // Unix timestamp
    size_t size_bytes;
    std::optional<std::string> git_hash;
    std::optional<size_t> shader_count;
    std::optional<std::string> metal_version;
    bool is_stale = false;     // Whether the metallib needs rebuild
};

/**
 * @brief Staleness details for debugging rebuild decisions.
 */
struct StalenessDetails {
    std::string metallib_path;
    bool metallib_exists;
    bool is_stale;
    std::string reason;
    std::vector<std::string> added_files;
    std::vector<std::string> removed_files;
    std::vector<std::string> modified_files;
    bool has_manifest;
};

/**
 * @brief Checksum manifest for tracking source file changes.
 */
struct ChecksumManifest {
    int version = 1;
    std::string metallib_name;
    size_t file_count = 0;
    std::unordered_map<std::string, std::string> checksums;  // path -> sha256
};

/**
 * @brief Manages loading and caching of precompiled Metal libraries (.metallib).
 *
 * LibraryManager provides C++ equivalents to Python's metallib_loader.py:
 * - Load precompiled .metallib files (100-1000x faster than JIT)
 * - Module-level caching for repeated access
 * - Kernel function retrieval by name
 * - Checksum-based staleness detection for rebuild decisions
 * - Source file checksum computation
 *
 * Thread-safe for concurrent access. Uses reference counting for library objects.
 *
 * Usage:
 *   auto& manager = LibraryManager::instance();
 *   auto library = manager.load_metallib("path/to/lib.metallib");
 *   auto kernel = manager.get_kernel("my_kernel", library);
 *
 *   // Check if rebuild is needed
 *   auto details = manager.get_staleness_details("path/to/lib.metallib");
 *   if (details.is_stale) {
 *       // Trigger rebuild
 *   }
 */
class LibraryManager {
public:
    /**
     * @brief Get the singleton instance of LibraryManager.
     * @return Reference to the global LibraryManager.
     */
    static LibraryManager& instance();

    // Non-copyable, non-movable singleton
    LibraryManager(const LibraryManager&) = delete;
    LibraryManager& operator=(const LibraryManager&) = delete;
    LibraryManager(LibraryManager&&) = delete;
    LibraryManager& operator=(LibraryManager&&) = delete;

    ~LibraryManager() = default;

    /**
     * @brief Load a precompiled Metal library from a .metallib file.
     *
     * Uses caching - subsequent calls with the same path return the cached
     * library. Call clear_cache() to force reload.
     *
     * @param path Path to .metallib file. If empty, uses default location.
     * @return MTLLibrary pointer (owned by manager, valid for lifetime).
     * @throws MetallibNotFoundError if metallib file doesn't exist.
     * @throws MetallibLoadError if metallib fails to load.
     */
    MTL::Library* load_metallib(const std::string& path = "");

    /**
     * @brief Get cached precompiled library, loading if needed.
     *
     * Unlike load_metallib(), this returns nullptr on failure instead of
     * throwing, allowing fallback to JIT compilation.
     *
     * @param path Path to .metallib file. If empty, uses default location.
     * @return MTLLibrary pointer or nullptr on failure.
     */
    MTL::Library* get_precompiled_library(const std::string& path = "");

    /**
     * @brief Get kernel function from a metallib.
     *
     * @param kernel_name Name of kernel function.
     * @param library MTLLibrary from load_metallib(), or nullptr to use cached.
     * @return MTLFunction pointer or nullptr if not found.
     */
    MTL::Function* get_kernel(const std::string& kernel_name,
                             MTL::Library* library = nullptr);

    /**
     * @brief Get version information from a metallib.
     *
     * @param path Path to .metallib file. If empty, uses default.
     * @return MetallibVersion with extracted info.
     */
    MetallibVersion get_metallib_version(const std::string& path = "");

    /**
     * @brief Check if metallib is stale by comparing source file checksums.
     *
     * Uses SHA-256 checksums for reliable staleness detection across
     * different filesystems and git operations. Falls back to mtime if no
     * checksum manifest exists.
     *
     * @param path Path to .metallib file. If empty, uses default.
     * @return True if metallib should be rebuilt, False otherwise.
     */
    bool is_metallib_stale(const std::string& path = "");

    /**
     * @brief Get detailed staleness information for debugging.
     *
     * @param path Path to .metallib file. If empty, uses default.
     * @return StalenessDetails with comprehensive information.
     */
    StalenessDetails get_staleness_details(const std::string& path = "");

    /**
     * @brief Clear cached library (for hot-reload during development).
     *
     * Clears all cached libraries, forcing reload on next access.
     */
    void clear_cache();

    /**
     * @brief Clear cache for a specific path only.
     *
     * @param path Path to .metallib file to clear from cache.
     */
    void clear_cache_for_path(const std::string& path);

    /**
     * @brief Set the default metallib path.
     *
     * @param path Default path to use when path argument is empty.
     */
    void set_default_path(const std::string& path);

    /**
     * @brief Get the default metallib path.
     *
     * @return Default path string.
     */
    const std::string& get_default_path() const { return _default_path; }

    // -------------------------------------------------------------------------
    // Checksum Management
    // -------------------------------------------------------------------------

    /**
     * @brief Compute checksums for all .metal source files.
     *
     * @param metallib_path Path to metallib (used to locate source dirs).
     * @return Map of relative file paths to SHA-256 checksums.
     */
    std::unordered_map<std::string, std::string>
    compute_source_checksums(const std::string& metallib_path = "");

    /**
     * @brief Save checksum manifest alongside the metallib.
     *
     * @param metallib_path Path to metallib. If empty, uses default.
     * @param checksums Precomputed checksums, or empty to compute fresh.
     */
    void save_checksum_manifest(
        const std::string& metallib_path = "",
        const std::unordered_map<std::string, std::string>& checksums = {});

    /**
     * @brief Load checksum manifest for a metallib.
     *
     * @param metallib_path Path to metallib. If empty, uses default.
     * @return ChecksumManifest or std::nullopt if manifest missing/invalid.
     */
    std::optional<ChecksumManifest>
    load_checksum_manifest(const std::string& metallib_path = "");

    // -------------------------------------------------------------------------
    // Utility Functions
    // -------------------------------------------------------------------------

    /**
     * @brief Compute SHA-256 checksum of a file.
     *
     * @param file_path Path to the file.
     * @return Hex-encoded SHA-256 hash.
     */
    std::string compute_file_checksum(const std::string& file_path);

    /**
     * @brief Check if Metal is available on this system.
     *
     * @return True if Metal framework is available.
     */
    static bool has_metal();

    /**
     * @brief Require Metal, throwing if not available.
     *
     * @throws std::runtime_error if Metal is not available.
     */
    static void require_metal();

private:
    LibraryManager() = default;

    // -------------------------------------------------------------------------
    // Internal Helpers
    // -------------------------------------------------------------------------

    /**
     * @brief Resolve path to actual file path (empty -> default).
     */
    std::string _resolve_path(const std::string& path) const;

    /**
     * @brief Get path to checksum manifest file for a metallib.
     */
    std::string _get_checksum_manifest_path(const std::string& metallib_path) const;

    /**
     * @brief Get Metal device (cached).
     */
    MTL::Device* _get_device();

    /**
     * @brief Extract version info embedded in metallib file.
     *
     * Looks for metadata strings containing git hash, shader count, etc.
     */
    void _extract_embedded_version_info(
        const std::string& path,
        MetallibVersion& version);

    /**
     * @brief Get all directories that may contain .metal source files.
     */
    std::vector<std::string> _get_metal_source_dirs(
        const std::string& metallib_path) const;

    /**
     * @brief Collect all .metal files from source directories.
     */
    std::vector<std::string> _collect_metal_files(
        const std::vector<std::string>& source_dirs) const;

    /**
     * @brief Check staleness using mtime (legacy fallback).
     */
    bool _is_metallib_stale_mtime(const std::string& path) const;

    /**
     * @brief Get project root for relative path computation.
     */
    std::string _get_project_root(const std::string& metallib_path) const;

    // -------------------------------------------------------------------------
    // Member Variables
    // -------------------------------------------------------------------------

    std::string _default_path;  // Default metallib location
    MTL::Device* _device_cache = nullptr;  // Cached Metal device

    // Library cache: path -> MTLLibrary
    // Libraries are retained by Metal-cpp, we just store pointers
    std::unordered_map<std::string, MTL::Library*> _library_cache;

    // Mutex for thread safety
    mutable std::mutex _mutex;
};

} // namespace metal_marlin
