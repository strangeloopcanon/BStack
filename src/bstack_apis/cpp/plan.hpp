#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bw::stack {

struct KvPageRef {
    std::string tensor;
    std::uint64_t page;
    std::uint32_t head;
    std::uint32_t layer;
};

enum class TransferKind : std::uint32_t {
    KIND_UNSPECIFIED = 0,
    H2D = 1,
    D2H = 2,
    P2P = 3,
    STORAGE2H = 4,
};

struct TransferOp {
    TransferKind kind;
    std::string src;
    std::string dst;
    std::uint64_t length;
    std::uint64_t src_offset{0};
    std::uint64_t dst_offset{0};
    std::vector<KvPageRef> kv_refs;
    std::string note;
};

struct FileChunk {
    std::string path;
    std::uint64_t offset;
    std::uint64_t length;
    std::string sha256;
};

struct WeightManifest {
    std::string model_id;
    std::string version;
    std::vector<FileChunk> files;
};

struct SwapWindow {
    std::uint64_t t_start_ns{0};
    std::uint64_t t_deadline_ns{0};
};

struct CachePlan {
    std::string plan_id;
    std::vector<TransferOp> ops;
    std::vector<KvPageRef> prefetch;
    std::vector<KvPageRef> evict;
};

struct SwapPlan {
    std::string plan_id;
    WeightManifest from;
    WeightManifest to;
    std::vector<TransferOp> ops;
    SwapWindow window;
};

}  // namespace bw::stack
