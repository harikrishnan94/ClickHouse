#include <Interpreters/PartitionedHashJoin/PartitionOutput.h>

namespace DB
{

void growPartitionOutput(PartitionOutput & po, Arena & arena, size_t cap, const size_t * elem_bytes, size_t num_cols)
{
    constexpr size_t kAlignMask = ~static_cast<size_t>(63);
    constexpr size_t hdr_size = (sizeof(OutBlock) + 63) & kAlignMask;
    static_assert(alignof(OutBlock) == 64, "OutBlock must be 64-byte aligned");

    size_t col_bytes_total = 0;
    for (size_t k = 0; k < num_cols; ++k)
        col_bytes_total += ((elem_bytes[k] * cap) + 63) & kAlignMask;

    /// 64-byte alignment is required: placement-new of alignas(64) OutBlock and
    /// SWWC's non-temporal stores into column buffers both depend on it.
    auto * raw = reinterpret_cast<uint8_t *>(arena.alignedAlloc(hdr_size + col_bytes_total, 64));
    auto * b = new (raw) OutBlock();
    b->capacity = cap;
    b->num_cols = static_cast<uint8_t>(num_cols);

    uint8_t * base = raw + hdr_size;
    for (size_t k = 0; k < num_cols; ++k)
    {
        b->cols[k] = base;
        base += ((elem_bytes[k] * cap) + 63) & kAlignMask;
    }

    b->next = po.head;
    po.head = b;
    po.cur = b;
    po.next_cap = nextOutBlockCap(cap);
}

}
