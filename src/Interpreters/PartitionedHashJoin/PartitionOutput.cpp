#include <Interpreters/PartitionedHashJoin/PartitionOutput.h>

namespace DB
{

void growPartitionOutput(PartitionOutput & po, BumpArena & arena, size_t cap, const size_t * elem_bytes, size_t num_cols)
{
    constexpr size_t kAlignMask = ~static_cast<size_t>(63);
    constexpr size_t hdr_size = (sizeof(OutBlock) + 63) & kAlignMask;

    size_t col_bytes_total = 0;
    for (size_t k = 0; k < num_cols; ++k)
        col_bytes_total += ((elem_bytes[k] * cap) + 63) & kAlignMask;

    uint8_t * raw = arena.alloc(hdr_size + col_bytes_total);
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
