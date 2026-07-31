#pragma once

#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/TwoLevelHashMap.h>

namespace DB
{

// The std::is_constructible trait isn't suitable here because some classes have template constructors with semantics different from providing size hints.
// Also string hash table variants are not supported due to the fact that both local perf tests and tests in CI showed slowdowns for them.
template <typename...>
struct HasConstructorOfNumberOfElements : std::false_type
{
};

template <typename... Ts>
struct HasConstructorOfNumberOfElements<HashMapTable<Ts...>> : std::true_type
{
};

template <typename Key, typename Cell, typename Hash, typename Grower, typename Allocator, template <typename...> typename ImplTable>
struct HasConstructorOfNumberOfElements<TwoLevelHashMapTable<Key, Cell, Hash, Grower, Allocator, ImplTable>> : std::true_type
{
};

template <typename... Ts>
struct HasConstructorOfNumberOfElements<HashTable<Ts...>> : std::true_type
{
};

/// Fixed-bucket tables expose a size-hint constructor. Runtime-sized tables (`bits_for_bucket == 0`)
/// take a bucket count instead, so they must not match this trait.
template <typename Key, typename Cell, typename Hash, typename Grower, typename Allocator, typename ImplTable, size_t bits_for_bucket, typename BucketHash>
struct HasConstructorOfNumberOfElements<TwoLevelHashTable<Key, Cell, Hash, Grower, Allocator, ImplTable, bits_for_bucket, BucketHash>>
    : std::bool_constant<(bits_for_bucket > 0)>
{
};

template <template <typename> typename Method, typename Base>
struct HasConstructorOfNumberOfElements<Method<Base>> : HasConstructorOfNumberOfElements<Base>
{
};

}
