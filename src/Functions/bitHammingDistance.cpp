#include <bit>
#include <Functions/FunctionBinaryArithmetic.h>
#include <Functions/FunctionFactory.h>
#include <Common/TargetSpecific.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_COLUMN;
}

template <typename A, typename B>
struct BitHammingDistanceImpl
{
    using ResultType = std::conditional_t<(sizeof(A) * 8 >= 256), UInt16, UInt8>;
    static constexpr bool allow_fixed_string = true;
    static constexpr bool allow_string_integer = false;

    template <typename Result = ResultType>
    static NO_SANITIZE_UNDEFINED Result apply(A a, B b)
    {
        /// Note: it's unspecified if signed integers should be promoted with sign-extension or with zero-fill.
        /// This behavior can change in the future.

        if constexpr (sizeof(A) <= sizeof(UInt64) && sizeof(B) <= sizeof(UInt64))
        {
            UInt64 res = static_cast<UInt64>(a) ^ static_cast<UInt64>(b);
            return static_cast<ResultType>(std::popcount(res));
        }
        else if constexpr (is_big_int_v<A> && is_big_int_v<B>)
        {
            auto xored = a ^ b;

            ResultType res = 0;
            for (auto item : xored.items)
                res += std::popcount(item);
            return res;
        }
        else
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Unsupported data type combination in function 'bitHammingDistance'");
    }

#if USE_EMBEDDED_COMPILER
    static constexpr bool compilable = false; /// special type handling, some other time
#endif
};

namespace
{
template <typename A, typename B, typename ResultType>
struct BitHammingDistanceTargetSpecificImpl
{
    using Op = BitHammingDistanceImpl<A, B>;

    /// Keep these kernels as straightforward scalar loops and rely on compiler auto-vectorization.
    /// For `d = popcount(a ^ b)`, modern x86 backends lower `std::popcount` over vectors to fast SIMD-friendly
    /// sequences (e.g. nibble-LUT + shuffle + horizontal sum), which is typically as good as or better
    /// than handwritten intrinsics while staying portable across target levels selected by multiversioning.
    MULTITARGET_FUNCTION_X86_V4_V3(
    MULTITARGET_FUNCTION_HEADER(static void NO_INLINE), vectorVectorKernel, MULTITARGET_FUNCTION_BODY(( /// NOLINT
        const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
            c[i] = Op::template apply<ResultType>(a[i], b[i]);
    }))

    MULTITARGET_FUNCTION_X86_V4_V3(
    MULTITARGET_FUNCTION_HEADER(static void NO_INLINE), leftConstantKernel, MULTITARGET_FUNCTION_BODY(( /// NOLINT
        const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
            c[i] = Op::template apply<ResultType>(*a, b[i]);
    }))

    MULTITARGET_FUNCTION_X86_V4_V3(
    MULTITARGET_FUNCTION_HEADER(static void NO_INLINE), rightConstantKernel, MULTITARGET_FUNCTION_BODY(( /// NOLINT
        const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
            c[i] = Op::template apply<ResultType>(a[i], *b);
    }))
};
}

namespace impl_
{
template <typename A, typename B, typename ResultType>
struct BinaryOperationImpl<A, B, BitHammingDistanceImpl<A, B>, ResultType>
{
    using Base = BinaryOperation<A, B, BitHammingDistanceImpl<A, B>, ResultType>;
    using TargetSpecificImpl = BitHammingDistanceTargetSpecificImpl<A, B, ResultType>;

    template <OpCase op_case>
    static void NO_INLINE
    process(const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size, const NullMap * = nullptr)
    {
        if constexpr (is_big_int_v<A> || is_big_int_v<B>)
            Base::template process<op_case>(a, b, c, size);
        else if constexpr (op_case == OpCase::Vector)
            processVectorVector(a, b, c, size);
        else if constexpr (op_case == OpCase::LeftConstant)
            processLeftConstant(a, b, c, size);
        else
            processRightConstant(a, b, c, size);
    }

    static ResultType process(A a, B b)
    {
        return BitHammingDistanceImpl<A, B>::template apply<ResultType>(a, b);
    }

private:
    template <typename FV3, typename FV4>
    static ALWAYS_INLINE bool dispatchX86V4V3([[maybe_unused]] const FV3 & fn_v3, [[maybe_unused]] const FV4 & fn_v4)
    {
#if USE_MULTITARGET_CODE
        if (isArchSupported(TargetArch::x86_64_v4))
        {
            fn_v4();
            return true;
        }

        if (isArchSupported(TargetArch::x86_64_v3))
        {
            fn_v3();
            return true;
        }
#endif
        return false;
    }

    static void NO_INLINE processVectorVector(const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size)
    {
#if USE_MULTITARGET_CODE
        if (!dispatchX86V4V3(
                [&] { TargetSpecificImpl::vectorVectorKernel_x86_64_v3(a, b, c, size); },
                [&] { TargetSpecificImpl::vectorVectorKernel_x86_64_v4(a, b, c, size); }))
#endif
            TargetSpecificImpl::vectorVectorKernel(a, b, c, size);
    }

    static void NO_INLINE processLeftConstant(const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size)
    {
#if USE_MULTITARGET_CODE
        if (!dispatchX86V4V3(
                [&] { TargetSpecificImpl::leftConstantKernel_x86_64_v3(a, b, c, size); },
                [&] { TargetSpecificImpl::leftConstantKernel_x86_64_v4(a, b, c, size); }))
#endif
            TargetSpecificImpl::leftConstantKernel(a, b, c, size);
    }

    static void NO_INLINE processRightConstant(const A * __restrict a, const B * __restrict b, ResultType * __restrict c, size_t size)
    {
#if USE_MULTITARGET_CODE
        if (!dispatchX86V4V3(
                [&] { TargetSpecificImpl::rightConstantKernel_x86_64_v3(a, b, c, size); },
                [&] { TargetSpecificImpl::rightConstantKernel_x86_64_v4(a, b, c, size); }))
#endif
            TargetSpecificImpl::rightConstantKernel(a, b, c, size);
    }
};
}

struct NameBitHammingDistance
{
    static constexpr auto name = "bitHammingDistance";
};
using FunctionBitHammingDistance = BinaryArithmeticOverloadResolver<BitHammingDistanceImpl, NameBitHammingDistance>;

REGISTER_FUNCTION(BitHammingDistance)
{
    FunctionDocumentation::Description description = R"(
Returns the [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance) between the bit representations of two numbers.
Can be used with [`SimHash`](../../sql-reference/functions/hash-functions.md#ngramSimHash) functions for detection of semi-duplicate strings.
The smaller the distance, the more similar the strings are.
)";
    FunctionDocumentation::Syntax syntax = "bitHammingDistance(x, y)";
    FunctionDocumentation::Arguments arguments = {
        {"x", "First number for Hamming distance calculation.", {"(U)Int*", "Float*"}},
        {"y", "Second number for Hamming distance calculation.", {"(U)Int*", "Float*"}},
    };
    FunctionDocumentation::ReturnedValue returned_value = {"Returns the hamming distance between `x` and `y`", {"UInt8"}};
    FunctionDocumentation::Examples examples = {{"Usage example", "SELECT bitHammingDistance(111, 121);",
        R"(
┌─bitHammingDistance(111, 121)─┐
│                            3 │
└──────────────────────────────┘
        )"}
    };
    FunctionDocumentation::IntroducedIn introduced_in = {21, 1};
    FunctionDocumentation::Category category = FunctionDocumentation::Category::Bit;
    FunctionDocumentation documentation = {description, syntax, arguments, {}, returned_value, examples, introduced_in, category};

    factory.registerFunction<FunctionBitHammingDistance>(documentation);
}
}
