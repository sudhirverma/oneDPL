// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H
#define _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H

//The file is an internal file and the code of that file is included by a major file into the following namespaces:
//namespace oneapi
//{
//namespace dpl
//{
//namespace __par_backend_hetero
//{

template <typename _KernelNameBase, uint16_t __wg_size = 256 /*work group size*/, uint16_t __block_size = 16,
          ::std::uint32_t __radix = 4, bool __is_asc = true,
          uint16_t __req_sub_group_size = (__block_size < 4 ? 32 : 16)>
struct __subgroup_radix_sort
{
    template <typename _RangeIn, typename _Proj>
    auto
    operator()(sycl::queue __q, _RangeIn&& __src, _Proj __proj)
    {
        using _KeyT = oneapi::dpl::__internal::__value_t<_RangeIn>;
        //check SLM size
        if (__ckeck_slm_size<_KeyT>(__q, __src.size()))
            return __submit<__i_kernel_name<_KernelNameBase, 0>, std::true_type /*SLM*/>(
                __q, ::std::forward<_RangeIn>(__src), __proj);
        else
            return __submit<__i_kernel_name<_KernelNameBase, 1>, std::false_type /*global memory*/>(
                __q, ::std::forward<_RangeIn>(__src), __proj);
    }

  private:
    template <typename _KeyT, typename>
    class _TempBuf;

    template <typename _KeyT>
    class _TempBuf<_KeyT, std::true_type /*shared local memory buffer*/>
    {
        uint16_t __buf_size;

      public:
        _TempBuf(uint16_t __n) : __buf_size(__n) {}
        auto
        get_acc(sycl::handler& __cgh)
        {
            return __dpl_sycl::__local_accessor<_KeyT>(__buf_size, __cgh);
        }
    };

    template <typename _KeyT>
    class _TempBuf<_KeyT, std::false_type /*global memory buffer*/>
    {
        sycl::buffer<_KeyT> __buf;

      public:
        _TempBuf(uint16_t __n) : __buf(__n) {}
        auto
        get_acc(sycl::handler& __cgh)
        {
            return sycl::accessor(__buf, __cgh, sycl::read_write, __dpl_sycl::__no_init{});
        }
    };

    template <typename _ValT, bool __uniform_space, typename _Wi, typename _Src, typename _Vals>
    static void
    __block_load(const _Wi __wi, const _Src& __src, _Vals& __vals, const uint32_t __n)
    {
        _ONEDPL_PRAGMA_UNROLL
        for (uint16_t __i = 0; __i < __block_size; ++__i)
        {
            const uint16_t __idx = __wi * __block_size + __i;
            if (__idx < __n)
                __vals[__i] = __src[__idx];
            else
            {
                if constexpr (__uniform_space)
                {
                    //we use numeric_limits::lowest for floating-point types with denormalization,
                    //due to numeric_limits::min gets the minimum positive normalized value
                    static constexpr _ValT __default_val =
                        __is_asc ? std::numeric_limits<_ValT>::max() : std::numeric_limits<_ValT>::lowest();

                    __vals[__i] = __default_val;
                }
            }
        }
    }

    template <bool __uniform_space, typename _Item, typename _Wi, typename _Lacc, typename _Vals, typename _Indices>
    static void
    __to_blocked(_Item __it, const _Wi __wi, _Lacc& __exchange_lacc, _Vals& __vals, const uint32_t __n, const _Indices& __indices)
    {
        _ONEDPL_PRAGMA_UNROLL
        for (uint16_t __i = 0; __i < __block_size; ++__i)
        {
            if constexpr (!__uniform_space)
            {
                const uint16_t __idx = __wi * __block_size + __i;
                if(__idx >= __n)
                    continue;
            }
            __exchange_lacc[__indices[__i]] = __vals[__i];
        }

        __dpl_sycl::__group_barrier(__it);

        _ONEDPL_PRAGMA_UNROLL
        for (uint16_t __i = 0; __i < __block_size; ++__i)
        {
            if constexpr (!__uniform_space)
            {
                const uint16_t __idx = __wi * __block_size + __i;
                if(__idx >= __n)
                    continue;
            }
            __vals[__i] = __exchange_lacc[__wi * __block_size + __i];
        }
    }

    static_assert(__wg_size <= 1024);
    static constexpr uint16_t __bin_count = 1 << __radix;
    static constexpr uint16_t __counter_buf_sz = __wg_size * __bin_count + 1; //+1(init value) for exclusive scan result

    template <typename _T, typename _Size>
    bool
    __ckeck_slm_size(sycl::queue __q, _Size __n)
    {
        assert(__n <= 1 << (sizeof(uint16_t) * 8)); //the kernel is designed for data size <= 64K

        const auto __max_slm_size = __q.get_device().template get_info<sycl::info::device::local_mem_size>();

        const auto __n_uniform = 1 << (::std::uint32_t(log2(__n - 1)) + 1);
        const auto __req_slm_size_val = sizeof(_T) * __n_uniform;
        const auto __req_slm_size_counters = __counter_buf_sz * sizeof(uint32_t);

        return __req_slm_size_val <= __max_slm_size - __req_slm_size_counters; //counters should be placed in SLM
    }

    template <typename _KernelName, typename _SLM_tag, typename _RangeIn, typename _Proj>
    auto
    __submit(sycl::queue __q, _RangeIn&& __src, _Proj __proj)
    {
        uint16_t __n = __src.size();
        assert(__n <= __block_size * __wg_size);

        using _ValT = oneapi::dpl::__internal::__value_t<_RangeIn>;
        using _KeyT = oneapi::dpl::__internal::__key_t<_Proj, _RangeIn>;

        //A value is a key: we use an uniform iteration space and use a defaultult(identical) value for __i >= __n
        constexpr bool __uniform_space = std::is_same_v<_KeyT, _ValT> && sizeof(_KeyT) == sizeof(_ValT);

        _TempBuf<_ValT, _SLM_tag> __buf_val(__block_size * __wg_size);
        _TempBuf<uint32_t, _SLM_tag> __buf_count(__counter_buf_sz);

        sycl::nd_range __range{sycl::range{__wg_size}, sycl::range{__wg_size}};
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __src);

            auto __exchange_lacc = __buf_val.get_acc(__cgh);
            auto __counter_lacc = __buf_count.get_acc(__cgh);

            __cgh.parallel_for<_KernelName>(
                __range, ([=](sycl::nd_item<1> __it)[[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__req_sub_group_size)]] {
                    _ValT __vals[__block_size];
                    uint16_t __wi = __it.get_local_linear_id();
                    uint16_t __begin_bit = 0;
                    constexpr uint16_t __end_bit = sizeof(_KeyT) * 8;

                    __block_load<_ValT, __uniform_space>(__wi, __src, __vals, __n);

                    __dpl_sycl::__group_barrier(__it);
                    while (true)
                    {
                        uint16_t __indices[__block_size]; //indices for indirect access in the "re-order" phase
                        {
                            uint32_t* __counters[__block_size]; //pointers(by performance reasons) to bucket's counters

                            //1. "counting" phase
                            //counter initialization
                            auto __pcounter = __counter_lacc.get_pointer() + __wi;

                            _ONEDPL_PRAGMA_UNROLL
                            for (uint16_t __i = 0; __i < __bin_count; ++__i)
                                __pcounter[__i * __wg_size] = 0;

                            _ONEDPL_PRAGMA_UNROLL
                            for (uint16_t __i = 0; __i < __block_size; ++__i)
                            {
                                if constexpr (!__uniform_space)
                                {
                                    const uint16_t __idx = __wi * __block_size + __i;
                                    if(__idx >= __n)
                                        continue;
                                }

                                const uint16_t __bin = __get_bucket</*mask*/ __bin_count - 1>(
                                    __order_preserving_cast<__is_asc>(__proj(__vals[__i])), __begin_bit);

                                //"counting" and local offset calculation
                                __counters[__i] = &__pcounter[__bin * __wg_size];
                                __indices[__i] = *__counters[__i];
                                *__counters[__i] = __indices[__i] + 1;
                            }
                            __dpl_sycl::__group_barrier(__it);

                            //2. scan phase
                            {
                                //TODO: probably can be further optimized

                                //scan contiguous numbers
                                uint16_t __bin_sum[__bin_count];
                                __bin_sum[0] = __counter_lacc[__wi * __bin_count];

                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 1; __i < __bin_count; ++__i)
                                    __bin_sum[__i] = __bin_sum[__i - 1] + __counter_lacc[__wi * __bin_count + __i];

                                __dpl_sycl::__group_barrier(__it);
                                //exclusive scan local sum
                                uint16_t __sum_scan = __dpl_sycl::__exclusive_scan_over_group(
                                    __it.get_group(), __bin_sum[__bin_count - 1], __dpl_sycl::__plus<uint16_t>());
                                //add to local sum, generate exclusive scan result
                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __bin_count; ++__i)
                                    __counter_lacc[__wi * __bin_count + __i + 1] = __sum_scan + __bin_sum[__i];

                                if (__wi == 0)
                                    __counter_lacc[0] = 0;
                                __dpl_sycl::__group_barrier(__it);
                            }

                            _ONEDPL_PRAGMA_UNROLL
                            for (uint16_t __i = 0; __i < __block_size; ++__i)
                                __indices[__i] += *__counters[__i];
                        }

                        __begin_bit += __radix;

                        //3. "re-order" phase
                        __dpl_sycl::__group_barrier(__it);
                        if (__begin_bit >= __end_bit)
                        {
                            // the last iteration - writing out the result
                            _ONEDPL_PRAGMA_UNROLL
                            for (uint16_t __i = 0; __i < __block_size; ++__i)
                            {
                                if constexpr (!__uniform_space)
                                {
                                    const uint16_t __idx = __wi * __block_size + __i;
                                    if(__idx >= __n)
                                        continue;
                                }
                                const uint16_t __r = __indices[__i];
                                if (__r < __n)
                                    __src[__r] = __vals[__i];
                            }
                            return;
                        }
                        __to_blocked<__uniform_space>(__it, __wi, __exchange_lacc, __vals, __n, __indices);
                        __dpl_sycl::__group_barrier(__it);
                    }
                }));
        });
    }
};

//} // namespace __par_backend_hetero
//} // namespace dpl
//} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
