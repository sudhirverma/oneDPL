//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// add_cv

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T, class U>
void
test_add_cv_imp()
{
    ASSERT_SAME_TYPE(const volatile U, typename s::add_cv<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(const volatile U, s::add_cv_t<T>);
#endif
}

template <class T>
void
test_add_cv()
{
    test_add_cv_imp<T, const volatile T>();
    test_add_cv_imp<const T, const volatile T>();
    test_add_cv_imp<volatile T, volatile const T>();
    test_add_cv_imp<const volatile T, const volatile T>();
}

cl::sycl::cl_bool
kernel_test()
{
    test_add_cv<void>();
    test_add_cv<int>();
    test_add_cv<int[3]>();
    test_add_cv<int&>();
    test_add_cv<const int&>();
    test_add_cv<int*>();
    test_add_cv<const int*>();

    return true;
}

class KernelTest;

int
main()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }

    return 0;
}
