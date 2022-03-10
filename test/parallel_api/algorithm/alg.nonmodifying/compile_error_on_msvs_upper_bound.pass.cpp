
#define _ITERATOR_DEBUG_LEVEL 0

#include <iostream>
#include <algorithm>
#include <memory>
#include <exception>

#include <CL/sycl.hpp>

int main()
{
    const int n = 100;

    try
    {
        sycl::queue q;
        std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        std::cout << "auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };" << std::endl;
        auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };

        std::cout << "std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);" << std::endl;
        std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);

        std::cout << "std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_2(sycl::malloc_shared<int>(n, q), usm_deleter);" << std::endl;
        std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_2(sycl::malloc_shared<int>(n, q), usm_deleter);

        std::cout << "std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_3(sycl::malloc_shared<int>(n, q), usm_deleter);" << std::endl;
        std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_3(sycl::malloc_shared<int>(n, q), usm_deleter);

        std::cout << "int* usm_ptr_1 = usm_uptr_1.get();" << std::endl;
        int* usm_ptr_1 = usm_uptr_1.get();

        std::cout << "int* usm_ptr_2 = usm_uptr_2.get();" << std::endl;
        int* usm_ptr_2 = usm_uptr_2.get();

        std::cout << "int* usm_ptr_3 = usm_uptr_3.get();" << std::endl;
        int* usm_ptr_3 = usm_uptr_3.get();

        std::cout << "auto event1 = q.fill(usm_ptr_1, 42, n);" << std::endl;
        auto event1 = q.fill(usm_ptr_1, 42, n);

        std::cout << "auto event_2 = q.parallel_for(sycl::range<>(n), {event1},....." << std::endl;
        auto event_2 = q.parallel_for(sycl::range<>(n), {event1},
            [=](auto id) {
                usm_ptr_2[id] = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 41) - usm_ptr_1;
                usm_ptr_3[id] = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 42) - usm_ptr_1;
            });

        std::cout << "event_2.wait();" << std::endl;
        event_2.wait();

        std::cout << usm_ptr_2[0] << " " << usm_ptr_3[0] << std::endl;
    }
    catch (const std::exception& exc)
    {
        std::cout << "Exception occurred : " << exc.what() << std::endl;
        return -1;
    }

    return 0;
}