#include <algorithm>
#include <memory>
#include <CL/sycl.hpp>

int main()
{
    const int n = 100;
    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_2(sycl::malloc_shared<int>(n, q), usm_deleter);
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_3(sycl::malloc_shared<int>(n, q), usm_deleter);
    int* usm_ptr_1 = usm_uptr_1.get();
    int* usm_ptr_2 = usm_uptr_2.get();
    int* usm_ptr_3 = usm_uptr_3.get();

    auto event1 = q.fill(usm_ptr_1, 42, n);

    auto event_2 = q.parallel_for(sycl::range<>(n), {event1},
        [=](auto id) {
            usm_ptr_2[id] = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 41) - usm_ptr_1;
            usm_ptr_3[id] = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 42) - usm_ptr_1;
        });
    event_2.wait();

    std::cout << usm_ptr_2[0] << " " << usm_ptr_3[0] << std::endl;
    return 0;
}