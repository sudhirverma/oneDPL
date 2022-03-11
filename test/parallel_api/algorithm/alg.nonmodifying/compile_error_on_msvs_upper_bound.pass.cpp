
#define _ITERATOR_DEBUG_LEVEL 0

#define USE_SYCL_CODE

#include <iostream>
#include <algorithm>
#include <memory>
#include <exception>
#include <thread>
#include <random>
#include <signal.h>

#ifdef USE_SYCL_CODE
#include <CL/sycl.hpp>
#endif

#define DISPLAY_LINE
#define WAIT_ON_START_SEC 30

void signal_handler(int signal)
{
    std::cout << "Signal received : " << signal << " - ";
    switch (signal)
    {
    case SIGINT:    std::cout <<   "SIGINT - external interrupt, usually initiated by the user";               break;
    case SIGILL :   std::cout <<   "SIGILL - invalid program image, such as invalid instruction";              break;
    case SIGFPE :   std::cout <<   "SIGFPE - erroneous arithmetic operation such as divide by zero";           break;
    case SIGSEGV :  std::cout <<  "SIGSEGV - invalid memory access (segmentation fault)";                      break;
    case SIGTERM :  std::cout <<  "SIGTERM - termination request, sent to the program";                        break;
    case SIGBREAK : std::cout << "SIGBREAK - Ctrl-Break sequence";                                             break;
    case SIGABRT :  std::cout <<  "SIGABRT - abnormal termination condition, as is e.g. initiated by abort()"; break;
    }
    std::cout << std::endl;

    //std::terminate();
}

void display_line(const char* msg)
{
#ifdef DISPLAY_LINE
    std::cout << msg << std::endl;
#endif // DISPLAY_LINE
}

int main()
{
    signal(SIGINT,   signal_handler);
    signal(SIGILL,   signal_handler);
    signal(SIGFPE,   signal_handler);
    signal(SIGSEGV,  signal_handler);
    signal(SIGTERM,  signal_handler);
    signal(SIGBREAK, signal_handler);
    signal(SIGABRT,  signal_handler);

    const int n = 100;

#ifdef WAIT_ON_START_SEC
    display_line("std::this_thread::sleep_for(std::chrono::seconds(30));");
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_ON_START_SEC));
#endif

    std::vector<int> data_to_sort = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data_to_sort.begin(), data_to_sort.end(), g);

    display_line("std::sort(data_to_sort.begin(), data_to_sort.end());");
    std::sort(data_to_sort.begin(), data_to_sort.end());

    display_line("auto it1 = ::std::upper_bound(data_to_sort.begin(), data_to_sort.end(), 11);");
    auto it1 = ::std::upper_bound(data_to_sort.begin(), data_to_sort.end(), 11);
    it1 = it1;

    display_line("auto it2 = ::std::lower_bound(data_to_sort.begin(), data_to_sort.end(), 13);");
    auto it2 = ::std::lower_bound(data_to_sort.begin(), data_to_sort.end(), 13);
    it2 = it2;

    try
    {
#ifdef USE_SYCL_CODE
        sycl::queue q;
        const auto name = q.get_device().get_info<sycl::info::device::name>();
        std::cout << "Running on ..."/* << name.c_str()*/ << std::endl;

        display_line("auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };");
        auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };

        display_line("std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);");
        std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);

        display_line("std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_2(sycl::malloc_shared<int>(n, q), usm_deleter);");
        std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_2(sycl::malloc_shared<int>(n, q), usm_deleter);

        display_line("std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_3(sycl::malloc_shared<int>(n, q), usm_deleter);");
        std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_3(sycl::malloc_shared<int>(n, q), usm_deleter);

        display_line("int* usm_ptr_1 = usm_uptr_1.get();");
        int* usm_ptr_1 = usm_uptr_1.get();

        display_line("int* usm_ptr_2 = usm_uptr_2.get();");
        int* usm_ptr_2 = usm_uptr_2.get();

        display_line("int* usm_ptr_3 = usm_uptr_3.get();");
        int* usm_ptr_3 = usm_uptr_3.get();

        display_line("auto event1 = q.fill(usm_ptr_1, 42, n);");
        auto event1 = q.fill(usm_ptr_1, 42, n);

        display_line("auto event_2 = q.parallel_for(sycl::range<>(n), {event1},.....");
        auto event_2 = q.parallel_for(sycl::range<>(n), {event1},
            [=](auto id) {
                usm_ptr_2[id] = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 41) - usm_ptr_1;
                usm_ptr_3[id] = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 42) - usm_ptr_1;
            });

        display_line("event_2.wait();");
        event_2.wait();

        std::cout << usm_ptr_2[0] << " " << usm_ptr_3[0] << std::endl;
#endif
    }
    catch (const std::exception& exc)
    {
        std::cout << "Exception occurred : ";
        if (exc.what())
            std::cout << exc.what();
        std::cout << std::endl;
        return -1;
    }

    return 0;
}