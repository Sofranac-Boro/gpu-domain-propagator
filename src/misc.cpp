#include "misc.h"

void measureTime(const char alg_name[30], std::chrono::_V2::steady_clock::time_point start,
                 std::chrono::_V2::steady_clock::time_point end) {
   std::cout << alg_name << " execution time : "
             << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
             << " microseconds" << std::endl;
}