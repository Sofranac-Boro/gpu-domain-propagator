#include "misc.h"

void measureTime(const char alg_name[30], std::chrono::_V2::steady_clock::time_point start,
                 std::chrono::_V2::steady_clock::time_point end) {
   std::cout << alg_name << " execution time : "
             << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
             << " nanoseconds" << std::endl;
}

int CUDAgetMaxNumResidentBlocksPerSM(int major, int minor)
{
   if (major >= 5 && major <=6)
      return 32;
   else if(major == 7 && (minor ==0 || minor == 1 || minor == 2) )
      return 32;
   else if (major ==7 && minor == 5)
      return 16;
   else if(major == 8 && minor == 0)
      return 32;
   else if(major == 8 && minor == 6)
      return 16;
   else
      throw "Unknown compute capability";
}
