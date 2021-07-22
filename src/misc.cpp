#include "misc.h"

void measureTime(const char alg_name[30], std::chrono::_V2::steady_clock::time_point start,
                 std::chrono::_V2::steady_clock::time_point end) {
   std::cout << alg_name << " execution time : "
             << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
             << " nanoseconds" << std::endl;
}

const char * sync_type_to_str(GDP_SYNCTYPE type)
{
   if (type == CPU_LOOP)
   {
      return "CPU loop";
   } else if (type == GPU_LOOP)
   {
      return "GPU loop";
   } else if (type == MEGAKERNEL)
   {
      return "Megakernel";
   } else
   {
      throw std::runtime_error(std::string("Unknown sync type\n"));
   }
}
