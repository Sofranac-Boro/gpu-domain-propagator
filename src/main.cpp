//
// Created by boro on 9/28/21.
//

#include <stdio.h>
#include <Python.h>

int main()
{
   char filename[] = "../fileReader/run_propagation.py";
   FILE* fp;
   int argc;
   wchar_t * argv[11];

   argc = 11;
   argv[0] = (wchar_t *)L"run_propagation.py";
   argv[1] = (wchar_t *)L"-f";
   argv[2] = (wchar_t *)L"/home/bzfsofra/miplib2017/b-ball.mps.gz";
   argv[3] = (wchar_t *)L"-d";
   argv[4] = (wchar_t *)L"double";
   argv[5] = (wchar_t *)L"-t";
   argv[6] = (wchar_t *)L"measure";
   argv[7] = (wchar_t *)L"-s";
   argv[8] = (wchar_t *)L"0";
   argv[9] = (wchar_t *)L"-c";
   argv[10] = (wchar_t *)L"2";

   Py_SetProgramName(argv[0]);
   Py_Initialize();
   PySys_SetArgv(argc, argv);
   fp = _Py_fopen(filename, "r");

   PyRun_SimpleFile(fp, filename);
   Py_Finalize();
   return 0;
}

