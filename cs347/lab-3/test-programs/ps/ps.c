#include "types.h"
#include "stat.h"
#include "user.h"

int main(int argc, char *argv[])
{

    int e_flag = 0;
    if (argc > 1 && strcmp(argv[1], "-e") == 0) {
        e_flag = 1;
    }   
    if (!e_flag)
    {
        printf(1, "PID\tNAME\tSTATE\tSYS\tINT\n");
    }
    else 
    {
        printf(1, "PID\tSTATE\tSYS\tINT\n"); 
    }
    exit();
}