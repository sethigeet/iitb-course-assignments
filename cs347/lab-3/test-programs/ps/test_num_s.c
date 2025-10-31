#include "types.h"
#include "stat.h"
#include "user.h"

void test_get_num_syscall(){
    int pid = getpid();
    int num_syscalls1 = get_num_syscall(pid);
    sleep(1);
    int num_syscalls2 = get_num_syscall(pid);
    
    printf(1,"\n_______TESTING get_num_syscall(pid)_______\n");
    printf(1, "get_num_syscall(%d): %d\n", pid, num_syscalls1);
    printf(1, "get_num_syscall(%d) [after sleep syscall]: %d\n", pid, num_syscalls2);
}

int main()
{
    test_get_num_syscall();
    exit();
}