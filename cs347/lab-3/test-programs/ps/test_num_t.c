#include "types.h"
#include "stat.h"
#include "user.h"

void test_get_num_timer_interrupts(){
    int pid = getpid();

    printf(1,"\n_______TESTING get_num_timer_interrupts(pid)_______\n");
    printf(1, "get_num_timer_interrupts(%d): %d\n", pid, get_num_timer_interrupts(pid));
    for(double i = 0; i < 10e6; i++);
    printf(1, "get_num_timer_interrupts(%d) [after long for loop]: %d\n", pid,  get_num_timer_interrupts(pid));
}

int main()
{
    test_get_num_timer_interrupts();
    exit();
}