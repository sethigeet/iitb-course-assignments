#include "types.h"
#include "stat.h"
#include "user.h"

void test_proc_state()
{
    printf(1,"\n_______TESTING get proc state(pid)_______\n");
    int pid = getpid();
    char state[16];
    if (get_proc_state(1, state, sizeof(state)) > 0)
        printf(1, "Process with pid (%d) has state: %s\n", 1, state);
    else
        printf(1, "Process not found\n");
    if (get_proc_state(2, state, sizeof(state)) > 0)
        printf(1, "Process with pid (%d) has state: %s\n", 2, state);
    else
        printf(1, "Process not found\n");
    if (get_proc_state(pid, state, sizeof(state)) > 0)
        printf(1, "Process with pid (%d) has state: %s\n", pid, state);
    else
        printf(1, "Process not found\n");
    if (get_proc_state(10000, state, sizeof(state)) > 0)
        printf(1, "Process with pid (%d) has state: %s\n", 10000, state);
    else
        printf(1, "Process not found\n");
}
int main()
{
    test_proc_state();
    exit();
}