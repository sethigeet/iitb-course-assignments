#include "types.h"
#include "stat.h"
#include "user.h"

void test_proc_name(){
    printf(1,"\n_______TESTING fill_proc_name(pid) and get_proc_name(pid)_______\n");

    int pid = getpid(); 
    char* buf = malloc(16);
    if (buf == 0) {
        printf(1, "Memory allocation failed\n");
        exit();
    }
    buf = "hello world!";
    int fill_status1 = fill_proc_name(pid, buf);
    printf(1, "fill_proc_name(%d): %s (Status: %d)\n", pid, buf, fill_status1);

    int fill_status2 = fill_proc_name(10000, buf);
    printf(1, "fill_proc_name(%d): %s (Status: %d)\n", 10000, buf, fill_status2);

    char name[16];
    if (get_proc_name(pid, name, sizeof(name)) > 0)
    printf(1, "Process with pid (%d) has name: %s\n",pid, name);
    else
    printf(1, "Process not found\n");

    if (get_proc_name(10000, name, sizeof(name)) > 0)
    printf(1, "Process with pid (%d) has name: %s\n",10000, name);
    else
    printf(1, "Process with pid (%d) was not found\n", 10000);
}

int main()
{
    test_proc_name();
    exit();
}