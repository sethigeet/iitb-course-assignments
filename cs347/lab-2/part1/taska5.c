#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int create_child(int n) {
  // Recursion base case
  if (n == 0)
    return 0;

  // Fork to create child
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork failed!\n");
    return 1;
  } else if (pid == 0) {
    // Child process: recursively execute the program with n-1
    sleep(1);
    printf("Process %d created, parent pid is %d\n", getpid(), getppid());
    create_child(n - 1);
    sleep(2);
    printf("Process %d exiting, parent PID is %d\n", getpid(), getppid());
  } else {
    // Parent process: wait for child to terminate
    wait(NULL);
  }

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    // Invalid number of arguments
    fprintf(stderr, "Usage: %s <positive integer>\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]);
  if (n < 0) {
    // Invalid argument
    fprintf(stderr, "Usage: %s <positive integer>\n", argv[0]);
    return 1;
  }

  return create_child(n);
}
