#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  char program[100]; // Buffer for program name

  while (1) {
    // Print prompt and read program name
    printf("Enter program: ");
    scanf("%s", program);

    pid_t pid = fork();
    if (pid < 0) {
      perror("fork failed\n");
      return 1;
    } else if (pid == 0) {
      // Child process
      execlp(program, program, NULL);

      // Print error message if exec failed
      printf("unable to execute program %s\n", program);
      return 1;
    } else {
      // Parent process
      pid_t status;
      wait(&status);
      printf("Child %d finished executing %s with exit code %d\n", pid, program,
             WEXITSTATUS(status));
    }
  }

  return 0;
}
