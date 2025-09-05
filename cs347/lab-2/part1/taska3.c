#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  int pipe_A_to_B[2];
  int pipe_B_to_A[2];
  if (pipe(pipe_A_to_B) == -1) {
    perror("pipe creation failed for A to B\n");
    return 1;
  }
  if (pipe(pipe_B_to_A) == -1) {
    perror("pipe creation failed for B to A\n");
    return 1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork failed\n");
    return 1;
  } else if (pid == 0) {
    // Child Process (Process B)

    // Close ununsed ends
    close(pipe_A_to_B[1]);
    close(pipe_B_to_A[0]);

    // Read num_a from parent
    int num_a;
    if (read(pipe_A_to_B[0], &num_a, sizeof(num_a)) == -1) {
      perror("error while reading num_a from parent\n");
      close(pipe_A_to_B[0]);
      close(pipe_B_to_A[1]);
      return 1;
    }
    close(pipe_A_to_B[0]);

    // Read num_b from user
    int num_b;
    printf("Process B (PID: %d): Enter a positive integer: ", getpid());
    if (scanf("%d", &num_b) != 1 || num_b < 0) {
      perror("invalid num_a\n");
      close(pipe_B_to_A[1]);
      return 1;
    }

    int result = num_a * num_b;
    if (write(pipe_B_to_A[1], &result, sizeof(result)) == -1) {
      perror("error while sending result to parent\n");
      close(pipe_B_to_A[1]);
      return 1;
    }
    close(pipe_B_to_A[1]);
  } else {
    // Parent Process (Process A)
    // Close ununsed ends
    close(pipe_A_to_B[0]);
    close(pipe_B_to_A[1]);

    // Read num_a from user
    int num_a;
    printf("Process A (PID: %d): Enter a positive integer: ", getpid());
    if (scanf("%d", &num_a) != 1 || num_a < 0) {
      perror("invalid num_a\n");
      return 1;
    }

    // Send num_a to child
    if (write(pipe_A_to_B[1], &num_a, sizeof(num_a)) == -1) {
      perror("error sending num_a to child\n");
      close(pipe_A_to_B[1]);
      close(pipe_B_to_A[0]);
      return 1;
    }
    close(pipe_A_to_B[1]);

    // Wait for child to finish
    wait(NULL);

    // Read result from child
    int result;
    if (read(pipe_B_to_A[0], &result, sizeof(result)) == -1) {
      perror("error reading result from child\n");
      close(pipe_B_to_A[0]);
      return 1;
    }
    close(pipe_B_to_A[0]);

    printf("Process A (PID: %d): Product is %d\n", getpid(), result);
  }

  return 0;
}
