#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <file1> <file2> <file3>\n", argv[0]);
    exit(1);
  }

  char buffer[BUFFER_SIZE];
  ssize_t bytes_read, bytes_written;

  int fd1 = open(argv[1], O_RDONLY);
  if (fd1 == -1) {
    perror("unable to open file 1");
    return 1;
  }

  int fd2 = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd2 == -1) {
    perror("unable to open file 2");
    return 1;
  }

  int fd3 = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd3 == -1) {
    perror("unable to open file 3");
    return 1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork failed");
    return 1;
  } else if (pid == 0) { // Child process
    // Copy content of file1 to file3
    while ((bytes_read = read(fd1, buffer, BUFFER_SIZE)) > 0) {
      bytes_written = write(fd3, buffer, bytes_read);
      if (bytes_written != bytes_read) {
        perror("error while writing to file 3");
        exit(1);
      }
    }
    if (bytes_read == -1) {
      perror("error while reading from file 1");
      exit(1);
    }

    // Close all file descriptors
    if (close(fd1) == -1) {
      perror("error closing file 1");
      close(fd2);
      close(fd3);
    }
    if (close(fd2) == -1) {
      perror("error closing file 2");
      close(fd3);
    }
    if (close(fd3) == -1) {
      perror("error closing file 3");
    }

    exit(0);
  } else { // Parent process
    if (lseek(fd1, 0, SEEK_SET) == -1) {
      perror("error while seeking file 1");
      exit(1);
    }

    // Copy content of file1 to file2
    while ((bytes_read = read(fd1, buffer, BUFFER_SIZE)) > 0) {
      bytes_written = write(fd2, buffer, bytes_read);
      if (bytes_written != bytes_read) {
        perror("error while writing to file 2");
        exit(1);
      }
    }
    if (bytes_read == -1) {
      perror("error while reading from file 1");
      exit(1);
    }

    wait(NULL);

    // Close all file descriptors
    if (close(fd1) == -1) {
      perror("error closing file 1");
      close(fd2);
      close(fd3);
    }
    if (close(fd2) == -1) {
      perror("error closing file 2");
      close(fd3);
    }
    if (close(fd3) == -1) {
      perror("error closing file 3");
    }
  }

  return 0;
}
