#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <file1> <file2> <file3>\n", argv[0]);
    return 1;
  }

  int offset = atoi(argv[4]);
  if (offset < 0) {
    perror("offset must be a non-negative integer.\n");
    return 1;
  }

  // Open required files
  int fd1 = open(argv[1], O_RDONLY);
  if (fd1 == -1) {
    perror("unable to open file 1");
    return 1;
  }
  int fd2 = open(argv[2], O_RDONLY);
  if (fd2 == -1) {
    perror("unable to open file 2");
    return 1;
  }
  int fd3 = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd3 == -1) {
    perror("unable to open file 3");
    return 1;
  }

  // Read from file2 and write to file3
  char buffer[BUFFER_SIZE];
  ssize_t bytesRead, bytesWritten;

  while ((bytesRead = read(fd2, buffer, BUFFER_SIZE)) > 0) {
    bytesWritten = write(fd3, buffer, bytesRead);
    if (bytesWritten != bytesRead) {
      perror("error writing content to file 3");
      close(fd1);
      close(fd2);
      close(fd3);
      return 1;
    }
  }
  if (bytesRead == -1) {
    perror("error reading from file 2");
    close(fd1);
    close(fd2);
    close(fd3);
    return 1;
  }

  // copy file1 content starting at the specified offset in file3
  if (lseek(fd3, offset, SEEK_SET) == -1) {
    perror("unable to seek in file 3");
  }

  // Read from file1 and write to file3 at the specified offset
  while ((bytesRead = read(fd1, buffer, BUFFER_SIZE)) > 0) {
    bytesWritten = write(fd3, buffer, bytesRead);
    if (bytesWritten != bytesRead) {
      perror("error writing content to file 3");
      close(fd1);
      close(fd2);
      close(fd3);
      return 1;
    }
  }
  if (bytesRead == -1) {
    perror("error reading from file 1");
    close(fd1);
    close(fd2);
    close(fd3);
    return 1;
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

  return 0;
}
