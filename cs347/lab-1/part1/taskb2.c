#include <stdio.h>

int main(int argc, char *argv[]) {
  int m, k, n;
  scanf("%d %d %d", &m, &k, &n);

  int bloom[m], div[k];
  for (int i = 0; i < k; i++) {
    scanf("%d", &div[i]);
  }

  int op, arg, valid;
  for (int i = 0; i < n; i++) {
    scanf("%d %d", &op, &arg);

    switch (op) {
    case 1:
      for (int j = 0; j < k; j++) {
        bloom[(arg % div[k]) % m] = 1;
      }
      break;

    case 2:
      valid = 1;
      for (int j = 0; j < k; j++) {
        if (bloom[(arg % div[k]) % m] != 1) {
          valid = 0;
          break;
        }
      }

      printf("%d\n", valid);

      break;
    }
  }

  return 0;
}
