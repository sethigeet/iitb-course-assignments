#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  double aks[5];
  for (int i = 0; i < 5; i++) {
    scanf("%lf", &aks[i]);
  }

  for (int i = 0; i < 5; i++) {
    float bk = sin(1.0 / (1.0 + exp(-1.0 * aks[i])));
    printf("%.3f\n", bk);
  }

  return 0;
}
