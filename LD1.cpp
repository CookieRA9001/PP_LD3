#include <iostream>
#include <chrono>
#include <cmath>
#include <functional>

using namespace std::chrono;

#pragma acc routine
float inline random(float min, float max) {
    return min + static_cast<float>(rand()) / RAND_MAX * (max - min);
}

float montecarlo(
    const std::function<float(float,float, float, float)>& integrand,
    float xmin, float xmax,
    float ymin, float ymax,
    float a, float b, float c,
    const size_t points
) {
    signed long long int count = 0;
    float y, f;
    #pragma acc data copy(count)
    #pragma acc parallel loop private(y, f) independent 
    for (int i = 0; i < points; i++) {
        y = random(ymin, ymax);
        f = integrand(random(xmin, xmax),a,b,c);
        if (f > 0 && y < f && y > 0) count++;
        if (f < 0 && y > f && y < 0) count--;
    }
    return (float)count / (float)points * (xmax - xmin) * (ymax - ymin);
}

float func(float x, float a, float b, float c) { return (x-a)*(x-b)*(x-c); }

int main() {
    int a = 0, b = 2, c = 6;
    int x_min = std::min(std::min(a, b),c) - 1;
    int x_max = std::max(std::max(a, b),c) + 1;
    int y_min = -21, y_max = 35;

    auto start = system_clock::now();
    auto integral = montecarlo(func, x_min, x_max, y_min, y_max, a, b, c, 1000);
    std::cout << "Definite integral = " << integral << "\n";
    auto end = system_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "Elapsed time = " << elapsed << "ms\n";

    start = system_clock::now();
    integral = montecarlo(func, x_min, x_max, y_min, y_max, a, b, c, 10000);
    std::cout << "Definite integral = " << integral << "\n";
    end = system_clock::now();
    elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "Elapsed time = " << elapsed << "ms\n";

    start = system_clock::now();
    integral = montecarlo(func, x_min, x_max, y_min, y_max, a, b, c, 1000000);
    std::cout << "Definite integral = " << integral << "\n";
    end = system_clock::now();
    elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "Elapsed time = " << elapsed << "ms\n";

    start = system_clock::now();
    integral = montecarlo(func, x_min, x_max, y_min, y_max, a, b, c, 10000000);
    std::cout << "Definite integral = " << integral << "\n";
    end = system_clock::now();
    elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "Elapsed time = " << elapsed << "ms\n";

    start = system_clock::now();
    integral = montecarlo(func, x_min, x_max, y_min, y_max, a, b, c, 1000000000);
    std::cout << "Definite integral = " << integral << "\n";
    end = system_clock::now();
    elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "Elapsed time = " << elapsed << "ms\n";

    std::cin.get();
}

// #include <stdio.h>
// #include <stdlib.h>
// #include "openacc.h"

// #define N 1000000

// int main() {
//     int *a = (int *)malloc(N * sizeof(int));
//     int *b = (int *)malloc(N * sizeof(int));
//     int *c = (int *)malloc(N * sizeof(int));

//     // Initialize the arrays
//     for (int i = 0; i < N; i++) {
//         a[i] = i;
//         b[i] = 2 * i;
//     }

//     // Offload the addition to the GPU using OpenACC
//     #pragma acc parallel loop copyin(a[0:N], b[0:N]) copyout(c[0:N])
//     for (int i = 0; i < N; i++) {
//         c[i] = a[i] + b[i];
//     }

//     // Print some results to verify correctness
//     if (c[0] == 0 && c[N-1] == 3 * (N-1)) {
//         printf("OpenACC test successful!\n");
//     } else {
//         printf("Test failed.\n");
//     }

//     // Free memory
//     free(a);
//     free(b);
//     free(c);

//     return 0;
// }