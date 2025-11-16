#include <time.h>
#include <stdio.h>
#include <stdlib.h>


static unsigned long long seed = 1;

double f1d(double x) {
    return x * x * x * x - 2 * x * x * x + 3 * x * x - 4 * x + 5;
}

double f2d(double x, double y) {
    return x * x - 2 * x * y + 2 * y * y - 2 * x - 4 * y + 27;
}

#ifdef __APPLE__
static double multiple(double a, double b) {
    double result;
    __asm__ volatile (
        "fmul %d0, %d1, %d2"
        : "=w" (result)
        : "w" (a), "w" (b)
        :
    );
    return result;
}
#endif

void dsrand(unsigned int i) {
    seed = (((long long int)i) << 16) | rand();
}

double drandom(double low, double high) {
    seed = (0x5DEECE66DLL * seed + 0xB16) & 0xFFFFFFFFFFFFLL;
    return low + (high - low) * ((double)(seed >> 16) / (double)0x100000000LL);
}

void progress_bar(int current, int epoch) {
    int width = 50;
    printf("\x1b[?25l");
    float percentage = (current + 1.0) / epoch;
    int filled = percentage * width;
    printf("\r[");
    for (int j = 0; j < width; j++) {
        if (j < filled) printf("\x1b[32m=\x1b[0m");
        else if (j == filled) printf(">");
        else printf(" ");
    }
    printf("] %7.2f%% (%d/%d)", percentage * 100, current + 1, epoch);
    fflush(stdout);

    if (current == epoch - 1) printf("\x1b[?25h\n");
}

double min(double *array, int length) {
    if (length == 1) return array[0];
    double result = array[0];
    for (int i = 1; i < length; i++) {
        if (result > array[i]) {
            result = array[i];
        }
    }
    return result;
}

int argmin(double *array, int length) {
    if (length == 1) return 0;
    int idx = 0;
    double result = array[0];
    for (int i = 1; i < length; i++) {
        if (result > array[i]) {
            result = array[i];
            idx = i;
        }
    }
    return idx;
}

double clip(double x, double low, double high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

double linear_decreasing_weight(int t, int t_max) {
    return 0.9 - 0.5 * t / t_max;
}

void particle_swarm_optimization_1d(
    int epoch,
    int n_particles,
    double w,
    double c1,
    double c2,
    double x_min,
    double x_max
) {
    double v_max = (x_max - x_min) / 5.0;

    double *x = (double *)malloc(sizeof(double) * n_particles);
    double *v = (double *)malloc(sizeof(double) * n_particles);
    double *p = (double *)malloc(sizeof(double) * n_particles);
    double *y = (double *)malloc(sizeof(double) * n_particles);

    for (int n = 0; n < n_particles; n++) {
        x[n] = drandom(x_min, x_max);
        v[n] = drandom(-v_max, v_max);
        p[n] = x[n];
        y[n] = f1d(x[n]);
    }

    double g = min(p, n_particles);
    double g_value = min(y, n_particles);

    for (int t = 0; t < epoch; t++) {
        // All particles are shared with the same random parameters.
        double r1 = drandom(0.0, 1.0);
        double r2 = drandom(0.0, 1.0);

        for (int n = 0; n < n_particles; n++) {
            v[n] = w * v[n] + c1 * r1 * (p[n] - x[n]) + c2 * r2 * (g - x[n]);
            v[n] = clip(v[n], -v_max, v_max);
            x[n] = x[n] + v[n];
            x[n] = clip(x[n], x_min, x_max);

            if (f1d(x[n]) < y[n]) {
                y[n] = f1d(x[n]);
                p[n] = x[n];
            }
        }

        int idx = argmin(y, n_particles);
        g_value = y[idx];
        g = p[idx];

        progress_bar(t, epoch);
    }

    printf("global optimal solution: %lf\n", g);
    printf("global optimal value:    %lf\n", g_value);

    free(x);
    free(v);
    free(p);
    free(y);
}

void particle_swarm_optimization_2d(
    int epoch,
    int n_particles,
    double c1,
    double c2,
    double x_min,
    double x_max,
    double y_min,
    double y_max
) {
    double v_x_max = (x_max - x_min) / 5.0;
    double v_y_max = (y_max - y_min) / 5.0;

    double (*x)[2] = (double (*)[2])malloc(sizeof(double) * n_particles * 2);
    double (*v)[2] = (double (*)[2])malloc(sizeof(double) * n_particles * 2);
    double (*p)[2] = (double (*)[2])malloc(sizeof(double) * n_particles * 2);
    double *y = (double *)malloc(sizeof(double) * n_particles);

    for (int n = 0; n < n_particles; n++) {
        x[n][0] = drandom(x_min, x_max);
        x[n][1] = drandom(y_min, y_max);
        v[n][0] = drandom(-v_x_max, v_x_max);
        v[n][1] = drandom(-v_y_max, v_y_max);
        p[n][0] = x[n][0];
        p[n][1] = x[n][1];
        y[n] = f2d(x[n][0], x[n][1]);
    }

    double g[2] = {min(p[0], n_particles), min(p[1], n_particles)};
    double g_value = min(y, n_particles);

    for (int t = 0; t < epoch; t++) {
        for (int n = 0; n < n_particles; n++) {
            // Each particle uses different random parameters.
            double r1 = drandom(0.0, 1.0);
            double r2 = drandom(0.0, 1.0);

            double w = linear_decreasing_weight(t, epoch);

            v[n][0] = multiple(w, v[n][0])
                    +
                    multiple(multiple(c1, r1), p[n][0] - x[n][0])
                    +
                    multiple(multiple(c2, r2), g[0] - x[n][0]);
            v[n][1] = multiple(w, v[n][1])
                    +
                    multiple(multiple(c1, r1), p[n][1] - x[n][1])
                    +
                    multiple(multiple(c2, r2), g[1] - x[n][1]);

            // v[n][0] = w * v[n][0] + c1 * r1 * (p[n][0] - x[n][0]) + c2 * r2 * (g[0] - x[n][0]);
            // v[n][1] = w * v[n][1] + c1 * r1 * (p[n][1] - x[n][1]) + c2 * r2 * (g[1] - x[n][1]);

            v[n][0] = clip(v[n][0], -v_x_max, v_x_max);
            v[n][1] = clip(v[n][1], -v_y_max, v_y_max);

            x[n][0] = x[n][0] + v[n][0];
            x[n][1] = x[n][1] + v[n][1];

            x[n][0] = clip(x[n][0], x_min, x_max);
            x[n][1] = clip(x[n][1], y_min, y_max);

            if (f2d(x[n][0], x[n][1]) < y[n]) {
                y[n] = f2d(x[n][0], x[n][1]);
                p[n][0] = x[n][0];
                p[n][1] = x[n][1];
            }
        }

        int idx = argmin(y, n_particles);
        g_value = y[idx];
        g[0] = p[idx][0];
        g[1] = p[idx][1];

        progress_bar(t, epoch);
    }

    printf("global optimal solution: (%lf, %lf)\n", g[0], g[1]);
    printf("global optimal value:    %lf\n", g_value);

    free(x);
    free(v);
    free(p);
    free(y);
}

int main(int argc, char *argv[], char *envs[]) {
    dsrand((unsigned int)time(NULL));

    int n_particles = 30;
    double w = 0.7;
    double c1 = 2.8;
    double c2 = 1.3;
    double x_min = -5.0;
    double x_max = 5.0;
    double y_min = -5.0;
    double y_max = 5.0;
    int epoch = 10000;

    particle_swarm_optimization_1d(epoch, n_particles, w, c1, c2, x_min, x_max);
    particle_swarm_optimization_2d(epoch, n_particles, c1, c2, x_min, x_max, y_min, y_max);

    return 0;
}