__device__ __inline__ void gradient_update(double *c, double *b, double x, double L, int n) {
    double factor = 1.0 - L * (n - 1.0) / x;
    double old_c = *c;
    *c = 0.9 * factor * old_c;
    *b = 0.1 * factor * old_c;
}