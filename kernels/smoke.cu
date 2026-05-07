extern "C" __global__ void avo_smoke_kernel(float *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 1.0f;
  }
}
