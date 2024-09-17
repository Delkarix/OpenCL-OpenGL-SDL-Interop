#include <stdio.h>
#include "helper_cuda.h"
#include "Mandelbrot_kernel.h"

// The dimensions of the thread block
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

// Increase the grid size by 1 if the image width or height does not divide
// evenly
// by the thread block dimensions
inline int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}  // iDivUp
