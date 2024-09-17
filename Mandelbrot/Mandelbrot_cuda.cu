#include <stdio.h>
#include "helper_cuda.h"
#include "Mandelbrot_kernel.h"
#include "Mandelbrot_kernel.cuh"

__device__ inline int CalcMandelbrot(const float xPos, const float yPos, const int crunch) {
  float x, y, xx, yy, xC, yC;
  int i = crunch;


    xC = xPos;
    yC = yPos;
    x = y = 0;
    xx = yy = 0;

  while (--i && (xx + yy < 4.0f)) {
    y = x * y + x * y + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;
  }

  return i;
}  // CalcMandelbrot

// The Mandelbrot CUDA GPU thread function
__global__ void Mandelbrot0(uchar4 *dst, const int imageW, const int imageH, const int crunch, const float xOff, const float yOff, const float scale, const uchar4 colors, const int gridWidth, const int numBlocks, const uchar4 color_mod) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // Calculate the location
      const float xPos = (float)ix * scale + xOff;
      const float yPos = (float)iy * scale + yOff;

      // Calculate the Mandelbrot index for the current location
      int m = CalcMandelbrot(xPos, yPos, crunch);
      //            int m = blockIdx.x;         // uncomment to see scheduling
      //            order
      m = m > 0 ? crunch - m : 0;

      // Convert the Mandelbrot index into a color
      uchar4 color;

      if (m) {
        color.x = m * colors.x + color_mod.x;
        color.y = m * colors.y + color_mod.y;
        color.z = m * colors.z + color_mod.z;
      } else {
        color.x = 0;
        color.y = 0;
        color.z = 0;
      }

      // Output the pixel
      //color.w = 0;
      dst[imageW*iy + ix] = color;

      dst[imageW*iy + iy] = color_mod; // Straight line down
    }
  }

}  // Mandelbrot0

// The host CPU Mandelbrot thread spawner
extern "C" void RunMandelbrot0(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff, const double scale, const uchar4 colors, const int numSMs) {
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

  uchar4 e = {static_cast<unsigned char>(rand()%256), static_cast<unsigned char>(rand()%256), static_cast<unsigned char>(rand()%256), 0};

  int numWorkerBlocks = numSMs;
  Mandelbrot0<<<numWorkerBlocks, threads>>>(dst, imageW, imageH, crunch, (float)xOff, (float)yOff, (float)scale, colors, grid.x, grid.x * grid.y, e);

  getLastCudaError("Mandelbrot0 kernel execution failed.\n");
}  // RunMandelbrot0