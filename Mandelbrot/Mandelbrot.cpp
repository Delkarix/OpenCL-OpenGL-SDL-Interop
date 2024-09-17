// SOURCE: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/Mandelbrot

// OpenGL Graphics includes
#include "helper_gl.h"

#include <GL/freeglut.h>
// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "helper_functions.h"
#include "helper_cuda.h"

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include "Mandelbrot_kernel.h"

// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange

// Destination image on the GPU side
uchar4 *d_dst = NULL;

// Original image width and height
const int imageW = 800;
const int imageH = 600;

// Starting iteration limit
int crunch = 512;

// Starting color multipliers and random seed
uchar4 colors;

// Timer ID
StopWatchInterface *hTimer = NULL;

bool haveDoubles = true;
int numSMs = 0;   // number of multiprocessors

// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;   // FPS count for averaging
int fpsLimit = 15;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define REFRESH_DELAY 10  // ms

#define BUFFER_DATA(i) ((char *)0 + i)

void computeFPS() {
  frameCount++;
  fpsCount++;

  //if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
    snprintf(fps, sizeof(fps), "<CUDA Mandelbrot Set> %3.1f fps", ifps);
    glutSetWindowTitle(fps);
    fpsCount = 0;

    fpsLimit = (int)MAX(1.f, (float)ifps);
    sdkResetTimer(&hTimer);
  //}
}

// render Mandelbrot image using CUDA or CPU
void renderImage() {
  float timeEstimate;
  sdkResetTimer(&hTimer);

  // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst,
  // gl_PBO));
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));

  // Render anti-aliasing passes until we run out time (60fps approximately)
  const float xs = 64.5f / 128.0f;
  const float ys = 64.5f / 128.0f;

  // Get the pixel scale and offset
  const double s = 3.2 / (float)imageW;
  const double x = (xs - (double)imageW * 0.5f) * s - 0.5;
  const double y = (ys - (double)imageH * 0.5f) * s;

  // Run the mandelbrot generator

  RunMandelbrot0(d_dst, imageW, imageH, crunch, x, y, s, colors, numSMs);

  // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// OpenGL display function
void displayFunc(void) {
  sdkStartTimer(&hTimer);

  // render the Mandelbrot image
  renderImage();

  // load texture from PBO
  //  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(0.0f, 0.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, 0.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(0.0f, 1.0f);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);

  sdkStopTimer(&hTimer);
  glutSwapBuffers();

  computeFPS();
}  // displayFunc

void cleanup() {
  sdkStopTimer(&hTimer);
  sdkDeleteTimer(&hTimer);

  // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  glDeleteBuffers(1, &gl_PBO);
  glDeleteTextures(1, &gl_Tex);
}

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int) {
  int seed;

  switch (k) {
    case '\033':
    case 'q':
    case 'Q':
      printf("Shutting down...\n");
      glutDestroyWindow(glutGetWindow());
  }

}  // keyboardFunc

void timerEvent(int value) {
  if (glutGetWindow()) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  }
}

void initOpenGLBuffers(int w, int h) {
  if (gl_Tex) {
    glDeleteTextures(1, &gl_Tex);
    gl_Tex = 0;
  }

  if (gl_PBO) {
    // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &gl_PBO);
    gl_PBO = 0;
  }

  printf("Creating GL texture...\n");
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_Tex);
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  printf("Texture created.\n");

  printf("Creating PBO...\n");
  glGenBuffers(1, &gl_PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, NULL, GL_STREAM_COPY);
  // While a PBO is registered to CUDA, it can't be used
  // as the destination for OpenGL drawing calls.
  // But in our particular case OpenGL is only used
  // to display the content of the PBO, specified by CUDA kernels,
  // so we need to register/unregister it only once.

  // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
  printf("PBO created.\n");
}

void initGL(int *argc, char **argv) {
  printf("Initializing GLUT...\n");
  glutInit(argc, argv);

  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(imageW, imageH);
  glutInitWindowPosition(0, 0);
  glutCreateWindow(argv[0]);

  glutDisplayFunc(displayFunc);
  glutKeyboardFunc(keyboardFunc);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

  printf("OpenGL window created.\n");
}

void initData(int argc, char **argv) {
  // check for hardware double precision support
  int dev = 0;
  dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  numSMs = deviceProp.multiProcessorCount;

  colors.w = 0;
  colors.x = 3;
  colors.y = 5;
  colors.z = 7;
  printf("Data initialization done.\n");
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  // Set the initial parameters for either Mandelbrot and Julia sets and reset
  // all parameters
  colors.x = 3;
  colors.y = 5;
  colors.z = 7;
  crunch = 512;

  // Otherwise it succeeds, we will continue to run this sample
  initData(argc, argv);

  // Initialize OpenGL context first before the CUDA context is created.  This
  // is needed
  // to achieve optimal performance with OpenGL/CUDA interop.
  
  initGL(&argc, argv);
  glViewport(0, 0, imageW, imageH);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, 1.0, 1.0, 0);

  initOpenGLBuffers(imageW, imageH);

  sdkCreateTimer(&hTimer);
  sdkStartTimer(&hTimer);
  glutCloseFunc(cleanup);

  glutMainLoop();
}  // main
