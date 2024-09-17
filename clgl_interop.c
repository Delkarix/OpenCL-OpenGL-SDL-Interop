// Compile with:
// clang clgl_interop.c -lSDL3 -lOpenCL -o clgl_interop -lglut -lGLU -lGL -lGLEW
#include <SDL3/SDL.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>

#define WIDTH 640
#define HEIGHT 480

int finish = 0;

const char* kernel_str =
"kernel void color_image(write_only image2d_t img, float r, float g, float b) {"
"   int id = get_global_id(0);"
"   write_imagef(img, (int2)(id, id), (float4)(r, g, b, 1));"
"}";

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GL_RGBA);
    
    SDL_Window* window;
    //SDL_Renderer* renderer;

    // Initialize SDL and test for success
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("[ERROR SDL_INIT_VIDEO] %s\n", SDL_GetError());
        exit(1);
    }

    // Create window and test for success
    window = SDL_CreateWindow("SDL Test", WIDTH, HEIGHT, SDL_WINDOW_OPENGL);
    if (!window) {
        SDL_Log("[ERROR SDL_CreateWindow()] %s\n", SDL_GetError());
        exit(1);
    }
    SDL_GLContext glcontext = SDL_GL_CreateContext(window);
    glewInit();

    printf("%d\n", SDL_GetWindowProperties(window));
    void* xdisplay = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
    if (xdisplay) {
        SDL_Log("Retrieved X11 Display Pointer.\n");
    }
    else {
        SDL_Log("Failed to acquire X11 Display Pointer: %s\n", SDL_GetError());
        return -1;
    }

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WIDTH, HEIGHT, 0);
    glMatrixMode(GL_MODELVIEW);
    glEnable( GL_TEXTURE_2D );

    SDL_GL_SetSwapInterval(0);



    // Create the renderer and test for success
    /*SDL_Renderer* renderer = SDL_CreateRenderer(window, "opengl");
    if (!renderer) {
        SDL_Log("[ERROR SDL_CreateRenderer()] %s\n", SDL_GetError());
        exit(1);
    }

    // Clear the window
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);*/

    // Make the window visible
    SDL_ShowWindow(window);

    // Create primary buffer
    /*SDL_PropertiesID sdl_prop = SDL_CreateProperties();
    SDL_SetNumberProperty(sdl_prop, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_ABGR8888);
    SDL_SetNumberProperty(sdl_prop, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_TARGET | SDL_TEXTUREACCESS_STREAMING);
    SDL_SetNumberProperty(sdl_prop, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, WIDTH);
    SDL_SetNumberProperty(sdl_prop, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, HEIGHT);
    SDL_SetNumberProperty(sdl_prop, SDL_PROP_TEXTURE_CREATE_OPENGL_TEXTURE_NUMBER, 0);
    SDL_Texture* main_buff = SDL_CreateTextureWithProperties(renderer, sdl_prop);*/
    
    //SDL_Texture* main_buff = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_TARGET | SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    
    

    // cl_mem texture_image = clCreateFromGLTexture2D();
    //SDL_Renderer* rend = SDL_GetRendererFromTexture(main_buff);

    // SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
    //SDL_RenderClear(rend);
    // SDL_SetRenderDrawColor(rend, 255, 0, 0, 255);
    // SDL_RenderLine(rend, 0, 0, 100, 100);

    SDL_Color* pix = NULL;

    // Get platform and device information
    cl_context context;
    cl_command_queue command_queue;
    //cl_program program;
    cl_kernel kernel;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (!ret) {
        SDL_Log("Detected platform id: %d", platform_id);
    }
    else {
        SDL_Log("Could not detect the platform. Error code: %d", ret);
        return -1;
    }
    SDL_Log("Platform count: %d\n", ret_num_platforms);
    unsigned char str[2048] = {0};
    size_t str_size = sizeof(str);
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 2048, str, &str_size);
    SDL_Log("Platform Name: %s\n", str);

    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    if (!ret) {
        SDL_Log("Detected device id: %d", device_id);
    }
    else {
        SDL_Log("Could not detect the device. Error code: %d", ret);
        return -1;
    }

        SDL_Log("Device count: %d\n", ret_num_devices);

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
        CL_GL_CONTEXT_KHR, (cl_context_properties)glcontext,
        CL_GLX_DISPLAY_KHR, (cl_context_properties)xdisplay,
        0
    };
    size_t _s;

cl_int (*_clGetGLContextInfoKHR)(
    const cl_context_properties* properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) = clGetExtensionFunctionAddressForPlatform(platform_id, "clGetGLContextInfoKHR");

    ret = _clGetGLContextInfoKHR(props, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(device_id), &device_id, &_s);
    if (!ret || _s > 0) {
        SDL_Log("Retrieved GL Context info");
    }
    else {
        SDL_Log("Failed to retrieve GL Context info. Error code: %d", ret);
        return -1;
    }
 
    // Create an OpenCL context
    context = clCreateContext(props, 1, &device_id, NULL, NULL, &ret);
    if (!ret) {
        SDL_Log("Successfully created a context");
    }
    else {
        SDL_Log("Failed to create a context. Error code: %d", ret);
        return -1;
    }

    //clGetGLContextInfoKHR_fn pclGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clGetGLContextInfoKHR");
    //pclGetGLContextInfoKHR(props, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(device_id), &device_id, NULL);

 
    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (!ret) {
        SDL_Log("Created a command queue");
    }
    else {
        SDL_Log("Failed to create a command queue. Error code: %d", ret);
        return -1;
    }

    //printf("%s\n", glGetString(GL_VERSION));
    
    cl_GLuint texture_id; // = SDL_GetNumberProperty(SDL_GetTextureProperties(main_buff), SDL_PROP_TEXTURE_OPENGL_TEXTURE_NUMBER, 0);
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    //cl_GLuint frame_buffer;
    //cl_GLuint colorRenderbuffer;
    
    /*glGenFramebuffers(1, &frame_buffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
    glGenRenderbuffers(1, &colorRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, WIDTH, HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderbuffer);*/

    /*
    if (!texture_id) {
        SDL_Log("Failed to get the texture ID: %d\n", texture_id);
        return -1;
    }
    else {
        SDL_Log("Texture ID: %d\n", texture_id);
    }*/
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //SDL_Color tex_pre[WIDTH*HEIGHT];
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
    glTexSubImage2D(GL_TEXTURE_2D, 1, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT, NULL);
    //glGenerateMipmap(GL_TEXTURE_2D);

    //cl_mem texture_buff = clCreateFromGLRenderbuffer(context, CL_MEM_READ_WRITE, colorRenderbuffer, &ret);
    cl_mem texture_buff = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture_id, &ret);
    if (!ret) {
        SDL_Log("Acquired the OpenGL texture.");
    }
    else {
        SDL_Log("Failed to acquire the OpenGL texture. Error code: %d", ret);
        return -1;
    }
    //printf("test\n");

    _s = strlen(kernel_str);
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_str, (const size_t *)&_s, &ret);
    if (!ret) {
        SDL_Log("Created the program");
    }
    else {
        SDL_Log("Failed to create the program. Error code: %d", ret);
        return -1;
    }
 
    // Build the program
    char build_options[256];
    snprintf(build_options, 256, "-cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -D WIDTH=%d -D HEIGHT=%d", WIDTH, HEIGHT); // Format the macros
    ret = clBuildProgram(program, 1, &device_id, build_options, NULL, NULL);
    if (!ret) {
        SDL_Log("Successfully built the program");
    }
    else {
        SDL_Log("Failed to build the program. Error code: %d", ret);
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char *log = SDL_malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        SDL_Log("%s", log);
        SDL_free(log);
        return -1;
    }
 
    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "color_image", &ret);
    if (!ret) {
        SDL_Log("Created the color_image kernel");
    }
    else {
        SDL_Log("Failed to create the color_image kernel. Error code: %d", ret);
        return -1;
    }

    ret = clEnqueueAcquireGLObjects(command_queue, 1, &texture_buff, 0, NULL, NULL);
    if (ret) {
        SDL_Log("Failed to acquire the texture object for OpenCL\n");
        return -1;
    }
    int curr_ticks_fps = SDL_GetTicks();
    int fps = 0;
    int frames = 0;
    // Continuously listen for events.
    SDL_Event event;
    while (!finish) {
        // Measure FPS
        int ticks_atm = SDL_GetTicks();
        if (ticks_atm >= curr_ticks_fps + 1000) {
            fps = frames;
            frames = 0;
            curr_ticks_fps = ticks_atm;

            printf("FPS: %d\n", fps);
        }

        // SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        // SDL_RenderClear(renderer);

        /*int pitch;
        SDL_LockTexture(main_buff, NULL, &pix, &pitch);
        for (int i = 0; i < WIDTH*HEIGHT; i++) {
            pix[i].r = rand() % 256;
            pix[i].g = rand() % 256;
            pix[i].b = rand() % 256;
            pix[i].a = rand() % 256;

        }
        SDL_UnlockTexture(main_buff);*/

        // Continuously process all events
        while (SDL_PollEvent(&event)) {
            // Test for the different types of events
            switch (event.type) {
                // Test for mouse motion
                /*
                case SDL_EVENT_MOUSE_MOTION:
                    SDL_RenderPoint(rend, event.motion.x, event.motion.y);
                    break;
                */
               
                // Test for keypresses
                case SDL_EVENT_KEY_DOWN:
                    // Process keypresses
                    switch (event.key.scancode) {
                        case SDL_SCANCODE_Q:
                            finish = 1;
                            break;
                    }

                    break;
            }
        }

        

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &texture_buff);
        float r = (float)(SDL_rand(128))/(SDL_rand(128) + 128);
        float g = (float)(SDL_rand(128))/(SDL_rand(128) + 128);
        float b = (float)(SDL_rand(128))/(SDL_rand(128) + 128);
        ret = clSetKernelArg(kernel, 1, sizeof(int), &r);
        ret = clSetKernelArg(kernel, 2, sizeof(int), &g);
        ret = clSetKernelArg(kernel, 3, sizeof(int), &b);
        if (ret) {
            SDL_Log("Failed to set kernel arg. Error code: %d", ret);
            return -1;
        }
        size_t global_size = 256;
        size_t local_size = 1;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if (ret) {
            SDL_Log("Failed to enqueue the kernel. Error code: %d", ret);
            return -1;
        }

        //SDL_UpdateWindowSurface(window);
        //SDL_RenderTexture(renderer, main_buff, NULL, NULL);
        //SDL_RenderPresent(renderer);

        
        //glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glBegin( GL_QUADS );
            glTexCoord2f( 0.f, 0.f ); glVertex2f(           0.f,            0.f );
            glTexCoord2f( 1.f, 0.f ); glVertex2f( WIDTH,            0.f );
            glTexCoord2f( 1.f, 1.f ); glVertex2f( WIDTH, HEIGHT );
            glTexCoord2f( 0.f, 1.f ); glVertex2f(           0.f, HEIGHT );
        glEnd();
        SDL_GL_SwapWindow(window);
        frames++;
    }

    ret = clEnqueueReleaseGLObjects(command_queue, 1, &texture_buff, 0, NULL, NULL);

    // Destroy everything
    clFlush(command_queue);
    clFinish(command_queue);
    //clReleaseMemObject(texture_buff);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    //SDL_DestroyTexture(main_buff);
    //SDL_DestroyRenderer(renderer);
    //glDeleteRenderbuffers(1, &colorRenderbuffer);
    //glDeleteFramebuffers(1, &frame_buffer);
    glDeleteTextures(1, &texture_id);
    SDL_GL_DestroyContext(glcontext);
    SDL_DestroyWindow(window);

    // Quit program (uninitialize SDL)
    SDL_Quit();
}
