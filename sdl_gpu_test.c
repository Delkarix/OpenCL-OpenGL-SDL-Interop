// Compile with:
// clang sdl_gpu_test.c -lSDL3 -o sdl_gpu_test

#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480

static int finish = 0;
const char* kernel_str =
"kernel void color_image(write_only image2d_t img, float r, float g, float b) {"
"   int id = get_global_id(0);"
"   write_imagef(img, (int2)(id, id), (float4)(r, g, b, 1));"
"}";

struct color {
    float r;
    float g;
    float b;
};

int main() {
    SDL_Window* window;
    SDL_GPUDevice* gpu;
    SDL_GPUCommandBuffer* command_buffer;
    SDL_GPUTexture* texture;
    SDL_GPUTexture* render_texture;
    // SDL_GPUTransferBuffer* transfer_buffer;
    // SDL_GPUCopyPass* copy_pass;
    SDL_GPUComputePipeline* compute_pipeline;
    SDL_GPUComputePass* compute_pass;

    // read the kernel
    FILE* f = fopen("color_line_kernel.spv", "r");
    void* code = malloc(32768);

    size_t code_size = fread(code, 1, 32768, f);

    fclose(f);
    
    // Initialize SDL and test for success
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        printf("[ERROR SDL_INIT_VIDEO] %s\n", SDL_GetError());
        exit(1);
    }

    // Create window and test for success
    window = SDL_CreateWindow("SDL Test", WIDTH, HEIGHT, 0);
    if (!window) {
        printf("[ERROR SDL_CreateWindow()] %s\n", SDL_GetError());
        exit(1);
    }

    // Access the GPU device
    gpu = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, 1, NULL);
    if (!gpu) {
        printf("[ERROR SDL_CreateGPUDevice()] %s\n", SDL_GetError());
        exit(1);
    }

    // Claim ownership over the window so we can get its texture
    if (!SDL_ClaimWindowForGPUDevice(gpu, window)) {
        printf("[ERROR SDL_ClaimWindowForGPUDevice()] %s\n", SDL_GetError());
        exit(1);
    }

    if (!SDL_SetGPUSwapchainParameters(gpu, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_MAILBOX)) {
        printf("[ERROR SDL_SetGPUSwapchainParameters()] %s\n", SDL_GetError());
        exit(1);
    }

    SDL_GPUTextureCreateInfo texinfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GetGPUSwapchainTextureFormat(gpu, window),
        .usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = WIDTH,
        .height = HEIGHT,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = 0,
        .props = 0
    };
    render_texture = SDL_CreateGPUTexture(gpu, &texinfo);
    if (!render_texture) {
        printf("[ERROR SDL_CreateGPUTexture()] %s\n", SDL_GetError());
        exit(1);
    }

    // SDL_GPUBufferCreateInfo buffer_info = {
        // .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        // .size = 4,
        // .props = 0
    // };
    // buffers[0] = SDL_CreateGPUBuffer(gpu, &buffer_info);
    // buffers[1] = SDL_CreateGPUBuffer(gpu, &buffer_info);
    // buffers[2] = SDL_CreateGPUBuffer(gpu, &buffer_info);

    /*
    if (!SDL_WindowSupportsGPUSwapchainComposition(gpu, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR)) {
        printf("[ERROR SDL_WindowSupportsGPUSwapchainComposition()] SDL_LINEAR not supported.\n"); // Maybe we need to set?
        exit(1);
    }
    */
    

    // Create a transfer buffer. This will be used to throw pixels in at our window.
    // SDL_GPUTransferBufferCreateInfo tb_create_info = {.size = WIDTH*HEIGHT*4, .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD};
    // SDL_GPUTransferBufferCreateInfo tb_create_info = {.size = 4, .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD};
    // transfer_buffer = SDL_CreateGPUTransferBuffer(gpu, &tb_create_info);
    // if (!transfer_buffer) {
        // printf("[ERROR SDL_CreateGPUTransferBuffer()] %s\n", SDL_GetError());
        // exit(1);
    // }

    SDL_GPUComputePipelineCreateInfo comp_pipe_info = {
        .code = code,
        .code_size = code_size,
        .entrypoint = "main",
        .format = SDL_GPU_SHADERFORMAT_SPIRV,
        
        .num_samplers = 0,
        .num_readonly_storage_textures = 0,
        .num_readonly_storage_buffers = 0,
        .num_readwrite_storage_buffers = 0,
        .num_readwrite_storage_textures = 1,
        .num_uniform_buffers = 1,
        
        .threadcount_x = 64,
        .threadcount_y = 1,
        .threadcount_z = 1,
        .props = 0
    };
    
    compute_pipeline = SDL_CreateGPUComputePipeline(gpu, &comp_pipe_info);
    if (!compute_pipeline) {
        printf("[ERROR SDL_CreateGPUComputePipeline()] %s\n", SDL_GetError());
        exit(1);
    }

    int curr_ticks_fps = SDL_GetTicks();
    int fps = 0;
    int frames = 0;

    // Continuously listen for events.
    SDL_Event event;
    while (!finish) {
        int ticks_atm = SDL_GetTicks();
        if (ticks_atm >= curr_ticks_fps + 1000) {
            fps = frames;
            frames = 0;
            curr_ticks_fps = ticks_atm;

            printf("FPS: %d\n", fps);
        }
        
        // Continuously process all events
        while (SDL_PollEvent(&event)) {
            // Test for the different types of events
            switch (event.type) {
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

        // Create a new command buffer
        command_buffer = SDL_AcquireGPUCommandBuffer(gpu);
        if (!command_buffer) {
            printf("[ERROR SDL_AcquireGPUCommandBuffer()] %s\n", SDL_GetError());
            exit(1);
        }

        if (!SDL_AcquireGPUSwapchainTexture(command_buffer, window, &texture, NULL, NULL)) {
            printf("[ERROR SDL_AcquireGPUSwapchainTexture()] %s\n", SDL_GetError());
            exit(1);
        }

        /*
        char* pixels = SDL_MapGPUTransferBuffer(gpu, transfer_buffer, 0);
            if (!pixels) {
                printf("[ERROR SDL_MapGPUTransferBuffer()] %s\n", SDL_GetError());
                exit(1);
            }
            for (int i = 0; i < WIDTH*HEIGHT*4; i++) {
                pixels[i] = rand()%256;
            }
        SDL_UnmapGPUTransferBuffer(gpu, transfer_buffer);
        */

        
        SDL_GPUStorageTextureReadWriteBinding tex_read_write = {.texture = render_texture, .mip_level = 0, .layer = 0, .cycle = 1};
        compute_pass = SDL_BeginGPUComputePass(command_buffer, &tex_read_write, 1, NULL, 0);
            if (!compute_pass) {
                printf("[ERROR SDL_BeginGPUComputePass()] %s\n", SDL_GetError());
                exit(1);
            }

            SDL_BindGPUComputePipeline(compute_pass, compute_pipeline);

            // SDL_BindGPUComputeStorageBuffers(compute_pass, 0, buffers, 3);
            struct color col = {.r = (rand()%256)/256.0, .g = (rand()%256)/256.0, .b = (rand()%256)/256.0};
            SDL_PushGPUComputeUniformData(command_buffer, 0, &col, sizeof(struct color));

            

            // SDL_PushGPUComputeUniformData(command_buffer, 1, &r, 4);
            // SDL_PushGPUComputeUniformData(command_buffer, 1, &g, 4);
            // SDL_PushGPUComputeUniformData(command_buffer, 1, &b, 4);

            SDL_DispatchGPUCompute(compute_pass, 4, 1, 1);

        SDL_EndGPUComputePass(compute_pass);
        

        // Begin the copy

        /*
        copy_pass = SDL_BeginGPUCopyPass(command_buffer);
            if (!copy_pass) {
                printf("[ERROR SDL_BeginGPUCopyPass()] %s\n", SDL_GetError());
                exit(1);
            }


            SDL_GPUTransferBufferLocation trans_loc = {.transfer_buffer = transfer_buffer, .offset = 0};
            for (int i = 0; i < 3; i++) {
                    float* col = SDL_MapGPUTransferBuffer(gpu, transfer_buffer, 1);
                        if (!col) {
                            printf("[ERROR SDL_MapGPUTransferBuffer()] %s\n", SDL_GetError());
                            exit(1);
                        }
                        *col = (rand()%256)/256.0;
                    SDL_UnmapGPUTransferBuffer(gpu, transfer_buffer);
                    
                    SDL_GPUBufferRegion buf_reg = {.buffer = buffers[i], .offset = 0, .size = 4};

                    SDL_UploadToGPUBuffer(copy_pass, &trans_loc, &buf_reg, 1);
            }

            // SDL_GPUTextureRegion tex_reg = {.texture = texture, .mip_level = 0, .layer = 0, .x = 0, .y = 0, .z = 0, .w = WIDTH, .h = HEIGHT, .d = 1};

            // SDL_GPUTextureTransferInfo trans_info = {.transfer_buffer = transfer_buffer, .pixels_per_row = WIDTH, .rows_per_layer = 0, .offset = 0};
            // SDL_UploadToGPUTexture(copy_pass, &trans_info, &tex_reg, 0);

        // End the copy
        SDL_EndGPUCopyPass(copy_pass);
        */
        

        SDL_GPUBlitInfo blit_info = {
            .source.texture = render_texture,
            .source.w = WIDTH,
            .source.h = HEIGHT,
            .source.x = 0,
            .source.y = 0,
            .source.layer_or_depth_plane = 0,
            .source.mip_level = 0,
            
            .destination.texture = texture,
            .destination.w = WIDTH,
            .destination.h = HEIGHT,
            .destination.x = 0,
            .destination.y = 0,
            .destination.layer_or_depth_plane = 0,
            .destination.mip_level = 0,

            .load_op = SDL_GPU_LOADOP_DONT_CARE,
            .flip_mode = SDL_FLIP_NONE,
            .filter = SDL_GPU_FILTER_LINEAR,
            .cycle = 1
        };
        SDL_BlitGPUTexture(command_buffer, &blit_info);

        SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(command_buffer);
        // if (!SDL_SubmitGPUCommandBuffer(command_buffer)) {
        if (!fence) {
            printf("[ERROR SDL_SubmitGPUCommandBuffer()] %s\n", SDL_GetError());
            break;
        }

        SDL_WaitForGPUFences(gpu, 0, &fence, 1);
        SDL_ReleaseGPUFence(gpu, fence);

        // blit
        // command_buffer = SDL_AcquireGPUCommandBuffer(gpu);
        // SDL_BlitGPUTexture(command_buffer, &blit_info);
        // SDL_SubmitGPUCommandBuffer(command_buffer);

        frames++;
    }

    SDL_ReleaseGPUComputePipeline(gpu, compute_pipeline);
    // SDL_ReleaseGPUTransferBuffer(gpu, transfer_buffer);
    // SDL_ReleaseGPUBuffer(gpu, buffers[2]);
    // SDL_ReleaseGPUBuffer(gpu, buffers[1]);
    // SDL_ReleaseGPUBuffer(gpu, buffers[0]);
    SDL_ReleaseGPUTexture(gpu, render_texture);
    SDL_ReleaseWindowFromGPUDevice(gpu, window);
    SDL_DestroyGPUDevice(gpu);
    SDL_DestroyWindow(window);
    free(code);

    // Quit program (uninitialize SDL)
    SDL_Quit();
}
