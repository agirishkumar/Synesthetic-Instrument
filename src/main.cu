// src/main.cu

// CUDA headers first
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// OpenGL headers (GLEW must come before other GL headers)
#define GLEW_STATIC
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Standard library
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

// Project headers 
#include "cuda_utils.h"
#include "config.h"
#include "midi_parser.h"
// CUDA-OpenGL interop resources
GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;
struct ParticleVertex {
    float x, y, z;    // position
    float r, g, b, a; // color
    float size;       // point size
};

// Shaders
const char* vertex_shader_source = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec4 aColor;
    layout (location = 2) in float aSize;
    
    uniform mat4 projection;
    uniform mat4 view;
    
    out vec4 Color;
    
    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
        gl_PointSize = aSize * 100.0;
        Color = aColor;
    }
)";

const char* fragment_shader_source = R"(
    #version 330 core
    in vec4 Color;
    out vec4 FragColor;
    
    void main() {
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        float alpha = 1.0 - length(circCoord);
        FragColor = Color * alpha;
    }
)";

// OpenGL shader compilation helper
GLuint compile_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed: " << info_log << std::endl;
        return 0;
    }
    return shader;
}

// CUDA kernel to update vertex buffer
__global__ void update_vertex_buffer(ParticleVertex* vertices, const Particle* particles, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count || !particles[idx].active) {
        vertices[idx].a = 0.0f; // Make inactive particles invisible
        return;
    }
    
    vertices[idx].x = particles[idx].position.x;
    vertices[idx].y = particles[idx].position.y;
    vertices[idx].z = particles[idx].position.z;
    
    vertices[idx].r = particles[idx].color.x;
    vertices[idx].g = particles[idx].color.y;
    vertices[idx].b = particles[idx].color.z;
    vertices[idx].a = particles[idx].color.w;
    
    vertices[idx].size = particles[idx].size;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <midi_file>\n";
        std::cout << "  or   " << argv[0] << " --live [device_name]\n";
        std::cout << "  or   " << argv[0] << " --list-devices\n";
        return 1;
    }

    std::string arg(argv[1]);
    if (arg == "--list-devices") {
        MidiParser::list_devices();
        return 0;
    }

    // Tell NVIDIA OpenGL driver to use the discrete GPU
    putenv((char*)"__NV_PRIME_RENDER_OFFLOAD=1");
    putenv((char*)"__GLX_VENDOR_LIBRARY_NAME=nvidia");

    // Initialize CUDA first
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " 
                  << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Find the best CUDA device
    int best_device = -1;
    cudaDeviceProp best_prop;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cuda_err = cudaGetDeviceProperties(&prop, i);
        if (cuda_err != cudaSuccess) continue;

        // Look for a discrete GPU
        if (prop.integrated == 0) {
            best_device = i;
            best_prop = prop;
            break;
        }
    }

    if (best_device == -1 && device_count > 0) {
        // If no discrete GPU found, use the first available device
        best_device = 0;
        cuda_err = cudaGetDeviceProperties(&best_prop, best_device);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to get device properties: " 
                      << cudaGetErrorString(cuda_err) << std::endl;
            return -1;
        }
    }

    if (best_device == -1) {
        std::cerr << "No CUDA device found" << std::endl;
        return -1;
    }

    cuda_err = cudaSetDevice(best_device);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " 
                  << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    std::cout << "Using CUDA device " << best_device << ": " 
              << best_prop.name << std::endl;
    std::cout << "Compute capability: " << best_prop.major << "." 
              << best_prop.minor << std::endl;



    // Initialize GLFW and OpenGL
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "MIDI Visualizer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

     // Print OpenGL info
     std::cout << "OpenGL Version: " 
     << glGetString(GL_VERSION) << std::endl;
    std::cout << "OpenGL Vendor: " 
        << glGetString(GL_VENDOR) << std::endl;
    std::cout << "OpenGL Renderer: " 
        << glGetString(GL_RENDERER) << std::endl;

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Set up CUDA-GL interop
    cuda_err = cudaGLSetGLDevice(best_device);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to set GL device: " 
                  << cudaGetErrorString(cuda_err) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Create and bind VBO

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(ParticleVertex), nullptr, GL_DYNAMIC_DRAW);

    // Register VBO with CUDA
    cuda_err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, 
        vbo,
        cudaGraphicsMapFlagsWriteDiscard);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to register GL buffer: " 
                << cudaGetErrorString(cuda_err) << std::endl;
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Create and compile shaders
    GLuint vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER);
    GLuint fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER);
    
    GLuint shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // Create vertex buffer
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(ParticleVertex), nullptr, GL_DYNAMIC_DRAW);

    // Set up vertex attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)(7 * sizeof(float)));

    // Register VBO with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    // Initialize CUDA particle system
    Particle* d_particles;
    CUDA_CHECK(cudaMalloc(&d_particles, MAX_PARTICLES * sizeof(Particle)));
    CUDA_CHECK(cudaMemset(d_particles, 0, MAX_PARTICLES * sizeof(Particle)));

    // Initialize MIDI parser
    std::string device_name;
    if (arg == "--live" && argc > 2) {
        device_name = argv[2];
    }
    
    MidiParser parser(device_name);
    
    if (arg != "--live") {
        if (!parser.load_midi_file(arg)) {
            std::cerr << "Failed to load MIDI file: " << arg << std::endl;
            return 1;
        }
    } else if (!parser.is_device_found()) {
        std::cerr << "No MIDI device found" << std::endl;
        return 1;
    }

    // Enable blending for particles
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Camera setup
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0f),  // Camera position
        glm::vec3(0.0f, 0.0f, 0.0f),  // Look at point
        glm::vec3(0.0f, 1.0f, 0.0f)   // Up vector
    );

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        // Get window size for aspect ratio
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        // Update projection matrix for new aspect ratio
        float aspect = width / (float)height;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);

        // Process MIDI events and update particles
        if (arg == "--live") {
            parser.process_events(d_particles, MAX_PARTICLES);
        } else {
            parser.process_midi_file_events(d_particles, MAX_PARTICLES);
        }

        // Map OpenGL buffer object for writing from CUDA
        ParticleVertex* d_vertices;
        size_t num_bytes;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource));

        // Update vertex buffer
        int block_size = 256;
        dim3 blocks = calculate_grid_dim(MAX_PARTICLES, block_size);
        update_vertex_buffer<<<blocks, block_size>>>(d_vertices, d_particles, MAX_PARTICLES);
        CUDA_CHECK(cudaGetLastError());

        // Unmap buffer object
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glUseProgram(shader_program);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glDrawArrays(GL_POINTS, 0, MAX_PARTICLES);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Handle escape key to exit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // Check for OpenGL errors
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    CUDA_CHECK(cudaFree(d_particles));
    
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shader_program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}