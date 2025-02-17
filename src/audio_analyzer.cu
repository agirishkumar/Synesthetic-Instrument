// src/audio_analyzer.cu
#include "audio_analyzer.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// CUDA kernel to compute FFT magnitudes
__global__ void compute_magnitudes(cufftComplex* fft_data, float* magnitudes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real = fft_data[idx].x;
        float imag = fft_data[idx].y;
        magnitudes[idx] = sqrtf(real * real + imag * imag);
    }
}

// Helper function to calculate frequency
float note_to_freq(int note) {
    return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

AudioAnalyzer::AudioAnalyzer(int buffer_size)
    : buffer_size(buffer_size),
      fft_size(buffer_size / 2 + 1),
      audio_buffer(buffer_size, 0.0f),
      d_audio_buffer(nullptr),
      d_fft_output(nullptr),
      d_magnitudes(nullptr) {
    
    initialize_cuda_buffers();
    
    // Calculate frequency band indices (assuming 44.1kHz sample rate)
    float sample_rate = 44100.0f;
    float freq_resolution = sample_rate / buffer_size;
    
    bass_low = static_cast<int>(20.0f / freq_resolution);
    bass_high = static_cast<int>(150.0f / freq_resolution);
    mid_low = static_cast<int>(150.0f / freq_resolution);
    mid_high = static_cast<int>(4000.0f / freq_resolution);
    treble_low = static_cast<int>(4000.0f / freq_resolution);
    treble_high = static_cast<int>(20000.0f / freq_resolution);
}

AudioAnalyzer::~AudioAnalyzer() {
    cleanup_cuda_buffers();
}

void AudioAnalyzer::initialize_cuda_buffers() {
    // Allocate device memory
    cudaMalloc(&d_audio_buffer, buffer_size * sizeof(float));
    cudaMalloc(&d_fft_output, fft_size * sizeof(cufftComplex));
    cudaMalloc(&d_magnitudes, fft_size * sizeof(float));
    
    // Create FFT plan
    cufftPlan1d(&fft_plan, buffer_size, CUFFT_R2C, 1);
}

void AudioAnalyzer::cleanup_cuda_buffers() {
    if (d_audio_buffer) cudaFree(d_audio_buffer);
    if (d_fft_output) cudaFree(d_fft_output);
    if (d_magnitudes) cudaFree(d_magnitudes);
    cufftDestroy(fft_plan);
}

void AudioAnalyzer::process_midi_note(int note, float velocity, float duration) {
    float freq = note_to_freq(note);
    float amplitude = velocity;
    float phase = 0.0f;
    float sample_rate = 44100.0f;
    
    // Generate simple sine wave for the note
    for (int i = 0; i < buffer_size; i++) {
        phase += 2.0f * M_PI * freq / sample_rate;
        if (phase > 2.0f * M_PI) phase -= 2.0f * M_PI;
        
        // Add harmonics for richer sound
        float sample = amplitude * sinf(phase);  // Fundamental
        sample += 0.5f * amplitude * sinf(2 * phase);  // 1st harmonic
        sample += 0.25f * amplitude * sinf(3 * phase); // 2nd harmonic
        
        // Apply simple envelope
        float t = static_cast<float>(i) / buffer_size;
        float envelope = expf(-t * 5.0f);
        
        audio_buffer[i] = sample * envelope;
    }
}

void AudioAnalyzer::analyze() {
    // Copy audio data to device
    cudaMemcpy(d_audio_buffer, audio_buffer.data(), 
               buffer_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform FFT
    cufftExecR2C(fft_plan, d_audio_buffer, d_fft_output);
    
    // Compute magnitudes
    int block_size = 256;
    int num_blocks = (fft_size + block_size - 1) / block_size;
    compute_magnitudes<<<num_blocks, block_size>>>(d_fft_output, d_magnitudes, fft_size);
    
    cudaDeviceSynchronize();
}

float AudioAnalyzer::get_bass_magnitude() const {
    float sum = 0.0f;
    float max_val = 0.0f;
    
    // Copy segment of magnitudes to host
    std::vector<float> magnitudes(bass_high - bass_low + 1);
    cudaMemcpy(magnitudes.data(), d_magnitudes + bass_low,
               magnitudes.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (float mag : magnitudes) {
        sum += mag;
        max_val = std::max(max_val, mag);
    }
    
    return max_val > 0.0f ? sum / (magnitudes.size() * max_val) : 0.0f;
}

float AudioAnalyzer::get_mid_magnitude() const {
    float sum = 0.0f;
    float max_val = 0.0f;
    
    std::vector<float> magnitudes(mid_high - mid_low + 1);
    cudaMemcpy(magnitudes.data(), d_magnitudes + mid_low,
               magnitudes.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (float mag : magnitudes) {
        sum += mag;
        max_val = std::max(max_val, mag);
    }
    
    return max_val > 0.0f ? sum / (magnitudes.size() * max_val) : 0.0f;
}

float AudioAnalyzer::get_treble_magnitude() const {
    float sum = 0.0f;
    float max_val = 0.0f;
    
    std::vector<float> magnitudes(treble_high - treble_low + 1);
    cudaMemcpy(magnitudes.data(), d_magnitudes + treble_low,
               magnitudes.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (float mag : magnitudes) {
        sum += mag;
        max_val = std::max(max_val, mag);
    }
    
    return max_val > 0.0f ? sum / (magnitudes.size() * max_val) : 0.0f;
}