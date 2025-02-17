#ifndef AUDIO_ANALYZER_H
#define AUDIO_ANALYZER_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

class AudioAnalyzer {
public:
    AudioAnalyzer(int buffer_size = 2048);
    ~AudioAnalyzer();

    // Process MIDI note data into audio samples
    void process_midi_note(int note, float velocity, float duration);
    
    // Run FFT analysis
    void analyze();
    
    // Get frequency band magnitudes (normalized 0-1)
    float get_bass_magnitude() const;      // 20-150 Hz
    float get_mid_magnitude() const;       // 150-4000 Hz
    float get_treble_magnitude() const;    // 4000-20000 Hz
    
    // Get raw FFT data
    const float* get_fft_data() const { return d_magnitudes; }
    int get_fft_size() const { return fft_size; }

private:
    int buffer_size;
    int fft_size;
    
    // Host buffers
    std::vector<float> audio_buffer;
    
    // Device buffers
    cufftHandle fft_plan;
    float* d_audio_buffer;
    cufftComplex* d_fft_output;
    float* d_magnitudes;
    
    // Frequency band indices
    int bass_low, bass_high;
    int mid_low, mid_high;
    int treble_low, treble_high;
    
    void initialize_cuda_buffers();
    void cleanup_cuda_buffers();
};

#endif