// src/midi_processor.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#include "midi_parser.h"
#include "particle_kernels.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <chrono>

MidiParser::MidiParser(const std::string& device_name) 
    : midi_stream(nullptr), device_found(false), smf(nullptr), 
      current_event(nullptr), current_time(0.0), tempo(500000.0), 
      particle_index(0), audio_analyzer(new AudioAnalyzer(2048)) {
    
    if (!device_name.empty()) {
        initialize_midi();
        
        int device_count = Pm_CountDevices();
        int device_id = -1;
        
        for(int i = 0; i < device_count; ++i) {
            const PmDeviceInfo* info = Pm_GetDeviceInfo(i);
            if(info->input && std::string(info->name).find(device_name) != std::string::npos) {
                device_id = i;
                break;
            }
        }

        if(device_id != -1) {
            PmError err = Pm_OpenInput(&midi_stream, device_id, nullptr, 
                                     MIDI_BUFFER_SIZE, nullptr, nullptr);
            
            if(err == pmNoError) {
                device_found = true;
            } else {
                cleanup_midi();
                throw std::runtime_error("Failed to open MIDI input");
            }
        }
    }
}

MidiParser::~MidiParser() {
    cleanup_midi();
    delete audio_analyzer;
}

void MidiParser::initialize_midi() {
    if(Pm_Initialize() != pmNoError) {
        throw std::runtime_error("Failed to initialize PortMidi");
    }
}

void MidiParser::cleanup_midi() {
    if(midi_stream) {
        Pm_Close(midi_stream);
        midi_stream = nullptr;
    }
    
    if(smf) {
        smf_delete(smf);
        smf = nullptr;
    }
    
    Pm_Terminate();
}

void MidiParser::list_devices() {
    Pm_Initialize();
    
    int device_count = Pm_CountDevices();
    std::cout << "Available MIDI input devices:\n";
    
    for(int i = 0; i < device_count; ++i) {
        const PmDeviceInfo* info = Pm_GetDeviceInfo(i);
        if(info->input) {
            std::cout << i << ": " << info->name << 
                        " (interface: " << info->interf << ")\n";
        }
    }
    
    Pm_Terminate();
}

void MidiParser::process_events(Particle* particles, int max_particles) {
    if (!device_found || !particles) return;

    static auto last_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float delta_time = std::chrono::duration<float>(now - last_time).count();
    last_time = now;

    // Create and update audio parameters
    AudioParams audio_params;
    audio_analyzer->analyze();
    audio_params.bass_magnitude = audio_analyzer->get_bass_magnitude();
    audio_params.mid_magnitude = audio_analyzer->get_mid_magnitude();
    audio_params.treble_magnitude = audio_analyzer->get_treble_magnitude();
    audio_params.global_intensity = (audio_params.bass_magnitude + 
                                   audio_params.mid_magnitude + 
                                   audio_params.treble_magnitude) / 3.0f;

    // Allocate device memory for audio parameters
    AudioParams* d_audio_params;
    cudaMalloc(&d_audio_params, sizeof(AudioParams));
    cudaMemcpy(d_audio_params, &audio_params, sizeof(AudioParams), cudaMemcpyHostToDevice);

    // Update particles with audio parameters
    int threadsPerBlock = 256;
    int numBlocks = (max_particles + threadsPerBlock - 1) / threadsPerBlock;
    update_particles<<<numBlocks, threadsPerBlock>>>(particles, max_particles, delta_time, d_audio_params);
    cudaDeviceSynchronize();
    
    // Process MIDI events
    PmEvent buffer[MIDI_BUFFER_SIZE];
    int count = Pm_Read(midi_stream, buffer, MIDI_BUFFER_SIZE);
    
    if(count > 0) {
        for(int i = 0; i < count; ++i) {
            PmMessage msg = buffer[i].message;
            uint8_t status = Pm_MessageStatus(msg);
            uint8_t channel = status & 0x0F;
            if((status & 0xF0) == 0x90) { // Note On event
                uint8_t note = Pm_MessageData1(msg);
                float velocity = Pm_MessageData2(msg) / 127.0f;
                
                if(velocity > 0.0f) {
                    // Process note for audio analysis
                    audio_analyzer->process_midi_note(note, velocity, PARTICLE_LIFETIME);
                    
                    int particles_per_note = 128;
                    int start_idx = (particle_index % (max_particles / particles_per_note)) * particles_per_note;
                    
                    if (start_idx + particles_per_note <= max_particles) {
                        dim3 noteBlocks(4);
                        dim3 noteThreads(32);
                        handle_note_event<<<noteBlocks, noteThreads>>>(
                            particles, note, velocity, start_idx, particles_per_note, 
                            channel, d_audio_params
                        );
                        cudaDeviceSynchronize();
                        
                        particle_index++;
                    }
                }
            }
        }
    }

    // Clean up
    cudaFree(d_audio_params);
}


bool MidiParser::load_midi_file(const std::string& filename) {
    if (smf != nullptr) {
        smf_delete(smf);
    }
    
    smf = smf_load(filename.c_str());
    if (!smf) {
        std::cerr << "Failed to load MIDI file: " << filename << std::endl;
        return false;
    }
    
    reset();
    return true;
}

void MidiParser::reset() {
    if (smf) {
        current_event = nullptr;
        current_time = 0.0;
        particle_index = 0;
        smf_rewind(smf);
        start_time = std::chrono::high_resolution_clock::now();
    }
}

void MidiParser::process_midi_file_events(Particle* particles, int max_particles) {
    if (!smf || !particles) return;

    static auto last_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float delta_time = std::chrono::duration<float>(now - last_time).count();
    last_time = now;

    // Create and update audio parameters
    AudioParams audio_params;
    audio_analyzer->analyze();
    audio_params.bass_magnitude = audio_analyzer->get_bass_magnitude();
    audio_params.mid_magnitude = audio_analyzer->get_mid_magnitude();
    audio_params.treble_magnitude = audio_analyzer->get_treble_magnitude();
    audio_params.global_intensity = (audio_params.bass_magnitude + 
                                   audio_params.mid_magnitude + 
                                   audio_params.treble_magnitude) / 3.0f;

    // Allocate device memory for audio parameters
    AudioParams* d_audio_params;
    cudaMalloc(&d_audio_params, sizeof(AudioParams));
    cudaMemcpy(d_audio_params, &audio_params, sizeof(AudioParams), cudaMemcpyHostToDevice);

    // Update particles
    int threadsPerBlock = 256;
    int numBlocks = (max_particles + threadsPerBlock - 1) / threadsPerBlock;
    update_particles<<<numBlocks, threadsPerBlock>>>(particles, max_particles, delta_time, d_audio_params);
    cudaDeviceSynchronize();
    
    // Process MIDI file events
    auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    current_time = std::chrono::duration<double>(elapsed).count();
    
    while ((current_event = smf_get_next_event(smf)) != nullptr) {
        double seconds = (current_event->time_pulses * (tempo / 1000000.0)) / smf->ppqn;
        
        if (seconds > current_time) {
            break;
        }
        
        if ((current_event->midi_buffer[0] & 0xF0) == 0x90) {
            uint8_t channel = current_event->midi_buffer[0] & 0x0F;
            uint8_t note = current_event->midi_buffer[1];
            float velocity = current_event->midi_buffer[2] / 127.0f;
            
            if (velocity > 0.0f) {
                // Process note for audio analysis
                audio_analyzer->process_midi_note(note, velocity, PARTICLE_LIFETIME);
                
                int particles_per_note = 128;
                int start_idx = (particle_index % (max_particles / particles_per_note)) * particles_per_note;
                
                if (start_idx + particles_per_note <= max_particles) {
                    dim3 noteBlocks(4);
                    dim3 noteThreads(32);
                    handle_note_event<<<noteBlocks, noteThreads>>>(
                        particles, note, velocity, start_idx, particles_per_note, 
                        channel, d_audio_params
                    );
                    cudaDeviceSynchronize();
                    
                    particle_index++;
                }
            }
        }
    }

    // Clean up
    cudaFree(d_audio_params);
    
    if (smf_get_next_event(smf) == nullptr) {
        reset();
    }
}