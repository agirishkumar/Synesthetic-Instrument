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
      last_processed_pulses(-1), particle_index(0), audio_analyzer(new AudioAnalyzer(2048)) {
    
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
    
    // Find initial tempo from the file (look at first few events)
    smf_event_t* event;
    tempo = 500000.0; // default tempo
    int events_checked = 0;
    while ((event = smf_get_next_event(smf)) != nullptr && events_checked < 100) {
        events_checked++;
        // Check for tempo change events (Meta Event 0xFF 0x51)
        if (event->midi_buffer_length >= 3 && 
            event->midi_buffer[0] == 0xFF && 
            event->midi_buffer[1] == 0x51) {
            if (event->midi_buffer_length >= 6) {
                tempo = (event->midi_buffer[3] << 16) | 
                        (event->midi_buffer[4] << 8) | 
                        event->midi_buffer[5];
                break; // Found first tempo, use it as initial
            }
        }
    }
    
    reset();
    return true;
}

void MidiParser::reset() {
    if (smf) {
        current_event = nullptr;
        current_time = 0.0;
        last_processed_pulses = -1;
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
    
    bool reached_end = false;
    long max_processed_pulses = last_processed_pulses;
    
    // Process all events that are due now
    // Use a lookahead window to avoid skipping events due to timing precision
    while ((current_event = smf_get_next_event(smf)) != nullptr) {
        // Only process events we haven't processed yet (avoid duplicates on rewind)
        if (last_processed_pulses >= 0 && current_event->time_pulses <= last_processed_pulses) {
            continue;
        }
        
        // Calculate time for this event
        double seconds = (current_event->time_pulses * (tempo / 1000000.0)) / smf->ppqn;
        
        // If this event is significantly in the future (> 0.2s), break and process it next frame
        // This prevents skipping events while also not processing too far ahead
        if (seconds > current_time + 0.2) {
            break;
        }
        
        // Track the maximum time_pulses we've seen in this frame
        if (current_event->time_pulses > max_processed_pulses) {
            max_processed_pulses = current_event->time_pulses;
        }
        
        // Handle tempo change events (Meta Event 0xFF 0x51) - process these immediately
        if (current_event->midi_buffer_length >= 3 && 
            current_event->midi_buffer[0] == 0xFF && 
            current_event->midi_buffer[1] == 0x51) {
            // Tempo change: bytes are in microseconds per quarter note (24-bit)
            if (current_event->midi_buffer_length >= 6) {
                tempo = (current_event->midi_buffer[3] << 16) | 
                        (current_event->midi_buffer[4] << 8) | 
                        current_event->midi_buffer[5];
            }
            continue;
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
    
    // Update last processed position to the maximum we've seen this frame
    // This ensures we process all events at the same time_pulses value in one frame
    if (max_processed_pulses > last_processed_pulses) {
        last_processed_pulses = max_processed_pulses;
    }
    
    // Check if we've reached the end of the file
    if (current_event == nullptr) {
        reached_end = true;
    }

    // Clean up
    cudaFree(d_audio_params);
    
    // Reset and loop if we've reached the end
    if (reached_end) {
        reset();
    }
}