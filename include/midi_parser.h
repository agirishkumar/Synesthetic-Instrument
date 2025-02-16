// include/midi_parser.h
#ifndef MIDI_PARSER_H
#define MIDI_PARSER_H

// Include CUDA headers first if needed
#include <cuda_runtime.h>

// Undefine any conflicting macros
#undef __has_attribute

// Standard library
#include <string>
#include <chrono>

// External libraries
#include <portmidi.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <smf.h>
#ifdef __cplusplus
}
#endif

#include "particle.h"

class MidiParser {
public:
    MidiParser(const std::string& device_name = "");
    ~MidiParser();
    
    void process_events(Particle* particles, int max_particles);
    bool load_midi_file(const std::string& filename);
    void process_midi_file_events(Particle* particles, int max_particles);
    void reset();
    
    static void list_devices();
    bool is_device_found() const { return device_found; }

private:
    PortMidiStream* midi_stream;
    bool device_found;
    smf_t* smf;
    smf_event_t* current_event;
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    double current_time;
    double tempo;
    
    int particle_index;
    
    void initialize_midi();
    void cleanup_midi();
};

#endif