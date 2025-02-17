// include/particle_kernels.h
#ifndef PARTICLE_KERNELS_H
#define PARTICLE_KERNELS_H

#include "particle.h"
#include "config.h"

// Update particle physics and properties
__global__ void update_particles(Particle* particles, int count, float delta_time, const AudioParams* audio_params);

// Handle MIDI note events by spawning or modifying particles
__global__ void handle_note_event(Particle* particles, int note, float velocity, 
                                int start_idx, int count, int channel, const AudioParams* audio_params);

// Reset particle system
__global__ void reset_particles(Particle* particles, int count);

#endif