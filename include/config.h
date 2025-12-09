// include/config.h
#pragma once

// Maximum number of particles that can be simulated
constexpr int MAX_PARTICLES = 32768;  

// MIDI configuration
constexpr int MIDI_BUFFER_SIZE = 1024;

// Visual settings
constexpr float PARTICLE_LIFETIME = 8.0f;
constexpr float MIN_PARTICLE_SIZE = 0.01f;
constexpr float MAX_PARTICLE_SIZE = 0.12f;

// Physics constants
constexpr float GRAVITY_STRENGTH = -0.1f;
constexpr float VELOCITY_DAMPING = 0.99f;
constexpr float ALPHA_DECAY_RATE = 0.995f;

// CUDA settings
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int PARTICLES_PER_NOTE = 512;  

// Effect settings
constexpr float TURBULENCE_SCALE = 0.02f;
constexpr float WAVE_FREQUENCY = 2.0f;
constexpr float SPIRAL_RADIUS = 0.75f;
constexpr float EXPLOSION_FORCE = 3.0f;