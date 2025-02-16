// include/particle.h
#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

enum EffectType {
    EFFECT_SPIRAL = 0,
    EFFECT_FOUNTAIN,
    EFFECT_WAVE,
    EFFECT_EXPLOSION,
    NUM_EFFECTS
};

struct Particle {
    float3 position;     // Position in 3D space
    float3 velocity;     // Velocity vector
    float4 color;        // RGBA color
    float size;          // Current particle size
    float initial_size;  // Initial particle size
    float lifetime;      // Remaining lifetime
    float initial_lifetime; // Starting lifetime
    bool active;         // Whether particle is active
    EffectType effect_type; // Type of visual effect
};

#endif