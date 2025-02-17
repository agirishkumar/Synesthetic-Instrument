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
    
    // Audio-reactive properties
    float frequency;     // Base frequency of the particle
    float frequency_magnitude; // Current magnitude at particle's frequency
    float base_speed;    // Original speed before audio influence
};

// Audio influence parameters
struct AudioParams {
    float bass_magnitude;
    float mid_magnitude;
    float treble_magnitude;
    float global_intensity;
};

#endif