// kernels/particle_kernels.cu
#include "particle_kernels.h"
#include <cuda_runtime.h>
#include "config.h"
#include <math.h>

__constant__ float3 GRAVITY = {0.0f, GRAVITY_STRENGTH, 0.0f};

// Helper functions for vector operations
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ void operator*=(float3& v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
}


__device__ 
float4 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float hh = fmodf(h, 360.0f) / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float m = v - c;
    
    float4 rgba = {0.0f, 0.0f, 0.0f, 1.0f};
    
    if(hh <= 1.0f)      { rgba.x = c; rgba.y = x; }
    else if(hh <= 2.0f) { rgba.x = x; rgba.y = c; }
    else if(hh <= 3.0f) { rgba.y = c; rgba.z = x; }
    else if(hh <= 4.0f) { rgba.y = x; rgba.z = c; }
    else if(hh <= 5.0f) { rgba.x = x; rgba.z = c; }
    else                { rgba.x = c; rgba.z = x; }
    
    rgba.x += m;
    rgba.y += m;
    rgba.z += m;
    
    return rgba;
}

__device__
void generate_spiral_effect(Particle* particle, int note, float velocity, float angle_offset, int pattern_idx) {
    float angle = note * (2.0f * M_PI / 127.0f) + angle_offset;
    float radius = 0.5f + velocity * 0.3f;
    float height_offset = sinf(angle * 3.0f) * 0.2f;
    
    particle->position = make_float3(
        cosf(angle) * radius,
        height_offset,
        sinf(angle) * radius
    );
    
    particle->velocity = make_float3(
        cosf(angle) * velocity * 1.5f,
        velocity * 2.0f + cosf(angle) * 0.5f,
        sinf(angle) * velocity * 1.5f
    );
}

__device__
void generate_fountain_effect(Particle* particle, int note, float velocity, float angle_offset, int pattern_idx) {
    float angle = note * (2.0f * M_PI / 127.0f) + angle_offset;
    float spread = pattern_idx * 0.2f;
    
    particle->position = make_float3(
        cosf(angle) * spread * 0.3f,
        -1.0f,
        sinf(angle) * spread * 0.3f
    );
    
    particle->velocity = make_float3(
        cosf(angle) * velocity * 2.0f,
        velocity * 4.0f,
        sinf(angle) * velocity * 2.0f
    );
}

__device__
void generate_wave_effect(Particle* particle, int note, float velocity, float angle_offset, int pattern_idx) {
    float base_x = ((float)pattern_idx / PARTICLES_PER_NOTE - 0.5f) * 2.0f;
    float wave_height = sinf(base_x * M_PI * 2.0f + angle_offset);
    
    particle->position = make_float3(
        base_x,
        wave_height * 0.3f - 0.5f,
        0.0f
    );
    
    particle->velocity = make_float3(
        velocity * cosf(wave_height) * 0.5f,
        velocity * 2.0f,
        velocity * sinf(base_x * M_PI) * 0.5f
    );
}

__device__
void generate_explosion_effect(Particle* particle, int note, float velocity, float angle_offset, int pattern_idx) {
    float phi = pattern_idx * (M_PI / PARTICLES_PER_NOTE);
    float theta = angle_offset + note * (2.0f * M_PI / 127.0f);
    
    particle->position = make_float3(0.0f, 0.0f, 0.0f);
    
    particle->velocity = make_float3(
        velocity * sinf(phi) * cosf(theta) * 3.0f,
        velocity * cosf(phi) * 3.0f,
        velocity * sinf(phi) * sinf(theta) * 3.0f
    );
}

__global__
void update_particles(Particle* particles, int count, float delta_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count || !particles[idx].active) return;

    if(particles[idx].lifetime > 0.0f) {
        // Apply custom forces based on effect type
        switch(particles[idx].effect_type) {
            case EFFECT_SPIRAL:
                particles[idx].velocity.y += sinf(particles[idx].lifetime * 5.0f) * 0.01f;
                break;
            case EFFECT_FOUNTAIN:
                particles[idx].velocity += GRAVITY * delta_time;
                break;
            case EFFECT_WAVE:
                particles[idx].velocity.x *= 0.98f;
                particles[idx].velocity.z = cosf(particles[idx].lifetime * 10.0f) * 0.1f;
                break;
            case EFFECT_EXPLOSION:
                particles[idx].velocity *= 0.99f;
                break;
        }
        
        // Update position with delta time
        float3 delta_pos = particles[idx].velocity * delta_time;
        particles[idx].position += delta_pos;
        
        // Add some turbulence
        float turbulence = sinf(particles[idx].lifetime * 4.0f + particles[idx].position.x) * 0.02f;
        particles[idx].position.x += turbulence;
        particles[idx].position.z += turbulence;
        
        // Update lifetime and color
        particles[idx].lifetime -= delta_time;
        float life_ratio = particles[idx].lifetime / particles[idx].initial_lifetime;
        particles[idx].color.w = life_ratio * 0.8f + 0.2f;
        particles[idx].size = particles[idx].initial_size * (life_ratio * 0.8f + 0.2f);
        
        // Deactivate if needed
        if(particles[idx].lifetime <= 0.0f ||
           fabsf(particles[idx].position.x) > 2.0f ||
           fabsf(particles[idx].position.y) > 2.0f ||
           fabsf(particles[idx].position.z) > 2.0f) {
            particles[idx].active = false;
        }
    }
}

__global__
void handle_note_event(Particle* particles, int note, float velocity, 
                      int start_idx, int count, int channel) {
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(local_idx >= count) return;
    
    int idx = start_idx + local_idx;
    if(velocity > 0.01f) {
        float angle_offset = local_idx * (2.0f * M_PI / count);
        
        // Choose effect type based on MIDI channel
        EffectType effect = static_cast<EffectType>(channel % NUM_EFFECTS);
        particles[idx].effect_type = effect;
        
        // Generate initial position and velocity based on effect
        switch(effect) {
            case EFFECT_SPIRAL:
                generate_spiral_effect(&particles[idx], note, velocity, angle_offset, local_idx);
                break;
            case EFFECT_FOUNTAIN:
                generate_fountain_effect(&particles[idx], note, velocity, angle_offset, local_idx);
                break;
            case EFFECT_WAVE:
                generate_wave_effect(&particles[idx], note, velocity, angle_offset, local_idx);
                break;
            case EFFECT_EXPLOSION:
                generate_explosion_effect(&particles[idx], note, velocity, angle_offset, local_idx);
                break;
        }
        
        // Set color based on note and effect type
        float hue = fmodf(note * 2.8f + effect * 90.0f, 360.0f);
        particles[idx].color = hsv_to_rgb(hue, 0.8f + velocity * 0.2f, 0.8f + velocity * 0.2f);
        
        // Set other properties
        particles[idx].initial_lifetime = PARTICLE_LIFETIME * (0.5f + velocity * 0.5f);
        particles[idx].lifetime = particles[idx].initial_lifetime;
        particles[idx].initial_size = MIN_PARTICLE_SIZE + (MAX_PARTICLE_SIZE - MIN_PARTICLE_SIZE) * velocity;
        particles[idx].size = particles[idx].initial_size;
        particles[idx].active = true;
    }
}

__global__
void reset_particles(Particle* particles, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;
    
    particles[idx].active = false;
    particles[idx].lifetime = 0.0f;
    particles[idx].color.w = 0.0f;
}