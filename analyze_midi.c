#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <smf.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <midi_file>\n", argv[0]);
        return 1;
    }
    
    smf_t* smf = smf_load(argv[1]);
    if (!smf) {
        printf("Failed to load MIDI file: %s\n", argv[1]);
        return 1;
    }
    
    printf("=== MIDI File Analysis ===\n");
    printf("File: %s\n", argv[1]);
    printf("Format: %d\n", smf->format);
    printf("Number of tracks: %d\n", smf->number_of_tracks);
    printf("PPQN (ticks per quarter note): %d\n", smf->ppqn);
    
    // Process all events to find duration and count notes
    smf_event_t* event;
    double max_time = 0.0;
    double tempo = 500000.0; // default tempo in microseconds per quarter note
    int note_on_count = 0;
    int total_events = 0;
    int tempo_changes = 0;
    
    while ((event = smf_get_next_event(smf)) != NULL) {
        total_events++;
        
        // Calculate time in seconds
        double seconds = (event->time_pulses * (tempo / 1000000.0)) / smf->ppqn;
        if (seconds > max_time) {
            max_time = seconds;
        }
        
        // Check for tempo changes
        if (event->midi_buffer_length >= 3 && 
            event->midi_buffer[0] == 0xFF && 
            event->midi_buffer[1] == 0x51) {
            if (event->midi_buffer_length >= 6) {
                tempo = (event->midi_buffer[3] << 16) | 
                        (event->midi_buffer[4] << 8) | 
                        event->midi_buffer[5];
                tempo_changes++;
            }
        }
        
        // Count note on events
        if ((event->midi_buffer[0] & 0xF0) == 0x90) {
            uint8_t velocity = event->midi_buffer[2];
            if (velocity > 0) {
                note_on_count++;
            }
        }
    }
    
    printf("\n=== Timing Information ===\n");
    printf("Total duration: %.2f seconds (%.2f minutes)\n", max_time, max_time / 60.0);
    printf("Initial tempo: %.0f microseconds/quarter note (%.1f BPM)\n", 
           tempo, 60000000.0 / tempo);
    printf("Tempo changes: %d\n", tempo_changes);
    
    printf("\n=== Event Statistics ===\n");
    printf("Total events: %d\n", total_events);
    printf("Note On events (velocity > 0): %d\n", note_on_count);
    printf("Average notes per second: %.2f\n", note_on_count / max_time);
    
    // Reset and show first few events
    smf_rewind(smf);
    printf("\n=== First 20 Events ===\n");
    int event_count = 0;
    double current_tempo = 500000.0;
    while ((event = smf_get_next_event(smf)) != NULL && event_count < 20) {
        double seconds = (event->time_pulses * (current_tempo / 1000000.0)) / smf->ppqn;
        
        if (event->midi_buffer_length >= 3 && 
            event->midi_buffer[0] == 0xFF && 
            event->midi_buffer[1] == 0x51) {
            if (event->midi_buffer_length >= 6) {
                current_tempo = (event->midi_buffer[3] << 16) | 
                               (event->midi_buffer[4] << 8) | 
                               event->midi_buffer[5];
                printf("[%.3fs] Tempo change: %.0f microseconds/quarter note (%.1f BPM)\n", 
                       seconds, current_tempo, 60000000.0 / current_tempo);
            }
        } else if ((event->midi_buffer[0] & 0xF0) == 0x90) {
            uint8_t note = event->midi_buffer[1];
            uint8_t velocity = event->midi_buffer[2];
            if (velocity > 0) {
                printf("[%.3fs] Note On: Note %d (MIDI %d), Velocity %d\n", 
                       seconds, note, note, velocity);
            }
        }
        event_count++;
    }
    
    smf_delete(smf);
    return 0;
}

