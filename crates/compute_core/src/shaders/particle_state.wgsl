// Particle state flags and the reversible resting state. Needs sim_settings.

// Fastest particle speed (m/s) taken as physical. Dense flow avalanches reach
// about 60-80 m/s and the integer atomic grid holds momentum up to
// MAX_FLOW_VELOCITY = 100 m/s for a full MAX_FLOW_THICKNESS node (faster for
// thinner nodes); a faster particle comes from a numerical blow-up (e.g. a cliff of no-data elevations) and is stopped.
// Mirrors compute_core::MAX_PARTICLE_SPEED.
const MAX_PARTICLE_SPEED: f32 = 150.0;
// stopped for exceeding MAX_PARTICLE_SPEED
const PARTICLE_UNPHYSICAL_SPEED: u32 = 1u << 25u;
// reversible low-speed state; the low bits (below PARTICLE_UNPHYSICAL_SPEED)
// hold the rest start step while resting and the consecutive slow-step count
// while moving
const PARTICLE_RESTING: u32 = 1u << 26u;
// consecutive slow steps before a particle rests
const REST_STEPS: u32 = 20u;
const PARTICLE_FLYING: u32 = 1u << 27u;
const PARTICLE_OUT_OF_BOUNDS: u32 = 1u << 28u;
const PARTICLE_IS_NAN: u32 = 1u << 29u;
const PARTICLE_OUT_OF_DEM_DATA: u32 = 1u << 30u;
const PARTICLE_STOPPED: u32 = 1u << 31u;

// Reversible resting: a particle rests after REST_STEPS consecutive slow steps
// and wakes once the gathered grid speed exceeds twice the threshold. Resting
// particles keep depositing mass but no momentum and do not move, which removes
// sub-threshold creep inside deposits. Returns the new state: its low bits hold
// the slow-step counter while moving and the rest start step while resting.
fn next_rest_state(previous_state: u32, speed: f32, timestep: u32) -> u32 {
    let low_mask = PARTICLE_UNPHYSICAL_SPEED - 1u;
    var resting = (previous_state & PARTICLE_RESTING) != 0u;
    var low_bits = previous_state & low_mask;
    if resting {
        if speed > 2.0 * sim_settings.velocity_threshold {
            resting = false;
            low_bits = 0u;
        }
    } else if speed < sim_settings.velocity_threshold {
        low_bits = min(low_bits + 1u, REST_STEPS);
        if low_bits >= REST_STEPS {
            resting = true;
            low_bits = timestep & low_mask;
        }
    } else {
        low_bits = 0u;
    }
    return (previous_state & ~(PARTICLE_RESTING | low_mask))
        | select(0u, PARTICLE_RESTING, resting)
        | low_bits;
}
