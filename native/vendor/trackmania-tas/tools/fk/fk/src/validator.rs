//! The controlled car, resolved from the validator's own simulation objects.
//!
//! This module is intentionally separate from [`crate::locate`]. The latter is
//! forensic tooling for finding coherent state-shaped records in arbitrary
//! captures. It cannot establish player identity. `ValidatorCar` starts at the
//! callback the `/validatepath` state machine itself invokes and follows only
//! typed ownership fields; there is no candidate enumeration or ranking.

use forkoracle::forksrv::{ForkServer, Rec};
use forkoracle::layout::Layout;
use forkoracle::procmem;

use crate::locate::{qualify2, ClockHit};

/// Build 128182 (`date=2026-05-15_18_00`) validator/player ownership layout.
#[derive(Clone, Copy, Debug)]
struct Offsets {
    controller_sim: u64,
    sim_playground: u64,
    playground_players: u64,
    playground_player_count: u64,
    participant_vehicle_class: u64,
    participant_vehicle: u64,
    vehicle_state_pos: u64,
}

const BUILD_128182: Offsets = Offsets {
    // 0x118c170: `mov [rdi+0x1a70], rcx`.
    controller_sim: 0x1a70,
    // 0x1218e3d: `mov rax,[r14+0x18]`, where r14 is the callback sim.
    sim_playground: 0x18,
    // 0x1218e41/4e: the sole validation-player vector.
    playground_players: 0x660,
    playground_player_count: 0x668,
    // 0x11a9b16..21: after the CGameVehiclePhy class check, store the class id
    // and pointer in the participant's primary vehicle slot.
    participant_vehicle_class: 0x1110,
    participant_vehicle: 0x1118,
    // CGameVehiclePhy: q(wxyz) at pos-16, world position, then velocity.
    // Writes/reads are visible at 0x11f38fe..0x11f3919 and 0x9cdb14 onward.
    vehicle_state_pos: 0x12f0,
};

/// CGameVehiclePhy's class id on build 128182.
/// Registered at 0xc3b62f as `CGameVehiclePhy`.
pub const CGAME_VEHICLE_PHY: u32 = 0x032e_2000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorCarProvenance {
    pub controller: u64,
    pub sim: u64,
    pub playground: u64,
    pub players: u64,
    pub participant: u64,
    pub vehicle: u64,
    pub state_pos: u64,
}

/// A car whose identity came from the validator's controlled-player ownership
/// chain. Its inner `Layout` is private so production callers cannot substitute
/// a state-shaped address found by a scanner.
#[derive(Clone, Debug)]
pub struct ValidatorCar {
    layout: Layout,
    provenance: ValidatorCarProvenance,
}

impl ValidatorCar {
    /// Locate the race clock, then resolve the controlled vehicle from validator
    /// ownership. The clock scan labels samples; it does not participate in car
    /// identity.
    #[allow(clippy::too_many_arguments)]
    pub fn locate(
        srv: &mut ForkServer,
        probe: usize,
        recs: &[Rec],
        start_offset_ms: i32,
        bounds: (f64, f64, f64, f64, f64, f64),
        bias_max: i64,
        verbose: bool,
    ) -> Result<Self, String> {
        let clock =
            crate::locate::find_clock2(srv, probe, recs, start_offset_ms, bias_max, verbose)?;
        Self::resolve(srv, probe, recs, clock, bounds, verbose)
    }

    /// Resolve and behaviorally validate the one controlled vehicle. Every hop
    /// is an exact pointer/field read. Structural physics checks reject a stale
    /// chain, but never choose between candidates.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        srv: &mut ForkServer,
        probe: usize,
        recs: &[Rec],
        clock: ClockHit,
        bounds: (f64, f64, f64, f64, f64, f64),
        verbose: bool,
    ) -> Result<Self, String> {
        let provenance = resolve_with(
            srv.validator_controller,
            srv.validation_sim,
            BUILD_128182,
            |a, n| procmem::read_at(srv.pid(), a, n),
        )?;
        let hit = qualify2(srv, probe, recs, clock.addr, provenance.state_pos, 150, bounds)
            .ok_or_else(|| {
                format!(
                    "validator-owned CGameVehiclePhy state at {:#x} failed the structural trajectory check",
                    provenance.state_pos
                )
            })?;
        if verbose {
            println!(
                "VALIDATOR CAR controller {:#x} -> sim {:#x} -> playground {:#x} -> player {:#x} -> CGameVehiclePhy {:#x} -> state {:#x}; verr {:.4} m/s, |q|-1 {:.2e}",
                provenance.controller,
                provenance.sim,
                provenance.playground,
                provenance.participant,
                provenance.vehicle,
                provenance.state_pos,
                hit.verr,
                hit.qerr
            );
        }
        Ok(Self {
            layout: Layout {
                pos: provenance.state_pos,
                clock: clock.addr,
                clock_bias: clock.bias,
                rms: hit.verr,
                max_dev: hit.qerr,
            },
            provenance,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn provenance(&self) -> &ValidatorCarProvenance {
        &self.provenance
    }
}

fn word<const N: usize>(
    read: &mut impl FnMut(u64, usize) -> Option<Vec<u8>>,
    at: u64,
) -> Result<[u8; N], String> {
    let b = read(at, N).ok_or_else(|| format!("unreadable validator pointer hop at {:#x}", at))?;
    b.as_slice().try_into().map_err(|_| {
        format!(
            "short validator pointer read at {:#x}: {} of {} bytes",
            at,
            b.len(),
            N
        )
    })
}

fn ptr(
    read: &mut impl FnMut(u64, usize) -> Option<Vec<u8>>,
    at: u64,
    hop: &str,
) -> Result<u64, String> {
    let v = u64::from_le_bytes(word::<8>(read, at)?);
    if v < 0x1000 {
        return Err(format!(
            "validator pointer hop {} at {:#x} is null/invalid ({:#x})",
            hop, at, v
        ));
    }
    Ok(v)
}

fn u32_at(read: &mut impl FnMut(u64, usize) -> Option<Vec<u8>>, at: u64) -> Result<u32, String> {
    Ok(u32::from_le_bytes(word::<4>(read, at)?))
}

fn resolve_with(
    controller: u64,
    captured_sim: u64,
    o: Offsets,
    mut read: impl FnMut(u64, usize) -> Option<Vec<u8>>,
) -> Result<ValidatorCarProvenance, String> {
    if controller < 0x1000 || captured_sim < 0x1000 {
        return Err(
            "validator simulation callback was not captured; refusing heuristic fallback".into(),
        );
    }
    let sim = ptr(&mut read, controller + o.controller_sim, "controller.sim")?;
    if sim != captured_sim {
        return Err(format!(
            "validator callback sim {:#x} disagrees with controller+{:#x} -> {:#x}",
            captured_sim, o.controller_sim, sim
        ));
    }
    let playground = ptr(&mut read, sim + o.sim_playground, "sim.playground")?;
    let n = u32_at(&mut read, playground + o.playground_player_count)?;
    if n != 1 {
        return Err(format!(
            "validator playground has {} players, expected exactly one for a solo /validatepath run",
            n
        ));
    }
    let players = ptr(
        &mut read,
        playground + o.playground_players,
        "playground.players",
    )?;
    let participant = ptr(&mut read, players, "players[0]")?;
    let class = u32_at(&mut read, participant + o.participant_vehicle_class)?;
    if class != CGAME_VEHICLE_PHY {
        return Err(format!(
            "participant primary vehicle class is {:#x}, expected CGameVehiclePhy {:#x}",
            class, CGAME_VEHICLE_PHY
        ));
    }
    let vehicle = ptr(
        &mut read,
        participant + o.participant_vehicle,
        "participant.vehicle",
    )?;
    let state_pos = vehicle + o.vehicle_state_pos;
    let state = word::<40>(&mut read, state_pos - 16)?;
    let f = |i: usize| f32::from_le_bytes(state[i..i + 4].try_into().unwrap());
    if !(0..10).all(|i| f(i * 4).is_finite()) {
        return Err(format!(
            "validator-owned state at {:#x} contains non-finite values",
            state_pos
        ));
    }
    let qn = (f(0).powi(2) + f(4).powi(2) + f(8).powi(2) + f(12).powi(2)).sqrt();
    if (qn - 1.0).abs() > 1e-3 {
        return Err(format!(
            "validator-owned state at {:#x} has quaternion norm {}, expected 1",
            state_pos, qn
        ));
    }
    Ok(ValidatorCarProvenance {
        controller,
        sim,
        playground,
        players,
        participant,
        vehicle,
        state_pos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn fixture() -> (u64, u64, BTreeMap<u64, Vec<u8>>) {
        let (controller, sim, playground, players, participant, vehicle) = (
            0x10000u64, 0x20000u64, 0x30000u64, 0x40000u64, 0x50000u64, 0x60000u64,
        );
        let mut m = BTreeMap::new();
        m.insert(
            controller + BUILD_128182.controller_sim,
            sim.to_le_bytes().to_vec(),
        );
        m.insert(
            sim + BUILD_128182.sim_playground,
            playground.to_le_bytes().to_vec(),
        );
        m.insert(
            playground + BUILD_128182.playground_players,
            players.to_le_bytes().to_vec(),
        );
        m.insert(
            playground + BUILD_128182.playground_player_count,
            1u32.to_le_bytes().to_vec(),
        );
        m.insert(players, participant.to_le_bytes().to_vec());
        m.insert(
            participant + BUILD_128182.participant_vehicle_class,
            CGAME_VEHICLE_PHY.to_le_bytes().to_vec(),
        );
        m.insert(
            participant + BUILD_128182.participant_vehicle,
            vehicle.to_le_bytes().to_vec(),
        );
        let mut state = vec![0u8; 40];
        state[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        m.insert(vehicle + BUILD_128182.vehicle_state_pos - 16, state);
        (controller, sim, m)
    }

    fn resolve_fixture(
        controller: u64,
        sim: u64,
        m: &BTreeMap<u64, Vec<u8>>,
        o: Offsets,
    ) -> Result<ValidatorCarProvenance, String> {
        resolve_with(controller, sim, o, |a, n| {
            m.get(&a).filter(|b| b.len() == n).cloned()
        })
    }

    #[test]
    fn follows_the_validator_owned_chain_without_searching() {
        let (controller, sim, m) = fixture();
        let p = resolve_fixture(controller, sim, &m, BUILD_128182).unwrap();
        assert_eq!(p.participant, 0x50000);
        assert_eq!(p.vehicle, 0x60000);
        assert_eq!(p.state_pos, 0x612f0);
    }

    #[test]
    fn a_perturbed_hop_fails_loudly_instead_of_finding_another_object() {
        let (controller, sim, m) = fixture();
        let mut broken = BUILD_128182;
        broken.participant_vehicle += 8;
        let e = resolve_fixture(controller, sim, &m, broken).unwrap_err();
        assert!(e.contains("unreadable validator pointer hop"), "{e}");
    }

    #[test]
    fn callback_and_controller_must_name_the_same_simulation() {
        let (controller, sim, m) = fixture();
        let e = resolve_fixture(controller, sim + 8, &m, BUILD_128182).unwrap_err();
        assert!(e.contains("disagrees"), "{e}");
    }

    #[test]
    fn a_non_vehicle_primary_slot_is_rejected() {
        let (controller, sim, mut m) = fixture();
        m.insert(
            0x50000 + BUILD_128182.participant_vehicle_class,
            0x0a02_0000u32.to_le_bytes().to_vec(),
        );
        let e = resolve_fixture(controller, sim, &m, BUILD_128182).unwrap_err();
        assert!(e.contains("CGameVehiclePhy"), "{e}");
    }
}
