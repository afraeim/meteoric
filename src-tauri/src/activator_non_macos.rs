//! Non-macOS global activation listener.
//!
//! Detects a double-tap on Shift (left or right) and invokes the provided
//! callback. Intended for Linux/Windows where the macOS CGEventTap activator
//! is unavailable.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rdev::{listen, Event, EventType, Key};

/// Maximum allowed gap between Shift taps.
const ACTIVATION_WINDOW: Duration = Duration::from_millis(400);
/// Debounce interval between successful activations.
const ACTIVATION_COOLDOWN: Duration = Duration::from_millis(600);

struct ActivationState {
    last_shift_tap: Option<Instant>,
    left_down: bool,
    right_down: bool,
    last_activation: Option<Instant>,
}

fn update_key_state(state: &mut ActivationState, key: Key, is_press: bool) {
    match key {
        Key::ShiftLeft => state.left_down = is_press,
        Key::ShiftRight => state.right_down = is_press,
        _ => {}
    }
}

fn is_already_down(state: &ActivationState, key: Key) -> bool {
    match key {
        Key::ShiftLeft => state.left_down,
        Key::ShiftRight => state.right_down,
        _ => false,
    }
}

fn evaluate_double_shift(state: &mut ActivationState, key: Key, is_press: bool) -> bool {
    if !matches!(key, Key::ShiftLeft | Key::ShiftRight) {
        return false;
    }

    if is_press {
        // Ignore auto-repeat while key is physically held.
        if is_already_down(state, key) {
            return false;
        }
        update_key_state(state, key, true);

        let now = Instant::now();
        if let Some(last_act) = state.last_activation {
            if now.duration_since(last_act) < ACTIVATION_COOLDOWN {
                return false;
            }
        }

        if let Some(last_tap) = state.last_shift_tap {
            if now.duration_since(last_tap) < ACTIVATION_WINDOW {
                state.last_shift_tap = None;
                state.last_activation = Some(now);
                return true;
            }
        }

        state.last_shift_tap = Some(now);
        false
    } else {
        update_key_state(state, key, false);
        false
    }
}

/// Lifecycle wrapper for the background key listener.
pub struct OverlayActivator {
    is_active: Arc<AtomicBool>,
}

impl OverlayActivator {
    pub fn new() -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start<F>(&self, on_activation: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        if self.is_active.load(Ordering::SeqCst) {
            return;
        }
        self.is_active.store(true, Ordering::SeqCst);

        let is_active = self.is_active.clone();
        let on_activation = Arc::new(on_activation);
        let state = Arc::new(Mutex::new(ActivationState {
            last_shift_tap: None,
            left_down: false,
            right_down: false,
            last_activation: None,
        }));

        std::thread::spawn(move || {
            let callback = move |event: Event| {
                if !is_active.load(Ordering::SeqCst) {
                    return;
                }

                let (key, is_press) = match event.event_type {
                    EventType::KeyPress(k) => (k, true),
                    EventType::KeyRelease(k) => (k, false),
                    _ => return,
                };

                if !matches!(key, Key::ShiftLeft | Key::ShiftRight) {
                    return;
                }

                let mut s = state.lock().unwrap();
                if evaluate_double_shift(&mut s, key, is_press) {
                    on_activation();
                }
            };

            if let Err(err) = listen(callback) {
                eprintln!("meteoric: [activator] non-macOS listener failed: {err:?}");
            }
        });
    }
}
