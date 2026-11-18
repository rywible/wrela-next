//! A person's input in a window: the keyboard, the pointer and the wheel from the window's
//! events (winit), and gamepads (gilrs), as the program's events ([`wrela_abi::input`]): keys
//! by their DOM code, pointer positions in the frame's pixels, pads in the W3C standard
//! mapping. A frame takes every event that came before it ([`Live::take`]).
//!
//! What the ABI promises, kept here: every `KeyUp` follows its `KeyDown`, and a window that
//! loses focus lets go of the keys and buttons held in it (no platform tells a window of a key
//! let go elsewhere, and a held key would stay held in the program); a pad's values are its
//! own, with no dead zone (gilrs's default filters, a dead zone and a jitter filter, are off;
//! only its D-pad-as-axis filter is kept, which makes a hat's D-pad the standard buttons); a
//! button sends one event per change the pad made, its value and whether it's pressed together;
//! a pad that leaves lets go of what it held.

use std::collections::HashMap;
use std::time::SystemTime;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use wrela_abi::input::{ALT, CONTROL, Event, EventKind, META, PADS, SHIFT, key_code};

/// The keys the ABI names, by winit's code: winit names its codes after the W3C UI Events
/// codes, as the ABI does (`KeyA`, `ArrowLeft`), except the OS key, which it calls `Super`.
pub const KEYS: [(KeyCode, &str); 89] = {
    use KeyCode::*;
    [
        (KeyA, "KeyA"),
        (KeyB, "KeyB"),
        (KeyC, "KeyC"),
        (KeyD, "KeyD"),
        (KeyE, "KeyE"),
        (KeyF, "KeyF"),
        (KeyG, "KeyG"),
        (KeyH, "KeyH"),
        (KeyI, "KeyI"),
        (KeyJ, "KeyJ"),
        (KeyK, "KeyK"),
        (KeyL, "KeyL"),
        (KeyM, "KeyM"),
        (KeyN, "KeyN"),
        (KeyO, "KeyO"),
        (KeyP, "KeyP"),
        (KeyQ, "KeyQ"),
        (KeyR, "KeyR"),
        (KeyS, "KeyS"),
        (KeyT, "KeyT"),
        (KeyU, "KeyU"),
        (KeyV, "KeyV"),
        (KeyW, "KeyW"),
        (KeyX, "KeyX"),
        (KeyY, "KeyY"),
        (KeyZ, "KeyZ"),
        (Digit0, "Digit0"),
        (Digit1, "Digit1"),
        (Digit2, "Digit2"),
        (Digit3, "Digit3"),
        (Digit4, "Digit4"),
        (Digit5, "Digit5"),
        (Digit6, "Digit6"),
        (Digit7, "Digit7"),
        (Digit8, "Digit8"),
        (Digit9, "Digit9"),
        (Enter, "Enter"),
        (Escape, "Escape"),
        (Backspace, "Backspace"),
        (Tab, "Tab"),
        (Space, "Space"),
        (ArrowLeft, "ArrowLeft"),
        (ArrowRight, "ArrowRight"),
        (ArrowUp, "ArrowUp"),
        (ArrowDown, "ArrowDown"),
        (Home, "Home"),
        (End, "End"),
        (PageUp, "PageUp"),
        (PageDown, "PageDown"),
        (Insert, "Insert"),
        (Delete, "Delete"),
        (ShiftLeft, "ShiftLeft"),
        (ShiftRight, "ShiftRight"),
        (ControlLeft, "ControlLeft"),
        (ControlRight, "ControlRight"),
        (AltLeft, "AltLeft"),
        (AltRight, "AltRight"),
        (SuperLeft, "MetaLeft"),
        (SuperRight, "MetaRight"),
        (CapsLock, "CapsLock"),
        (Minus, "Minus"),
        (Equal, "Equal"),
        (BracketLeft, "BracketLeft"),
        (BracketRight, "BracketRight"),
        (Backslash, "Backslash"),
        (Semicolon, "Semicolon"),
        (Quote, "Quote"),
        (Backquote, "Backquote"),
        (Comma, "Comma"),
        (Period, "Period"),
        (Slash, "Slash"),
        (F1, "F1"),
        (F2, "F2"),
        (F3, "F3"),
        (F4, "F4"),
        (F5, "F5"),
        (F6, "F6"),
        (F7, "F7"),
        (F8, "F8"),
        (F9, "F9"),
        (F10, "F10"),
        (F11, "F11"),
        (F12, "F12"),
        (NumpadEnter, "NumpadEnter"),
        (NumpadAdd, "NumpadAdd"),
        (NumpadSubtract, "NumpadSubtract"),
        (NumpadMultiply, "NumpadMultiply"),
        (NumpadDivide, "NumpadDivide"),
        (NumpadDecimal, "NumpadDecimal"),
    ]
};

/// A key's number ([`wrela_abi::input::KEYS`]); 0 for one the ABI doesn't name.
pub fn key(code: KeyCode) -> u32 {
    KEYS.iter().find(|(k, _)| *k == code).map_or(0, |(_, name)| key_code(name))
}

/// A pad's buttons in the standard mapping's order ([`wrela_abi::input::PAD_BUTTONS`]): gilrs
/// calls the shoulders triggers and the triggers `…Trigger2`.
pub const PAD_BUTTONS: [gilrs::Button; 17] = {
    use gilrs::Button::*;
    [
        South,
        East,
        West,
        North,
        LeftTrigger,
        RightTrigger,
        LeftTrigger2,
        RightTrigger2,
        Select,
        Start,
        LeftThumb,
        RightThumb,
        DPadUp,
        DPadDown,
        DPadLeft,
        DPadRight,
        Mode,
    ]
};

/// A pad's axes in the standard mapping's order ([`wrela_abi::input::PAD_AXES`]).
pub const PAD_AXES: [gilrs::Axis; 4] = {
    use gilrs::Axis::*;
    [LeftStickX, LeftStickY, RightStickX, RightStickY]
};

/// What's come from a person's devices since the last frame took it.
pub struct Live {
    pending: Vec<Event>,
    modifiers: u32,
    pointer: (f32, f32),
    /// Keys and pointer buttons held, in the order pressed: let go when focus goes.
    keys: Vec<u32>,
    buttons: Vec<u32>,
    /// The gamepads (none where gilrs can't start).
    gilrs: Option<gilrs::Gilrs>,
    pads: Pads<gilrs::GamepadId>,
}

impl Default for Live {
    fn default() -> Live {
        Live::new()
    }
}

impl Live {
    /// A person's input, with the machine's gamepads.
    pub fn new() -> Live {
        let gilrs = gilrs::GilrsBuilder::new().with_default_filters(false).build().ok();
        Live { gilrs, ..Live::without_pads() }
    }

    /// The keyboard and pointer alone (a test's).
    fn without_pads() -> Live {
        Live {
            pending: Vec::new(),
            modifiers: 0,
            pointer: (0.0, 0.0),
            keys: Vec::new(),
            buttons: Vec::new(),
            gilrs: None,
            pads: Pads::new(),
        }
    }

    /// The events since the last take, oldest first: the next frame's. The pads are looked at
    /// now, so their changes are as late as the frame.
    pub fn take(&mut self) -> Vec<Event> {
        self.poll_pads();
        self.pads.taken();
        std::mem::take(&mut self.pending)
    }

    /// An event from elsewhere that a person's would be (a test's, arriving as theirs do).
    pub fn push(&mut self, event: Event) {
        self.pending.push(event);
    }

    /// A window event, as the program's events.
    pub fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::ModifiersChanged(mods) => {
                let s = mods.state();
                let on = [
                    (s.shift_key(), SHIFT),
                    (s.control_key(), CONTROL),
                    (s.alt_key(), ALT),
                    (s.super_key(), META),
                ];
                self.modifiers = on.iter().filter(|(on, _)| *on).fold(0, |m, (_, bit)| m | bit);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.moved(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => 0,
                    MouseButton::Middle => 1,
                    MouseButton::Right => 2,
                    _ => return,
                };
                self.button(button, *state == ElementState::Pressed);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Right and down positive, in pixels (a line is 40, as a browser takes one);
                // winit's are the other way.
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (-x * 40.0, -y * 40.0),
                    MouseScrollDelta::PixelDelta(p) => (-p.x as f32, -p.y as f32),
                };
                self.wheel(dx, dy);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let down = event.state == ElementState::Pressed;
                self.key(key(code), down, event.repeat, event.text.as_deref());
            }
            WindowEvent::Focused(false) => self.unfocused(),
            _ => {}
        }
    }

    fn moved(&mut self, x: f32, y: f32) {
        self.pointer = (x, y);
        self.pending.push(Event::pointer(EventKind::PointerMove, x, y, 0, self.modifiers));
    }

    fn button(&mut self, button: u32, down: bool) {
        let held = self.buttons.iter().position(|&b| b == button);
        let kind = match (down, held) {
            (true, None) => {
                self.buttons.push(button);
                EventKind::PointerDown
            }
            (true, Some(_)) => return,
            (false, Some(i)) => {
                self.buttons.remove(i);
                EventKind::PointerUp
            }
            // Let go of a button pressed elsewhere, or already let go when focus went.
            (false, None) => return,
        };
        let (x, y) = self.pointer;
        self.pending.push(Event::pointer(kind, x, y, button, self.modifiers));
    }

    fn wheel(&mut self, dx: f32, dy: f32) {
        let (x, y) = self.pointer;
        self.pending.push(Event::wheel(x, y, dx, dy, self.modifiers));
    }

    /// Key `k` (0: one the ABI doesn't name, which sends no key event) went down or up, and
    /// what it typed.
    fn key(&mut self, k: u32, down: bool, repeat: bool, text: Option<&str>) {
        if k != 0 {
            let held = self.keys.iter().position(|&x| x == k);
            match (down, held) {
                (true, None) => self.keys.push(k),
                (true, Some(_)) => {}
                (false, Some(i)) => {
                    self.keys.remove(i);
                }
                // Let go of a key pressed elsewhere, or already let go when focus went.
                (false, None) => return,
            }
            self.pending.push(Event::key(down, k, repeat, self.modifiers));
        }
        // What it typed, after the key: printing characters only (Enter, Tab and Backspace
        // are keys).
        if down && let Some(text) = text {
            for c in text.chars().filter(|c| !c.is_control()) {
                self.pending.push(Event::text(c));
            }
        }
    }

    /// The window lost focus: the keys and buttons held in it are let go, in the order pressed.
    fn unfocused(&mut self) {
        for k in std::mem::take(&mut self.keys) {
            self.pending.push(Event::key(false, k, false, self.modifiers));
        }
        let (x, y) = self.pointer;
        for b in std::mem::take(&mut self.buttons) {
            self.pending.push(Event::pointer(EventKind::PointerUp, x, y, b, self.modifiers));
        }
    }

    /// The gamepads' changes since the last look.
    fn poll_pads(&mut self) {
        use gilrs::ev::filter::{Filter, axis_dpad_to_button};
        let Some(gilrs) = &mut self.gilrs else { return };
        while let Some(event) = gilrs.next_event().filter_ev(&axis_dpad_to_button, gilrs) {
            if event.is_dropped() {
                continue;
            }
            // gilrs keeps each pad's state from the events `next_event` returns, before this
            // filter: the button press it makes of a hat must be kept too, or the hat's release
            // finds no button pressed and is dropped (seen with a virtual pad's D-pad). Keeping
            // an event it kept already changes nothing.
            gilrs.update(&event);
            if let Some(change) = PadChange::of(event.event) {
                self.pads.change(event.id, change, event.time, &mut self.pending);
            }
        }
    }
}

/// A pad's change, by the standard mapping's numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadChange {
    /// It's here (it takes a pad's number, if one is free).
    Here,
    Pressed(u32),
    Released(u32),
    /// A button's value.
    Value(u32, f32),
    Axis(u32, f32),
    /// It left.
    Gone,
}

impl PadChange {
    /// A gilrs event's change; `None` for one the standard mapping doesn't name.
    pub fn of(event: gilrs::EventType) -> Option<PadChange> {
        use gilrs::EventType::*;
        let button = |b| PAD_BUTTONS.iter().position(|x| *x == b).map(|i| i as u32);
        Some(match event {
            Connected => PadChange::Here,
            Disconnected => PadChange::Gone,
            ButtonPressed(b, _) => PadChange::Pressed(button(b)?),
            ButtonReleased(b, _) => PadChange::Released(button(b)?),
            ButtonChanged(b, v, _) => PadChange::Value(button(b)?, v),
            AxisChanged(a, v, _) => PadChange::axis(a, v)?,
            _ => return None,
        })
    }

    /// Axis `a` at `v`, as gilrs has it, in the standard mapping.
    fn axis(a: gilrs::Axis, v: f32) -> Option<PadChange> {
        let i = PAD_AXES.iter().position(|x| *x == a)?;
        // gilrs has a stick up positive; the standard mapping, down.
        let v = if i % 2 == 1 && v != 0.0 { -v } else { v };
        Some(PadChange::Axis(i as u32, v))
    }
}

/// The pads by their ids (`Id`: gilrs's), and what the program has been sent of each one's
/// buttons and axes.
struct Pads<Id> {
    ids: [Option<Id>; PADS as usize],
    /// Each button's value and whether it's pressed, and each axis's value, as last sent, by
    /// (pad, is a button, index).
    sent: HashMap<(u32, bool, u32), (f32, bool)>,
    /// Where each button's last event this frame is in the pending events, and the time of the
    /// pad's change it came from: gilrs splits one change into a press and a value, which go
    /// out as one event.
    merge: HashMap<(u32, u32), (usize, SystemTime)>,
}

impl<Id: Copy + PartialEq> Pads<Id> {
    fn new() -> Pads<Id> {
        Pads { ids: [None; PADS as usize], sent: HashMap::new(), merge: HashMap::new() }
    }

    /// The pending events were taken.
    fn taken(&mut self) {
        self.merge.clear();
    }

    /// Pad `id` changed (at `time`, the pad's): its events onto `out`.
    fn change(&mut self, id: Id, change: PadChange, time: SystemTime, out: &mut Vec<Event>) {
        let known = self.ids.iter().position(|p| *p == Some(id));
        if change == PadChange::Gone {
            let Some(p) = known else { return };
            self.ids[p] = None;
            let pad = p as u32;
            let mut held: Vec<_> =
                self.sent.iter().filter(|(k, _)| k.0 == pad).map(|(k, v)| (*k, *v)).collect();
            held.sort_by_key(|(k, _)| (!k.1, k.2));
            for ((_, button, i), (value, pressed)) in held {
                self.sent.remove(&(pad, button, i));
                self.merge.remove(&(pad, i));
                if value != 0.0 || pressed {
                    out.push(match button {
                        true => Event::pad_button(pad, i, 0.0, false),
                        false => Event::pad_axis(pad, i, 0.0),
                    });
                }
            }
            return;
        }
        // A pad's number: the first free when it's first heard from.
        let p = match known {
            Some(p) => p,
            None => match self.ids.iter().position(Option::is_none) {
                Some(p) => {
                    self.ids[p] = Some(id);
                    p
                }
                None => return,
            },
        };
        let pad = p as u32;
        let (button, i, value, pressed) = match change {
            PadChange::Here | PadChange::Gone => return,
            PadChange::Axis(i, v) => (false, i, v, false),
            PadChange::Pressed(i) | PadChange::Released(i) | PadChange::Value(i, _) => {
                let (v, pressed) = self.sent.get(&(pad, true, i)).copied().unwrap_or_default();
                match change {
                    PadChange::Pressed(_) => (true, i, v, true),
                    PadChange::Released(_) => (true, i, v, false),
                    PadChange::Value(_, v) => (true, i, v, pressed),
                    _ => unreachable!(),
                }
            }
        };
        let last = self.sent.insert((pad, button, i), (value, pressed));
        if last.unwrap_or_default() == (value, pressed) {
            return;
        }
        if !button {
            out.push(Event::pad_axis(pad, i, value));
            return;
        }
        let event = Event::pad_button(pad, i, value, pressed);
        match self.merge.get(&(pad, i)) {
            Some(&(at, t)) if t == time && at < out.len() => out[at] = event,
            _ => {
                self.merge.insert((pad, i), (out.len(), time));
                out.push(event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use wrela_abi::input::{PAD_AXES as AXES, PAD_BUTTONS as BUTTONS};

    #[test]
    fn every_key_the_abi_names_has_one_winit_code_and_no_more() {
        let names: Vec<&str> = KEYS.iter().map(|(_, n)| *n).collect();
        let abi: Vec<&str> = wrela_abi::input::KEYS[1..].to_vec();
        assert_eq!(names, abi, "in the ABI's order, each once");
        for (code, name) in KEYS {
            assert_eq!(key(code), key_code(name));
            assert_ne!(key(code), 0);
        }
        assert_eq!(key(KeyCode::IntlBackslash), 0, "a key the ABI doesn't name is 0");
    }

    #[test]
    fn every_pad_button_and_axis_has_its_standard_number() {
        assert_eq!(PAD_BUTTONS.len(), BUTTONS.len());
        assert_eq!(PAD_AXES.len(), AXES.len());
        for (i, b) in PAD_BUTTONS.iter().enumerate() {
            assert_eq!(PAD_BUTTONS.iter().position(|x| x == b), Some(i), "{b:?} once");
        }
        let at = |name| BUTTONS.iter().position(|n| *n == name);
        assert_eq!(PAD_BUTTONS[at("south").unwrap()], gilrs::Button::South);
        assert_eq!(PAD_BUTTONS[at("leftshoulder").unwrap()], gilrs::Button::LeftTrigger);
        assert_eq!(PAD_BUTTONS[at("righttrigger").unwrap()], gilrs::Button::RightTrigger2);
        assert_eq!(PAD_BUTTONS[at("home").unwrap()], gilrs::Button::Mode);
        let lefty = AXES.iter().position(|n| *n == "lefty").unwrap() as u32;
        let up = PadChange::axis(gilrs::Axis::LeftStickY, 1.0);
        assert_eq!(up, Some(PadChange::Axis(lefty, -1.0)), "up is negative");
        assert_eq!(PadChange::axis(gilrs::Axis::LeftStickX, 0.5), Some(PadChange::Axis(0, 0.5)));
        assert_eq!(PadChange::axis(gilrs::Axis::LeftZ, 0.5), None, "not in the mapping");
    }

    #[test]
    fn keys_go_up_once_after_they_went_down_and_type_after_the_key() {
        let mut live = Live::without_pads();
        let (w, space) = (key_code("KeyW"), key_code("Space"));
        live.key(w, false, false, None); // pressed before the window had focus: no up
        live.modifiers = SHIFT;
        live.key(w, true, false, Some("W"));
        live.key(w, true, true, Some("W"));
        live.key(0, true, false, Some("é")); // a key the ABI doesn't name still types
        live.key(space, true, false, Some(" "));
        live.key(key_code("Enter"), true, false, Some("\r"));
        live.key(w, false, false, None);
        live.key(w, false, false, None);
        assert_eq!(
            live.take(),
            vec![
                Event::key(true, w, false, SHIFT),
                Event::text('W'),
                Event::key(true, w, true, SHIFT),
                Event::text('W'),
                Event::text('é'),
                Event::key(true, space, false, SHIFT),
                Event::text(' '),
                Event::key(true, key_code("Enter"), false, SHIFT),
                Event::key(false, w, false, SHIFT),
            ]
        );
        // Focus goes with Space and Enter held, and the primary button: each let go, once.
        live.moved(5.0, 6.0);
        live.button(0, true);
        live.button(0, true);
        live.unfocused();
        live.key(space, false, false, None);
        live.button(0, false);
        live.unfocused();
        assert_eq!(
            live.take(),
            vec![
                Event::pointer(EventKind::PointerMove, 5.0, 6.0, 0, SHIFT),
                Event::pointer(EventKind::PointerDown, 5.0, 6.0, 0, SHIFT),
                Event::key(false, space, false, SHIFT),
                Event::key(false, key_code("Enter"), false, SHIFT),
                Event::pointer(EventKind::PointerUp, 5.0, 6.0, 0, SHIFT),
            ]
        );
    }

    /// gilrs's events for a pad, as `Live::poll_pads` gives them, and what the program gets.
    #[test]
    fn a_pads_change_is_one_event_and_a_pad_that_leaves_lets_go() {
        let mut pads: Pads<u32> = Pads::new();
        let mut out = Vec::new();
        let t = |ms| SystemTime::UNIX_EPOCH + Duration::from_millis(ms);
        let (a, south, rt) = (77, 0, 7);
        use PadChange::*;
        // A digital press: a press and then its value, from one change of the pad.
        pads.change(a, Pressed(south), t(1), &mut out);
        pads.change(a, Value(south, 1.0), t(1), &mut out);
        // A trigger pulled: values, then past the threshold a press with its value.
        pads.change(a, Value(rt, 0.25), t(2), &mut out);
        pads.change(a, Value(rt, 0.25), t(3), &mut out); // no change: nothing
        pads.change(a, Pressed(rt), t(4), &mut out);
        pads.change(a, Value(rt, 0.8), t(4), &mut out);
        // A stick, a little off centre: no dead zone.
        pads.change(a, Axis(0, 0.01), t(5), &mut out);
        // A tap within one frame: a press and its release, each its own change.
        pads.change(a, Released(south), t(6), &mut out);
        pads.change(a, Value(south, 0.0), t(6), &mut out);
        pads.change(a, Pressed(south), t(7), &mut out);
        pads.change(a, Value(south, 1.0), t(7), &mut out);
        assert_eq!(
            out,
            vec![
                Event::pad_button(0, south, 1.0, true),
                Event::pad_button(0, rt, 0.25, false),
                Event::pad_button(0, rt, 0.8, true),
                Event::pad_axis(0, 0, 0.01),
                Event::pad_button(0, south, 0.0, false),
                Event::pad_button(0, south, 1.0, true),
            ]
        );
        out.clear();
        pads.taken();
        // A second pad takes the next number; the first leaves holding a button, a trigger
        // and a stick off centre, and lets go of them, buttons first; then the second pad's
        // events go on, and a pad that comes back takes the first free number.
        pads.change(9, Here, t(8), &mut out);
        pads.change(a, Gone, t(9), &mut out);
        pads.change(9, Axis(1, -0.5), t(10), &mut out);
        pads.change(a, Here, t(11), &mut out);
        pads.change(a, Pressed(south), t(12), &mut out);
        pads.change(a, Value(south, 1.0), t(12), &mut out);
        assert_eq!(
            out,
            vec![
                Event::pad_button(0, south, 0.0, false),
                Event::pad_button(0, rt, 0.0, false),
                Event::pad_axis(0, 0, 0.0),
                Event::pad_axis(1, 1, -0.5),
                Event::pad_button(0, south, 1.0, true),
            ]
        );
    }
}
