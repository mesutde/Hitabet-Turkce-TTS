// Release'de konsol penceresi acilmaz (GUI uygulamasi).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ema_lightning_tts_lib::run();
}
