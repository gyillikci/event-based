/*
 * Active LED Markers — Arduino Nano 33 BLE (nRF52840)
 * ----------------------------------------------------
 * Blinks the board's built-in LEDs, each at a distinct, fixed frequency, so an
 * event-based camera can detect and uniquely identify every LED purely from its
 * blink frequency (see detect_active_markers.py on the PC side).
 *
 * This is the standalone MVP: no serial/BLE commands are needed. The Arduino
 * just blinks the markers at the frequencies below; the PC identifies them.
 *
 * Built-in LEDs on the Nano 33 BLE / BLE Sense:
 *   LED_BUILTIN (D13)  orange user LED   — ACTIVE HIGH
 *   LEDR / LEDG / LEDB onboard RGB LED    — ACTIVE LOW (lit when pin is LOW)
 *
 * Frequencies MUST match active_markers.json used by the PC detector.
 *   BUILTIN -> 800 Hz
 *   RED     -> 1100 Hz
 *   GREEN   -> 1500 Hz
 *   BLUE    -> 2000 Hz
 *
 * IMPORTANT — the RGB LED is ONE physical package (LEDR/LEDG/LEDB are three
 * dies at the same spot). Driving more than one channel makes their blink
 * frequencies superimpose on the same pixels, which the per-pixel frequency
 * detector cannot separate. So this build drives only TWO spatially distinct
 * sources: the orange user LED (BUILTIN, 800 Hz) and ONE RGB channel (GREEN,
 * 1500 Hz). RED and BLUE are left off; enable them only if you split the RGB
 * channels onto separate, physically-spaced LEDs. (The green power LED next to
 * the user LED is always on, so it never flickers and is not a marker.)
 *
 * Timing: HARDWARE PWM. Each LED is driven by one of the nRF52840's four PWM
 * peripherals (via the mbed `PwmOut` API) at a fixed 50 % duty square wave. The
 * PWM hardware runs autonomously, so the output is exact and jitter-free at
 * every frequency and loop() does nothing. This replaces the earlier software
 * micros() toggling, which was fine at ~800-1100 Hz but became too jittery
 * above ~1.5 kHz: the slow mbed digitalWrite() smeared the shortest periods, so
 * the event camera could not lock onto the 1500 / 2000 Hz markers (and read the
 * ones it did see several percent high).
 *
 * To add soldered LEDs later, wire them to a free digital pin and add a row to
 * the markers[] table below (activeLow=false for a normal LED to GND). Any
 * Nano 33 BLE GPIO can be routed to a PWM channel; up to four LEDs are driven
 * with zero CPU cost. A fifth marker would need a software fallback.
 */

#include "mbed.h"

struct Marker {
  int pin;        // Arduino pin
  float freqHz;   // blink frequency (Hz)
  bool activeLow; // true if the LED is lit when the pin is LOW
};

// Edit this table to change frequencies or add soldered LEDs. Keep the values
// in sync with active_markers.json on the PC side.
//
// Only ONE RGB channel is driven (GREEN): the RGB dies share one spot, so
// driving several at once superimposes their frequencies on the same pixels.
// To add more markers, wire physically-separated LEDs to free pins and add
// rows here (activeLow=false for a normal LED to GND).
Marker markers[] = {
  //  pin          freqHz   activeLow
  { LED_BUILTIN,   800.0f,  false },  // orange user LED — its own location
  { LEDG,         1500.0f,  true  },  // RGB GREEN channel only (single spot)
  // { LEDR,      1100.0f,  true  }, // disabled: shares the RGB spot
  // { LEDB,      2000.0f,  true  }, // disabled: shares the RGB spot
};

const int NUM_MARKERS = sizeof(markers) / sizeof(markers[0]);

// One hardware PWM channel per marker (the nRF52840 has four PWM peripherals).
mbed::PwmOut *pwm[NUM_MARKERS];

void setup() {
  for (int i = 0; i < NUM_MARKERS; i++) {
    // Route the Arduino pin to a PWM peripheral.
    pwm[i] = new mbed::PwmOut(digitalPinToPinName(markers[i].pin));

    // period [us] = 1e6 / freq, rounded to the nearest microsecond.
    int periodUs = (int)(1000000.0f / markers[i].freqHz + 0.5f);
    pwm[i]->period_us(periodUs);

    // 50 % duty. Polarity only shifts the phase, so the detected frequency is
    // identical for active-high and active-low LEDs; 0.5 keeps each LED lit
    // half the time.
    pwm[i]->write(0.5f);
  }
}

void loop() {
  // Nothing to do: the PWM hardware generates all four square waves on its own.
}
