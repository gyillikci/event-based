/*
 * Interactive LED frequency sweep — Arduino Nano 33 BLE (nRF52840)
 * ----------------------------------------------------------------
 * Drives the built-in user LED at a frequency the PC sets over serial, so the
 * event camera side can dial a frequency and watch, live, whether the chain
 * still resolves the blink. See freq_sweep_live.py, which drives this and does
 * the measuring.
 *
 * The question this answers: the nRF52840's hardware PWM reaches megahertz, and
 * the IMX636 pixel's ~6 us refractory implies a ceiling near 160 kHz, but the
 * pixel also needs enough photocurrent to slew that fast, and the SDK's period
 * matching needs its tolerance to scale with 1/f. Only a measurement says which
 * of those binds first for THIS LED, lens and lighting.
 *
 * The RGB GREEN channel holds a fixed reference frequency throughout. It is the
 * control: if the swept LED washes out at 20 kHz while the reference is still
 * being measured cleanly, the loss is a frequency effect and not the board
 * drifting out of frame or the biases being wrong.
 *
 * SERIAL PROTOCOL, 115200 baud, one newline-terminated command in, one line out:
 *
 *   f <hz>   set the swept LED frequency; 0 turns it off
 *   r <hz>   set the reference LED frequency; 0 turns it off
 *   sweep    run the built-in ladder once, unattended
 *   stop     leave sweep mode, hold the current frequency
 *   ?        report state
 *
 * Every reply starts OK or ERR and reports the ACTUAL emitted frequency, which
 * is the number that matters: mbed's PwmOut::period_us() takes whole
 * microseconds, so only frequencies of the form 1e6/N can be produced. That is
 * invisible low down (1500 Hz is off by 0.05%) and severe high up - near
 * 100 kHz the reachable values are 100 kHz and 111 kHz with nothing between.
 * The PC compares against the actual, never the requested.
 */

#include "mbed.h"

// Rungs for the unattended ladder. Every value divides 1000000 exactly, so each
// is emitted at precisely its nominal frequency.
const unsigned long SWEEP_HZ[] = {500, 1000, 2000, 5000, 10000, 20000, 50000, 100000};
const int NUM_STEPS = sizeof(SWEEP_HZ) / sizeof(SWEEP_HZ[0]);

const unsigned long DWELL_MS = 3000;   // hold per rung in sweep mode
const unsigned long GAP_MS   = 600;    // blackout between rungs, so bursts are separable

const unsigned long REFERENCE_PERIOD_US = 1429;   // 1e6/1429 = 699.8 Hz
const unsigned long DEFAULT_SWEEP_HZ    = 1000;

mbed::PwmOut *sweep_led;      // LED_BUILTIN (D13), active HIGH, its own location
mbed::PwmOut *reference_led;  // LEDG, active LOW, inside the shared RGB package

unsigned long sweep_hz = DEFAULT_SWEEP_HZ;
unsigned long ref_hz   = 0;
bool sweeping          = false;
int sweep_index        = 0;
unsigned long step_started_ms = 0;
bool in_gap            = false;

// Returns the frequency actually emitted, which is 1e6/round(1e6/hz).
float set_pwm(mbed::PwmOut *pwm, unsigned long hz) {
  if (hz == 0) {
    pwm->write(0.0f);            // parks the pin low
    return 0.0f;
  }
  unsigned long period_us = 1000000UL / hz;
  if (period_us < 1) period_us = 1;
  pwm->period_us(period_us);
  // 50% duty. Polarity only shifts phase, so an active-low LED blinks at the
  // same frequency as an active-high one.
  pwm->write(0.5f);
  return 1000000.0f / period_us;
}

void report(const char *tag) {
  Serial.print(tag);
  Serial.print(" sweep_hz=");
  Serial.print(sweep_hz);
  Serial.print(" sweep_actual=");
  Serial.print(sweep_hz ? 1000000.0f / (1000000UL / sweep_hz) : 0.0f, 2);
  Serial.print(" ref_hz=");
  Serial.print(ref_hz);
  Serial.print(" ref_actual=");
  Serial.print(ref_hz ? 1000000.0f / (1000000UL / ref_hz) : 0.0f, 2);
  Serial.print(" sweeping=");
  Serial.println(sweeping ? 1 : 0);
}

void setup() {
  Serial.begin(115200);

  sweep_led     = new mbed::PwmOut(digitalPinToPinName(LED_BUILTIN));
  reference_led = new mbed::PwmOut(digitalPinToPinName(LEDG));

  ref_hz = 1000000UL / REFERENCE_PERIOD_US;
  set_pwm(reference_led, ref_hz);
  set_pwm(sweep_led, sweep_hz);

  report("OK ready");
}

void handle_command(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "?") {
    report("OK state");
  } else if (line == "sweep") {
    sweeping = true;
    sweep_index = 0;
    in_gap = true;
    step_started_ms = millis();
    set_pwm(sweep_led, 0);
    report("OK sweep");
  } else if (line == "stop") {
    sweeping = false;
    set_pwm(sweep_led, sweep_hz);
    report("OK stop");
  } else if (line.startsWith("f ") || line.startsWith("r ")) {
    const bool is_sweep = line.startsWith("f ");
    const long hz = line.substring(2).toInt();
    if (hz < 0 || hz > 1000000L) {
      Serial.println("ERR frequency out of range 0..1000000");
      return;
    }
    if (is_sweep) {
      sweeping = false;                 // a manual set takes over from the ladder
      sweep_hz = (unsigned long)hz;
      set_pwm(sweep_led, sweep_hz);
    } else {
      ref_hz = (unsigned long)hz;
      set_pwm(reference_led, ref_hz);
    }
    report("OK set");
  } else {
    Serial.println("ERR unknown command; use: f <hz> | r <hz> | sweep | stop | ?");
  }
}

void loop() {
  while (Serial.available()) {
    handle_command(Serial.readStringUntil('\n'));
  }

  if (!sweeping) return;

  const unsigned long now = millis();
  if (in_gap) {
    if (now - step_started_ms >= GAP_MS) {
      sweep_hz = SWEEP_HZ[sweep_index];
      set_pwm(sweep_led, sweep_hz);
      in_gap = false;
      step_started_ms = now;
      report("OK step");
    }
  } else if (now - step_started_ms >= DWELL_MS) {
    sweep_index++;
    if (sweep_index >= NUM_STEPS) {
      sweeping = false;
      report("OK sweep_done");
      return;
    }
    set_pwm(sweep_led, 0);              // blackout before the next rung
    in_gap = true;
    step_started_ms = now;
  }
}
