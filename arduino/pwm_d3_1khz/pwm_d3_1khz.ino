/*
 * 1 kHz hardware PWM on D3 — Arduino Nano 33 BLE / BLE Sense (nRF52840)
 * ---------------------------------------------------------------------
 * Outputs a continuous 1000 Hz square wave on digital pin D3 at 50 % duty.
 *
 * Uses the mbed `PwmOut` API instead of analogWrite(): analogWrite() on this
 * core is locked to ~500 Hz, while PwmOut lets the period be set explicitly and
 * runs on one of the nRF52840's PWM peripherals, so the signal is exact and
 * jitter-free with zero CPU load (loop() stays empty).
 *
 * Output levels are 0 / 3.3 V — do NOT feed 5 V logic back into this pin.
 * Max continuous current per GPIO is a few mA; drive LEDs through a series
 * resistor and anything heavier through a transistor/MOSFET.
 *
 * Board: Arduino Mbed OS Nano Boards -> Arduino Nano 33 BLE.
 */

#include "mbed.h"

const int   PWM_PIN     = 3;       // D3
const float PWM_FREQ_HZ = 1000.0f; // 1 kHz
const float PWM_DUTY    = 0.5f;    // 0.0 - 1.0 (0.5 = square wave)

mbed::PwmOut *pwm;

void setup() {
  pwm = new mbed::PwmOut(digitalPinToPinName(PWM_PIN));

  // period [us] = 1e6 / freq, rounded to the nearest microsecond -> 1000 us.
  int periodUs = (int)(1000000.0f / PWM_FREQ_HZ + 0.5f);
  pwm->period_us(periodUs);

  pwm->write(PWM_DUTY);
}

void loop() {
  // Nothing to do: the PWM peripheral generates the waveform on its own.
}
