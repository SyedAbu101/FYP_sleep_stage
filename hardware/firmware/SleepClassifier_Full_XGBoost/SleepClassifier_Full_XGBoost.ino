/*
 * Sleep Stage Classifier - FULL XGBOOST MODEL (50 trees)
 *
 * Hardware: XIAO nRF52840 Sense
 * Model: XGBoost (50 trees, depth 5, 134 features)
 * Performance: Kappa 0.77 (train), ~0.30-0.32 (CV)
 *
 * This is the REAL production model, not simplified!
 *
 * Wiring (optional):
 *   - LED (built-in) indicates sleep stage
 *   - Piezo buzzer on D2 for NREM detection (optional)
 *
 * Libraries needed:
 *   - Seeed_Arduino_LSM6DS3 (IMU)
 *   - Seeed_Arduino_Mic (Microphone)
 */

#include <Arduino.h>
#include <Wire.h>
#include "LSM6DS3.h"
// #include <mic.h>  // Uncomment if you want audio features

// Include the FULL XGBoost model (50 trees)
#include "sleep_model/xgboost_50tree.h"
#include "sleep_model/scaler_params.h"
#include "sleep_model/features.h"

// ==================== CONFIGURATION ====================
#define EPOCH_SECONDS       30      // Classify every 30 seconds
#define IMU_SAMPLE_RATE_HZ  50      // Sample IMU at 50 Hz
#define IMU_INTERVAL_MS     (1000 / IMU_SAMPLE_RATE_HZ)

#define LED_PIN             LED_BUILTIN
#define BUZZER_PIN          D2      // Optional piezo buzzer

// ==================== HARDWARE ====================
LSM6DS3 imu(I2C_MODE, 0x6A);

// ==================== XGBOOST CLASSIFIER ====================
Eloquent::ML::Port::XGBClassifier classifier;

// ==================== FEATURE ACCUMULATORS ====================
// We accumulate statistics over each 30-second epoch

struct ChannelStats {
    double sum;
    double sum_sq;
    double max_val;
    double min_val;
    uint32_t count;
};

// 7 IMU channels: ax, ay, az, gx, gy, gz, tempC
ChannelStats stats[7];
const char* stat_names[] = {"ax", "ay", "az", "gx", "gy", "gz", "tempC"};

// History for temporal features (previous epoch values)
struct EpochHistory {
    double acc_mag;
    double gyro_mag;
    double total_movement;
    double acc_mag_normalized;
    double gyro_mag_normalized;
};

EpochHistory prev_epoch = {0};
bool has_prev = false;

// Patient normalization (running statistics)
struct PatientStats {
    double acc_mag_sum;
    double acc_mag_sum_sq;
    double gyro_mag_sum;
    double gyro_mag_sum_sq;
    double total_mov_sum;
    double total_mov_sum_sq;
    uint32_t n_epochs;
};

PatientStats patient = {0};

// Timing
unsigned long epoch_start_time = 0;
unsigned long sleep_start_time = 0;
unsigned long last_sample_time = 0;
uint32_t epoch_number = 0;

// ==================== SETUP ====================
void setup() {
    Serial.begin(115200);

    // Wait for serial (with timeout)
    unsigned long start = millis();
    while (!Serial && (millis() - start) < 5000);

    Serial.println();
    Serial.println("========================================");
    Serial.println("  Sleep Stage Classifier - FULL MODEL  ");
    Serial.println("========================================");
    Serial.print("Model: XGBoost (");
    Serial.print(NUM_TREES);
    Serial.print(" trees, ");
    Serial.print(NUM_FEATURES);
    Serial.println(" features)");
    Serial.println("========================================");
    Serial.println();

    // Initialize LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Initialize buzzer (optional)
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    // Initialize IMU
    Serial.print("Initializing IMU... ");
    if (imu.begin() != 0) {
        Serial.println("FAILED!");
        Serial.println("Check IMU wiring and I2C address.");
        while (1) {
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
            delay(200);
        }
    }
    Serial.println("OK");

    // Initialize timing
    sleep_start_time = millis();
    epoch_start_time = millis();
    last_sample_time = millis();

    // Reset statistics
    reset_stats();

    Serial.println();
    Serial.println("Ready! Collecting data...");
    Serial.println("Epoch | Time(min) | Movement | Prediction");
    Serial.println("------+----------+----------+-----------");
}

// ==================== MAIN LOOP ====================
void loop() {
    unsigned long now = millis();

    // Sample IMU at fixed rate
    if (now - last_sample_time >= IMU_INTERVAL_MS) {
        last_sample_time = now;
        sample_imu();
    }

    // Every 30 seconds: run classification
    if (now - epoch_start_time >= EPOCH_SECONDS * 1000UL) {
        classify_epoch();

        // Reset for next epoch
        reset_stats();
        epoch_start_time = now;
        epoch_number++;
    }
}

// ==================== IMU SAMPLING ====================
void sample_imu() {
    // Read raw IMU data
    float ax = imu.readFloatAccelX();
    float ay = imu.readFloatAccelY();
    float az = imu.readFloatAccelZ();
    float gx = imu.readFloatGyroX();
    float gy = imu.readFloatGyroY();
    float gz = imu.readFloatGyroZ();
    float tempC = imu.readTempC();

    // Accumulate statistics
    update_stats(&stats[0], ax);
    update_stats(&stats[1], ay);
    update_stats(&stats[2], az);
    update_stats(&stats[3], gx);
    update_stats(&stats[4], gy);
    update_stats(&stats[5], gz);
    update_stats(&stats[6], tempC);
}

void update_stats(ChannelStats* s, float value) {
    if (s->count == 0) {
        s->max_val = value;
        s->min_val = value;
    } else {
        if (value > s->max_val) s->max_val = value;
        if (value < s->min_val) s->min_val = value;
    }

    s->sum += value;
    s->sum_sq += value * value;
    s->count++;
}

void reset_stats() {
    for (int i = 0; i < 7; i++) {
        stats[i].sum = 0;
        stats[i].sum_sq = 0;
        stats[i].max_val = 0;
        stats[i].min_val = 0;
        stats[i].count = 0;
    }
}

// ==================== CLASSIFICATION ====================
void classify_epoch() {

    // Build feature vector (134 features)
    double features[NUM_FEATURES];

    // Initialize all to 0 (important for missing features like audio)
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i] = 0.0;
    }

    // ---- 1. RAW IMU STATISTICS (30 features) ----
    // For each channel: mean, std, min, max

    features[FEAT_AX_MEAN] = compute_mean(&stats[0]);
    features[FEAT_AX_STD] = compute_std(&stats[0]);
    features[FEAT_AX_MIN] = stats[0].min_val;
    features[FEAT_AX_MAX] = stats[0].max_val;

    features[FEAT_AY_MEAN] = compute_mean(&stats[1]);
    features[FEAT_AY_STD] = compute_std(&stats[1]);
    features[FEAT_AY_MIN] = stats[1].min_val;
    features[FEAT_AY_MAX] = stats[1].max_val;

    features[FEAT_AZ_MEAN] = compute_mean(&stats[2]);
    features[FEAT_AZ_STD] = compute_std(&stats[2]);
    features[FEAT_AZ_MIN] = stats[2].min_val;
    features[FEAT_AZ_MAX] = stats[2].max_val;

    features[FEAT_GX_MEAN] = compute_mean(&stats[3]);
    features[FEAT_GX_STD] = compute_std(&stats[3]);
    features[FEAT_GX_MIN] = stats[3].min_val;
    features[FEAT_GX_MAX] = stats[3].max_val;

    features[FEAT_GY_MEAN] = compute_mean(&stats[4]);
    features[FEAT_GY_STD] = compute_std(&stats[4]);
    features[FEAT_GY_MIN] = stats[4].min_val;
    features[FEAT_GY_MAX] = stats[4].max_val;

    features[FEAT_GZ_MEAN] = compute_mean(&stats[5]);
    features[FEAT_GZ_STD] = compute_std(&stats[5]);
    features[FEAT_GZ_MIN] = stats[5].min_val;
    features[FEAT_GZ_MAX] = stats[5].max_val;

    features[FEAT_TEMPC_MEAN] = compute_mean(&stats[6]);
    features[FEAT_TEMPC_STD] = compute_std(&stats[6]);
    features[FEAT_TEMPC_MIN] = stats[6].min_val;
    features[FEAT_TEMPC_MAX] = stats[6].max_val;

    // Audio features (set to defaults - microphone not used in this demo)
    // In production, you'd fill these from actual mic data
    features[FEAT_MIC_RMS_MEAN] = 0.01;
    features[FEAT_ZCR_MEAN] = 0.1;
    // ... (other audio features default to 0)

    // ---- 2. DERIVED MOVEMENT FEATURES ----

    double ax_m = features[FEAT_AX_MEAN];
    double ay_m = features[FEAT_AY_MEAN];
    double az_m = features[FEAT_AZ_MEAN];
    double ax_s = features[FEAT_AX_STD];
    double ay_s = features[FEAT_AY_STD];
    double az_s = features[FEAT_AZ_STD];

    double gx_m = features[FEAT_GX_MEAN];
    double gy_m = features[FEAT_GY_MEAN];
    double gz_m = features[FEAT_GZ_MEAN];
    double gx_s = features[FEAT_GX_STD];
    double gy_s = features[FEAT_GY_STD];
    double gz_s = features[FEAT_GZ_STD];

    // Magnitudes
    double acc_mag = sqrt(ax_m*ax_m + ay_m*ay_m + az_m*az_m);
    double acc_mag_std = sqrt(ax_s*ax_s + ay_s*ay_s + az_s*az_s);
    double gyro_mag = sqrt(gx_m*gx_m + gy_m*gy_m + gz_m*gz_m);
    double gyro_mag_std = sqrt(gx_s*gx_s + gy_s*gy_s + gz_s*gz_s);
    double total_movement = acc_mag + gyro_mag;

    features[FEAT_ACC_MAG] = acc_mag;
    features[FEAT_ACC_MAG_STD] = acc_mag_std;
    features[FEAT_GYRO_MAG] = gyro_mag;
    features[FEAT_GYRO_MAG_STD] = gyro_mag_std;
    features[FEAT_TOTAL_MOVEMENT] = total_movement;
    features[FEAT_TOTAL_MOVEMENT_STD] = acc_mag_std + gyro_mag_std;

    // Dominance
    double axis_sum = ax_s + ay_s + az_s + 1e-6;
    features[FEAT_AX_DOMINANCE] = ax_s / axis_sum;
    features[FEAT_AY_DOMINANCE] = ay_s / axis_sum;
    features[FEAT_AZ_DOMINANCE] = az_s / axis_sum;

    // Tilt estimate
    features[FEAT_TILT_ESTIMATE] = atan2(sqrt(ax_m*ax_m + ay_m*ay_m), az_m + 1e-6);

    // ---- 3. PATIENT NORMALIZATION ----

    // Update running patient statistics
    patient.acc_mag_sum += acc_mag;
    patient.acc_mag_sum_sq += acc_mag * acc_mag;
    patient.gyro_mag_sum += gyro_mag;
    patient.gyro_mag_sum_sq += gyro_mag * gyro_mag;
    patient.total_mov_sum += total_movement;
    patient.total_mov_sum_sq += total_movement * total_movement;
    patient.n_epochs++;

    double patient_acc_mean = patient.acc_mag_sum / patient.n_epochs;
    double patient_acc_std = sqrt(max(0.0, patient.acc_mag_sum_sq / patient.n_epochs - patient_acc_mean * patient_acc_mean)) + 1e-6;
    double patient_gyro_mean = patient.gyro_mag_sum / patient.n_epochs;
    double patient_gyro_std = sqrt(max(0.0, patient.gyro_mag_sum_sq / patient.n_epochs - patient_gyro_mean * patient_gyro_mean)) + 1e-6;
    double patient_mov_mean = patient.total_mov_sum / patient.n_epochs;
    double patient_mov_std = sqrt(max(0.0, patient.total_mov_sum_sq / patient.n_epochs - patient_mov_mean * patient_mov_mean)) + 1e-6;

    features[FEAT_PATIENT_ACC_MAG_MEAN] = patient_acc_mean;
    features[FEAT_PATIENT_ACC_MAG_STD] = patient_acc_std;
    features[FEAT_PATIENT_GYRO_MAG_MEAN] = patient_gyro_mean;
    features[FEAT_PATIENT_GYRO_MAG_STD] = patient_gyro_std;
    features[FEAT_PATIENT_TOTAL_MOVEMENT_MEAN] = patient_mov_mean;
    features[FEAT_PATIENT_TOTAL_MOVEMENT_STD] = patient_mov_std;

    // Normalized features
    double acc_mag_norm = (acc_mag - patient_acc_mean) / patient_acc_std;
    double gyro_mag_norm = (gyro_mag - patient_gyro_mean) / patient_gyro_std;
    double total_mov_norm = (total_movement - patient_mov_mean) / patient_mov_std;

    features[FEAT_ACC_MAG_NORMALIZED] = acc_mag_norm;
    features[FEAT_GYRO_MAG_NORMALIZED] = gyro_mag_norm;
    features[FEAT_TOTAL_MOVEMENT_NORMALIZED] = total_mov_norm;

    // ---- 4. TEMPORAL FEATURES ----

    if (has_prev) {
        features[FEAT_ACC_MAG_PREV] = prev_epoch.acc_mag;
        features[FEAT_ACC_MAG_DIFF] = acc_mag - prev_epoch.acc_mag;
        features[FEAT_ACC_MAG_ABS_DIFF] = abs(acc_mag - prev_epoch.acc_mag);

        features[FEAT_GYRO_MAG_PREV] = prev_epoch.gyro_mag;
        features[FEAT_GYRO_MAG_DIFF] = gyro_mag - prev_epoch.gyro_mag;
        features[FEAT_GYRO_MAG_ABS_DIFF] = abs(gyro_mag - prev_epoch.gyro_mag);

        features[FEAT_TOTAL_MOVEMENT_PREV] = prev_epoch.total_movement;
        features[FEAT_TOTAL_MOVEMENT_DIFF] = total_movement - prev_epoch.total_movement;
        features[FEAT_TOTAL_MOVEMENT_ABS_DIFF] = abs(total_movement - prev_epoch.total_movement);

        features[FEAT_ACC_MAG_NORMALIZED_PREV] = prev_epoch.acc_mag_normalized;
        features[FEAT_ACC_MAG_NORMALIZED_DIFF] = acc_mag_norm - prev_epoch.acc_mag_normalized;
        features[FEAT_ACC_MAG_NORMALIZED_ABS_DIFF] = abs(acc_mag_norm - prev_epoch.acc_mag_normalized);

        features[FEAT_GYRO_MAG_NORMALIZED_PREV] = prev_epoch.gyro_mag_normalized;
        features[FEAT_GYRO_MAG_NORMALIZED_DIFF] = gyro_mag_norm - prev_epoch.gyro_mag_normalized;
        features[FEAT_GYRO_MAG_NORMALIZED_ABS_DIFF] = abs(gyro_mag_norm - prev_epoch.gyro_mag_normalized);
    }

    // Update history
    prev_epoch.acc_mag = acc_mag;
    prev_epoch.gyro_mag = gyro_mag;
    prev_epoch.total_movement = total_movement;
    prev_epoch.acc_mag_normalized = acc_mag_norm;
    prev_epoch.gyro_mag_normalized = gyro_mag_norm;
    has_prev = true;

    // ---- 5. TIME FEATURES ----

    double time_min = (millis() - sleep_start_time) / 60000.0;
    double time_hours = time_min / 60.0;

    features[FEAT_TIME_SINCE_START_MIN] = time_min;
    features[FEAT_TIME_SINCE_START_HOURS] = time_hours;

    // Sleep cycle (90-minute cycles)
    double cycle_phase = (time_min / 90.0) * 2.0 * PI;
    features[FEAT_SLEEP_CYCLE_SIN] = sin(cycle_phase);
    features[FEAT_SLEEP_CYCLE_COS] = cos(cycle_phase);

    // ---- 6. APPLY SCALING ----

    double scaled_features[NUM_FEATURES];
    apply_scaling(features, scaled_features);

    // ---- 7. RUN XGBOOST INFERENCE (THE REAL MODEL!) ----

    // Convert to float array for predict() function
    float scaled_features_float[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features_float[i] = (float)scaled_features[i];
    }

    unsigned long inference_start = micros();
    int prediction = classifier.predict(scaled_features_float);  // THIS IS THE FULL 50-TREE XGBOOST!
    unsigned long inference_time = micros() - inference_start;

    // ---- 8. DISPLAY RESULTS ----

    const char* stage_names[] = {"WAKE", "NREM", "REM"};

    // Print results (Arduino-compatible way)
    Serial.print(epoch_number);
    Serial.print(" | ");
    Serial.print(time_min, 2);
    Serial.print(" | ");
    Serial.print(total_movement, 4);
    Serial.print(" | ");
    Serial.print(stage_names[prediction]);
    Serial.print(" (");
    Serial.print(inference_time / 1000.0, 2);
    Serial.println(" ms)");

    // ---- 9. VISUAL/AUDIO FEEDBACK ----

    // LED: blink pattern based on stage
    if (prediction == 0) {
        // Wake: fast blink
        for (int i = 0; i < 5; i++) {
            digitalWrite(LED_PIN, HIGH);
            delay(100);
            digitalWrite(LED_PIN, LOW);
            delay(100);
        }
    } else if (prediction == 1) {
        // NREM: slow blink
        digitalWrite(LED_PIN, HIGH);
        delay(500);
        digitalWrite(LED_PIN, LOW);
    } else {
        // REM: double blink
        for (int i = 0; i < 2; i++) {
            digitalWrite(LED_PIN, HIGH);
            delay(200);
            digitalWrite(LED_PIN, LOW);
            delay(200);
        }
    }

    // Optional: Buzzer for NREM detection (demonstration)
    // if (prediction == 1) {
    //     tone(BUZZER_PIN, 1000, 200);
    // }
}

// ==================== HELPER FUNCTIONS ====================

double compute_mean(ChannelStats* s) {
    if (s->count == 0) return 0.0;
    return s->sum / s->count;
}

double compute_std(ChannelStats* s) {
    if (s->count < 2) return 0.0;
    double mean = compute_mean(s);
    double variance = (s->sum_sq / s->count) - (mean * mean);
    return sqrt(max(0.0, variance));
}
