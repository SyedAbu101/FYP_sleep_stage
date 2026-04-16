#ifndef SLEEP_TYPES_H
#define SLEEP_TYPES_H

#include <stdint.h>

// Per-channel IMU accumulator (mean, std, min, max over one epoch)
struct ChanStats {
    double   sum, sum_sq, hi, lo;
    uint32_t n;
};

// Rolling 5-epoch history for temporal diff features
struct EpochHist {
    double acc_mag[5];
    double acc_mag_norm;
    double acc_mag_val;
    double gyro_mag_val;
    double total_mov_val;
    int    idx;
    int    count;
};

// Running patient-level normalization (persists across all epochs)
struct PatientNorm {
    double   acc_sum,  acc_sum_sq;
    double   gyro_sum, gyro_sum_sq;
    double   mov_sum,  mov_sum_sq;
    uint32_t n_epochs;
    uint32_t n_wake, n_sleep;
};

// Per-epoch audio feature accumulators (reset each epoch)
struct AudioAccum {
    // Energy (RMS per window)
    double e_sum, e_sum_sq, e_max;
    // Zero-crossing rate
    double zcr_sum, zcr_sum_sq;
    // Crest factor (|peak| / RMS per window)
    double crest_sum, crest_sum_sq;
    // Silence / activity
    uint32_t silence_samples, total_samples, active_windows;
    // Spectral centroid (FFT, normalized 0–1)
    double cent_sum, cent_sum_sq;
    // Snoring band power (100–500 Hz, normalized by total power)
    double snor_sum, snor_sum_sq;
    // Breathing proxy band power (0–62 Hz, normalized by total power)
    double breath_sum, breath_sum_sq;
    // Spectral richness (fraction of bins > mean power)
    double rich_sum;
    uint32_t n_windows;      // windows where FFT was computed (non-silent)
    uint32_t n_all_windows;  // all windows including silent ones
};

#endif // SLEEP_TYPES_H
