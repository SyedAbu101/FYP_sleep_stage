/*
 * Sleep Stage Classifier — AUDIO-ENHANCED XGBOOST
 *
 * Hardware : XIAO nRF52840 Sense
 * Model    : XGBoost trained on 97 IMU + Top 20 audio features (117 total)
 * Expected : Kappa ~0.3223  (+0.029 over IMU-only baseline of 0.2938)
 *
 * ── BEFORE FLASHING ──────────────────────────────────────────────────────────
 *  1. Train the 117-feature model:
 *       cd src/evaluation
 *       python audio_comparison_weighted.py   (verify performance first)
 *       python train_audio_model.py           (TODO: write this script)
 *
 *  2. Export model to C headers:
 *       python src/modeling/export_audio_to_c.py   (TODO: write this script)
 *     Outputs → sleep_model_audio/xgboost_audio.h
 *                sleep_model_audio/scaler_audio.h
 *                sleep_model_audio/features_audio.h   (replaces placeholder)
 *
 *  3. Flash this sketch.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Audio Pipeline:
 *   PDM mic @ 16 kHz  →  256-sample window every 16 ms
 *   Per window        :  time-domain stats (RMS, ZCR, crest, silence)
 *                     +  256-point FFT (self-contained radix-2) for power bands
 *   Per epoch (30 s)  :  ~1875 windows → mean/std/max of each stat
 *                     →  20 audio features appended to 97 IMU features
 *
 * Libraries:
 *   Seeed_Arduino_LSM6DS3  (IMU)
 *   PDM.h                  (PDM microphone — included with nRF52840 board package)
 *   (no external FFT library needed — radix-2 FFT is self-contained)
 */

#include <Arduino.h>
#include <Wire.h>
#include "LSM6DS3.h"
#include <PDM.h>
#include "sleep_types.h"
// Generated model headers — produced by export_audio_to_c.py
// xgboost_audio.h  defines: score(double* input, double* output)
// scaler_audio.h   defines: apply_scaling(raw, scaled), SCALER_MEAN, SCALER_SCALE
// features_audio.h defines: FEAT_* indices, NUM_FEATURES
#include "sleep_model_audio/xgboost_audio.h"
#include "sleep_model_audio/scaler_audio.h"
#include "sleep_model_audio/features_audio.h"

// ── CONFIGURATION ─────────────────────────────────────────────────────────────

#define EPOCH_SECONDS        30
#define IMU_SAMPLE_RATE_HZ   50
#define IMU_INTERVAL_MS      (1000 / IMU_SAMPLE_RATE_HZ)

// Microphone
#define MIC_SAMPLE_RATE      16000
#define FFT_SIZE             256          // 256 samples = 16 ms window @ 16 kHz
#define FREQ_RESOLUTION_HZ   (MIC_SAMPLE_RATE / FFT_SIZE)   // 62.5 Hz/bin
// Separate silence thresholds to avoid comparing incompatible statistics:
//   SAMPLE: instantaneous |s| < 500  → count towards silence_samples ratio
//   RMS   : window RMS < 200        → mark window inactive / skip FFT
//   Both are in int16 amplitude units (0–32767).
#define SILENCE_THRESHOLD_SAMPLE  500    // ~1.5% full scale, per-sample
#define SILENCE_THRESHOLD_RMS     200    // ~0.6% full scale, per-window RMS

// Frequency bands (bin indices for 16 kHz / 256-point FFT)
// Each bin = 62.5 Hz
#define BIN_SNORING_LO    2    // ~125 Hz
#define BIN_SNORING_HI    8    // ~500 Hz
#define BIN_BREATH_LO     0    // DC / ~0 Hz
#define BIN_BREATH_HI     1    // ~62 Hz  (breathing proxy — low-freq energy)
#define BIN_NYQUIST       (FFT_SIZE / 2)   // 128

// RGB LED pins (XIAO nRF52840 Sense — all active-LOW)
// LEDR / LEDG / LEDB are defined by the board package.
// Wake → Red,  NREM → Green,  REM → Blue

// ── HARDWARE ──────────────────────────────────────────────────────────────────

LSM6DS3 imu(I2C_MODE, 0x6A);
Eloquent::ML::Port::XGBClassifier classifier;

// ── MICROPHONE (standard PDM.h — included with nRF52840 board package) ────────

volatile bool  audio_window_ready = false;
static int16_t mic_buf[FFT_SIZE];
static int16_t pdm_tmp[FFT_SIZE];      // PDM read buffer (filled by ISR)
static volatile int pdm_samples_read = 0;

void on_pdm_data() {
    int n = PDM.available();
    // PDM delivers 2 bytes per sample; read up to FFT_SIZE samples
    if (n > (int)(FFT_SIZE * 2)) n = FFT_SIZE * 2;
    PDM.read(pdm_tmp, n);
    pdm_samples_read = n / 2;
    if (pdm_samples_read >= FFT_SIZE) {
        memcpy(mic_buf, pdm_tmp, FFT_SIZE * sizeof(int16_t));
        audio_window_ready = true;
    }
}

// ── FFT (self-contained radix-2 DIT, no CMSIS-DSP required) ──────────────────

static float fft_in[FFT_SIZE];
static float fft_out[FFT_SIZE * 2];      // interleaved complex: [re0,im0, re1,im1, ...]
static float fft_mag[FFT_SIZE / 2];      // power spectrum

// In-place radix-2 DIT FFT. fft_buf is 2*n floats, interleaved real/imag.
static void fft_radix2(float *fft_buf, int n) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = fft_buf[2*i],   ti = fft_buf[2*i+1];
            fft_buf[2*i]   = fft_buf[2*j];   fft_buf[2*i+1] = fft_buf[2*j+1];
            fft_buf[2*j]   = tr;             fft_buf[2*j+1] = ti;
        }
    }
    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float ur = fft_buf[2*a], ui = fft_buf[2*a+1];
                float vr = fft_buf[2*b] * cur_re - fft_buf[2*b+1] * cur_im;
                float vi = fft_buf[2*b] * cur_im + fft_buf[2*b+1] * cur_re;
                fft_buf[2*a]   = ur + vr;  fft_buf[2*a+1] = ui + vi;
                fft_buf[2*b]   = ur - vr;  fft_buf[2*b+1] = ui - vi;
                float tmp = cur_re * wre - cur_im * wim;
                cur_im    = cur_re * wim + cur_im * wre;
                cur_re    = tmp;
            }
        }
    }
}

// ── IMU ACCUMULATORS ──────────────────────────────────────────────────────────

ChanStats imu_ch[7];   // ax ay az gx gy gz tempC

// Rolling history for temporal features (5 epochs)
EpochHist hist = {0};
bool has_prev = false;

// Running patient-level normalization (persists across all epochs)
PatientNorm pat = {0};

// Bout tracking (persists across epochs)
static uint32_t g_bout_count       = 0;
static uint32_t g_stillness_epochs = 0;
static bool     g_in_bout          = false;

// ── AUDIO ACCUMULATORS (reset each epoch) ────────────────────────────────────

AudioAccum aud = {0};

// ── TIMING ────────────────────────────────────────────────────────────────────

unsigned long epoch_start_ms = 0;
unsigned long sleep_start_ms = 0;
unsigned long last_imu_ms    = 0;
uint32_t      epoch_num      = 0;

// ── SETUP ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 5000);

    Serial.println();
    Serial.println("============================================");
    Serial.println("  Sleep Classifier  (Audio-Enhanced Model) ");
    Serial.println("============================================");
    Serial.print  ("  Features: IMU=97  Audio=20  Total="); Serial.println(NUM_FEATURES);
    Serial.println("  Expected Kappa: ~0.3223");
    Serial.println("============================================");
    Serial.println();

    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    set_rgb(false, false, false);  // all off

    // IMU
    Serial.print("IMU ... ");
    if (imu.begin() != 0) {
        Serial.println("FAILED — check wiring");
        while (1) { set_rgb(true, false, false); delay(200); set_rgb(false, false, false); delay(200); }
    }
    Serial.println("OK");

    // Microphone (PDM)
    Serial.print("Mic ... ");
    PDM.onReceive(on_pdm_data);
    PDM.setGain(20);
    if (!PDM.begin(1, MIC_SAMPLE_RATE)) {
        Serial.println("FAILED — audio features will be 0");
    } else {
        Serial.print("OK ("); Serial.print(MIC_SAMPLE_RATE); Serial.println(" Hz PDM)");
    }

    sleep_start_ms = millis();
    epoch_start_ms = millis();
    last_imu_ms    = millis();

    reset_imu();
    reset_audio();

    Serial.println();
    Serial.println("Collecting...  (first result in 30 s)");
    Serial.println("Epoch | Time (min) | Movement | Stage      | Infer (ms)");
    Serial.println("------+------------+----------+------------+-----------");
}

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────

void loop() {
    unsigned long now = millis();

    if (now - last_imu_ms >= IMU_INTERVAL_MS) {
        last_imu_ms = now;
        sample_imu();
    }

    if (audio_window_ready) {
        audio_window_ready = false;
        process_audio_window();
    }

    if (now - epoch_start_ms >= (unsigned long)EPOCH_SECONDS * 1000UL) {
        classify_epoch();
        reset_imu();
        reset_audio();
        epoch_start_ms = now;
        epoch_num++;
    }
}

// ── IMU SAMPLING ──────────────────────────────────────────────────────────────

void sample_imu() {
    float v[7] = {
        imu.readFloatAccelX(), imu.readFloatAccelY(), imu.readFloatAccelZ(),
        imu.readFloatGyroX(),  imu.readFloatGyroY(),  imu.readFloatGyroZ(),
        imu.readTempC()
    };
    for (int i = 0; i < 7; i++) accum_chan(&imu_ch[i], v[i]);
}

void accum_chan(ChanStats *s, float v) {
    if (s->n == 0) { s->hi = v; s->lo = v; }
    else { if (v > s->hi) s->hi = v; if (v < s->lo) s->lo = v; }
    s->sum    += v;
    s->sum_sq += (double)v * v;
    s->n++;
}

// ── AUDIO WINDOW PROCESSING (called every 256 samples = 16 ms) ────────────────

void process_audio_window() {
    // Thread safety: snapshot mic_buf while interrupts are disabled so that
    // an ISR firing mid-read cannot corrupt the samples we process.
    static int16_t local_buf[FFT_SIZE];
    noInterrupts();
    memcpy(local_buf, mic_buf, FFT_SIZE * sizeof(int16_t));
    interrupts();

    // ── Time-domain pass ─────────────────────────────────────────────────────
    double sum_sq  = 0.0;
    int    zcr     = 0;
    int32_t peak   = 0;
    uint32_t silent_samples = 0;
    int16_t prev_s = local_buf[0];

    for (int i = 0; i < FFT_SIZE; i++) {
        int16_t s = local_buf[i];
        sum_sq += (double)s * s;
        if (i > 0 && ((s >= 0) != (prev_s >= 0))) zcr++;
        if (abs(s) > peak) peak = abs(s);
        if (abs(s) < SILENCE_THRESHOLD_SAMPLE) silent_samples++;  // per-sample amplitude
        fft_in[i] = (float)s;   // raw int16 into FFT input
        prev_s = s;
    }

    double rms      = sqrt(sum_sq / FFT_SIZE);
    double zcr_rate = (double)zcr / FFT_SIZE;
    double crest    = (rms > 1.0) ? ((double)peak / rms) : 0.0;

    aud.e_sum     += rms;
    aud.e_sum_sq  += rms * rms;
    if (rms > aud.e_max) aud.e_max = rms;

    aud.zcr_sum    += zcr_rate;
    aud.zcr_sum_sq += zcr_rate * zcr_rate;

    aud.crest_sum    += crest;
    aud.crest_sum_sq += crest * crest;

    aud.silence_samples += silent_samples;
    aud.total_samples   += FFT_SIZE;
    if (rms > SILENCE_THRESHOLD_RMS) aud.active_windows++;  // per-window RMS
    aud.n_all_windows++;

    // ── FFT pass (skip very silent windows to save CPU) ────────────────────
    if (rms < SILENCE_THRESHOLD_RMS) return;

    // Pack real input into interleaved complex buffer (imaginary = 0)
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_out[2*i]   = fft_in[i];
        fft_out[2*i+1] = 0.0f;
    }
    fft_radix2(fft_out, FFT_SIZE);
    // Note: FFT output is NOT divided by N. This matches Python's np.fft.fft()
    // default behaviour. All spectral features below use band_power / total_pwr
    // ratios, so the absolute scale cancels out exactly.

    // Power spectrum (magnitude squared, one-sided)
    float total_pwr = 0.0f;
    for (int k = 0; k < BIN_NYQUIST; k++) {
        float re = fft_out[2 * k];
        float im = fft_out[2 * k + 1];
        fft_mag[k] = re * re + im * im;
        total_pwr += fft_mag[k];
    }

    if (total_pwr < 1e-10f) return;

    // Spectral centroid (normalized: 0 = DC, 1 = Nyquist).
    // Training used the same 0–1 normalisation (bin_index / BIN_NYQUIST),
    // NOT Hz units, so no FREQ_RESOLUTION_HZ conversion is needed.
    float centroid = 0.0f;
    for (int k = 1; k < BIN_NYQUIST; k++)
        centroid += (float)k * fft_mag[k];
    centroid = (centroid / total_pwr) / (float)BIN_NYQUIST;

    aud.cent_sum    += centroid;
    aud.cent_sum_sq += centroid * centroid;

    // Snoring band (100–500 Hz → bins BIN_SNORING_LO..BIN_SNORING_HI)
    float snor_pwr = 0.0f;
    for (int k = BIN_SNORING_LO; k <= BIN_SNORING_HI; k++)
        snor_pwr += fft_mag[k];
    float snor_norm = snor_pwr / total_pwr;

    aud.snor_sum    += snor_norm;
    aud.snor_sum_sq += snor_norm * snor_norm;

    // Breathing proxy band (0–62 Hz → bins 0..1)
    float breath_pwr = 0.0f;
    for (int k = BIN_BREATH_LO; k <= BIN_BREATH_HI; k++)
        breath_pwr += fft_mag[k];
    float breath_norm = breath_pwr / total_pwr;

    aud.breath_sum    += breath_norm;
    aud.breath_sum_sq += breath_norm * breath_norm;

    // Spectral richness: fraction of bins with power > mean power
    float mean_pwr = total_pwr / BIN_NYQUIST;
    int   rich_bins = 0;
    for (int k = 0; k < BIN_NYQUIST; k++)
        if (fft_mag[k] > mean_pwr) rich_bins++;
    aud.rich_sum += (float)rich_bins / (float)BIN_NYQUIST;

    aud.n_windows++;
}

// ── EPOCH CLASSIFICATION ──────────────────────────────────────────────────────

void classify_epoch() {
    double f[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) f[i] = 0.0;

    // ── 1. Raw IMU stats (28 features: 7 channels × mean/std/min/max) ─────
    // Channel layout: 0=ax 1=ay 2=az 3=gx 4=gy 5=gz 6=tempC
    // features.h defines FEAT_*_MEAN at offsets: ax=0, ay=4, az=8, gx=12, gy=16, gz=20, tempC=24
    const int ch_base[7] = {
        FEAT_AX_MEAN, FEAT_AY_MEAN, FEAT_AZ_MEAN,
        FEAT_GX_MEAN, FEAT_GY_MEAN, FEAT_GZ_MEAN, FEAT_TEMPC_MEAN
    };
    for (int ch = 0; ch < 7; ch++) {
        int b = ch_base[ch];
        f[b+0] = chan_mean(&imu_ch[ch]);
        f[b+1] = chan_std(&imu_ch[ch]);
        f[b+2] = imu_ch[ch].lo;
        f[b+3] = imu_ch[ch].hi;
    }

    // ── 2. Movement magnitudes ──────────────────────────────────────────────
    double ax = f[FEAT_AX_MEAN], ay = f[FEAT_AY_MEAN], az = f[FEAT_AZ_MEAN];
    double axs = f[FEAT_AX_STD], ays = f[FEAT_AY_STD], azs = f[FEAT_AZ_STD];
    double gx = f[FEAT_GX_MEAN], gy = f[FEAT_GY_MEAN], gz = f[FEAT_GZ_MEAN];
    double gxs = f[FEAT_GX_STD], gys = f[FEAT_GY_STD], gzs = f[FEAT_GZ_STD];

    double acc_mag  = sqrt(ax*ax + ay*ay + az*az);
    double acc_std  = sqrt(axs*axs + ays*ays + azs*azs);
    double gyro_mag = sqrt(gx*gx + gy*gy + gz*gz);
    double gyro_std = sqrt(gxs*gxs + gys*gys + gzs*gzs);
    double total_mov = acc_mag + gyro_mag;

    f[FEAT_ACC_MAG]          = acc_mag;
    f[FEAT_ACC_MAG_STD]      = acc_std;
    f[FEAT_ACC_MAG_MAX]      = imu_ch[0].hi;   // ax hi (proxy)
    f[FEAT_ACC_MAG_MIN]      = imu_ch[0].lo;
    f[FEAT_ACC_RANGE]        = imu_ch[0].hi - imu_ch[0].lo;
    f[FEAT_GYRO_MAG]         = gyro_mag;
    f[FEAT_GYRO_MAG_STD]     = gyro_std;
    f[FEAT_GYRO_MAG_MAX]     = imu_ch[3].hi;
    f[FEAT_GYRO_MAG_MIN]     = imu_ch[3].lo;
    f[FEAT_GYRO_RANGE]       = imu_ch[3].hi - imu_ch[3].lo;
    f[FEAT_TOTAL_MOVEMENT]   = total_mov;
    f[FEAT_TOTAL_MOVEMENT_STD] = acc_std + gyro_std;

    // ── 3. Orientation ─────────────────────────────────────────────────────
    double axis_sum = axs + ays + azs + 1e-9;
    f[FEAT_AX_DOMINANCE]  = axs / axis_sum;
    f[FEAT_AY_DOMINANCE]  = ays / axis_sum;
    f[FEAT_AZ_DOMINANCE]  = azs / axis_sum;
    f[FEAT_TILT_ESTIMATE] = atan2(sqrt(ax*ax + ay*ay), az + 1e-9);
    f[FEAT_AX_AY_RATIO]   = (ays > 1e-9) ? (axs / ays) : 0.0;

    // ── 4. Patient normalization (running, persists across epochs) ─────────
    pat.acc_sum     += acc_mag;   pat.acc_sum_sq  += acc_mag  * acc_mag;
    pat.gyro_sum    += gyro_mag;  pat.gyro_sum_sq += gyro_mag * gyro_mag;
    pat.mov_sum     += total_mov; pat.mov_sum_sq  += total_mov * total_mov;
    pat.n_epochs++;

    double p_acc_mean  = pat.acc_sum  / pat.n_epochs;
    double p_gyro_mean = pat.gyro_sum / pat.n_epochs;
    double p_mov_mean  = pat.mov_sum  / pat.n_epochs;

    double p_acc_std  = sqrt(max(0.0, pat.acc_sum_sq  / pat.n_epochs - p_acc_mean  * p_acc_mean))  + 1e-9;
    double p_gyro_std = sqrt(max(0.0, pat.gyro_sum_sq / pat.n_epochs - p_gyro_mean * p_gyro_mean)) + 1e-9;
    double p_mov_std  = sqrt(max(0.0, pat.mov_sum_sq  / pat.n_epochs - p_mov_mean  * p_mov_mean))  + 1e-9;

    f[FEAT_PATIENT_ACC_MEAN]      = p_acc_mean;
    f[FEAT_PATIENT_ACC_STD]       = p_acc_std;
    f[FEAT_PATIENT_GYRO_MEAN]     = p_gyro_mean;
    f[FEAT_PATIENT_GYRO_STD]      = p_gyro_std;
    // patient_total_movement_mean / std not in this dataset — left at 0
    f[FEAT_PATIENT_WAKE_RATIO]    = (pat.n_epochs > 0) ? (double)pat.n_wake / pat.n_epochs : 0.5;
    f[FEAT_PATIENT_ACC_MEDIAN]    = p_acc_mean;  // approximation: running mean ≈ median
    // True median would require storing all past acc_mag values. For a sleep
    // session the distribution is roughly unimodal, so mean is a reasonable
    // proxy. Acceptable inaccuracy: expected ~5–10% difference from true median.
    f[FEAT_ACC_VS_PATIENT_MEDIAN] = acc_mag - p_acc_mean;

    // Normalized
    double acc_norm = (acc_mag  - p_acc_mean)  / p_acc_std;
    double gyr_norm = (gyro_mag - p_gyro_mean) / p_gyro_std;

    f[FEAT_ACC_MAG_NORMALIZED]  = acc_norm;
    f[FEAT_GYRO_MAG_NORMALIZED] = gyr_norm;
    // total_movement_normalized / zscore not in this dataset — left at 0

    // ── 5. Temporal diff (vs previous epoch) ───────────────────────────────
    if (has_prev) {
        f[FEAT_ACC_MAG_PREV]        = hist.acc_mag_val;
        f[FEAT_ACC_MAG_DIFF]        = acc_mag  - hist.acc_mag_val;
        f[FEAT_ACC_MAG_DIFF_ABS]    = abs(acc_mag - hist.acc_mag_val);
        f[FEAT_GYRO_MAG_PREV]       = hist.gyro_mag_val;
        f[FEAT_GYRO_MAG_DIFF]       = gyro_mag - hist.gyro_mag_val;
        f[FEAT_GYRO_MAG_DIFF_ABS]   = abs(gyro_mag - hist.gyro_mag_val);
        f[FEAT_TOTAL_MOVEMENT_PREV] = hist.total_mov_val;
    }

    // Rolling mean/std (3-epoch and 5-epoch windows)
    hist.acc_mag[hist.idx] = acc_mag;
    hist.idx = (hist.idx + 1) % 5;
    if (hist.count < 5) hist.count++;

    auto rolling_stats = [&](int n, double &rm, double &rs) {
        double s = 0, ss = 0;
        int start = (hist.idx - 1 + 5) % 5;
        for (int i = 0; i < n; i++) {
            double v = hist.acc_mag[(start - i + 5) % 5];
            s += v; ss += v * v;
        }
        rm = s / n;
        rs = (n > 1) ? sqrt(max(0.0, ss/n - rm*rm)) : 0.0;
    };

    int n3 = min(hist.count, 3), n5 = min(hist.count, 5);
    double rm3, rs3, rm5, rs5;
    rolling_stats(n3, rm3, rs3);
    rolling_stats(n5, rm5, rs5);
    f[FEAT_ACC_MAG_ROLLING_MEAN_3]  = rm3;
    f[FEAT_ACC_MAG_ROLLING_STD_3]   = rs3;
    f[FEAT_ACC_MAG_ROLLING_MEAN_5]  = rm5;
    f[FEAT_ACC_MAG_ROLLING_STD_5]   = rs5;

    hist.acc_mag_val   = acc_mag;
    hist.gyro_mag_val  = gyro_mag;
    hist.total_mov_val = total_mov;
    hist.acc_mag_norm  = acc_norm;
    has_prev = true;

    // ── 6. Time / circadian ────────────────────────────────────────────────
    double time_min   = (millis() - sleep_start_ms) / 60000.0;
    double time_hours = time_min / 60.0;
    double cycle_rad  = (time_min / 90.0) * 2.0 * PI;

    f[FEAT_TIME_SINCE_START_MIN] = time_min;
    f[FEAT_SLEEP_CYCLE_PHASE]    = fmod(time_min / 90.0, 1.0);
    f[FEAT_SLEEP_CYCLE_SIN]      = sin(cycle_rad);
    f[FEAT_SLEEP_CYCLE_COS]      = cos(cycle_rad);

    // ── 7. Bout / stillness ────────────────────────────────────────────────
    bool moving = (acc_std > 0.05);
    if (moving && !g_in_bout) { g_bout_count++; g_in_bout = true; }
    if (!moving) { g_in_bout = false; g_stillness_epochs++; }

    f[FEAT_IS_MOVEMENT_BOUT]     = g_in_bout ? 1.0 : 0.0;
    f[FEAT_MOVEMENT_BOUT_COUNT]  = g_bout_count;
    f[FEAT_IS_STILLNESS]         = (!moving) ? 1.0 : 0.0;
    f[FEAT_STILLNESS_DURATION]   = g_stillness_epochs;
    f[FEAT_IS_HIGH_VARIABILITY]  = (acc_mag  > p_acc_mean  + 2.0 * p_acc_std)  ? 1.0 : 0.0;
    f[FEAT_IS_VERY_STILL]        = (acc_std < 0.01) ? 1.0 : 0.0;
    f[FEAT_ACC_COEFFICIENT_OF_VARIATION]  = (p_acc_mean  > 1e-9) ? (p_acc_std  / p_acc_mean)  : 0.0;
    f[FEAT_GYRO_COEFFICIENT_OF_VARIATION] = (p_gyro_mean > 1e-9) ? (p_gyro_std / p_gyro_mean) : 0.0;

    // ── 8. Top-20 audio features ───────────────────────────────────────────
    // All features derived from per-window statistics accumulated in aud.*
    uint32_t nw = aud.n_windows;      // windows with FFT data (non-silent)
    uint32_t na = aud.n_all_windows;  // all windows

    if (na > 0) {
        double mean_e  = aud.e_sum    / na;
        double mean_zcr = aud.zcr_sum / na;
        double mean_cr = aud.crest_sum / na;
        double sil_ratio = (double)aud.silence_samples / aud.total_samples;
        double rel_act   = (double)aud.active_windows  / na;
        double act_var   = rel_act * (1.0 - rel_act);   // Bernoulli variance

        double std_e   = (na > 1) ? sqrt(max(0.0, aud.e_sum_sq/na  - mean_e*mean_e))  : 0.0;
        double std_cr  = (na > 1) ? sqrt(max(0.0, aud.crest_sum_sq/na - mean_cr*mean_cr)) : 0.0;

        f[FEAT_ENERGY_ROLLING_MEAN] = mean_e;
        f[FEAT_ENERGY_ROLLING_STD]  = std_e;
        f[FEAT_ENERGY_STD]          = std_e;
        f[FEAT_ENERGY_MAX]          = aud.e_max;
        f[FEAT_ENERGY_PEAK_RATIO]   = (mean_e > 1e-9) ? aud.e_max / mean_e : 0.0;
        f[FEAT_ZCR_MEAN]            = mean_zcr;
        f[FEAT_CREST_FACTOR]        = mean_cr;
        f[FEAT_CREST_VARIABILITY]   = std_cr;
        f[FEAT_SILENCE_RATIO]       = sil_ratio;
        f[FEAT_RELATIVE_ACTIVITY]   = rel_act;
        f[FEAT_ACTIVITY_VARIABILITY] = act_var;
    }

    if (nw > 0) {
        double mean_cent   = aud.cent_sum  / nw;
        double mean_snor   = aud.snor_sum  / nw;
        double mean_breath = aud.breath_sum / nw;
        double mean_rich   = aud.rich_sum  / nw;

        double std_cent   = (nw > 1) ? sqrt(max(0.0, aud.cent_sum_sq/nw  - mean_cent*mean_cent))   : 0.0;
        double std_snor   = (nw > 1) ? sqrt(max(0.0, aud.snor_sum_sq/nw  - mean_snor*mean_snor))   : 0.0;
        double std_breath = (nw > 1) ? sqrt(max(0.0, aud.breath_sum_sq/nw - mean_breath*mean_breath)) : 0.0;

        f[FEAT_SPECTRAL_CENTROID_NORMALIZED] = mean_cent;
        f[FEAT_SPECTRAL_STABILITY]           = 1.0 / (1.0 + std_cent);
        f[FEAT_SPECTRAL_RICHNESS]            = mean_rich;
        f[FEAT_SNORING_POWER]                = mean_snor;
        f[FEAT_SNORING_NORMALIZED]           = mean_snor;  // already normalized by total_power in process_audio_window
        f[FEAT_SNORING_VARIABILITY]          = std_snor;
        // 90th-percentile approximation: mean + 1.28 * std  (valid for approx-normal distribution)
        f[FEAT_SNORING_PERCENTILE]           = mean_snor + 1.28 * std_snor;
        f[FEAT_BREATHING_STABILITY]          = 1.0 / (1.0 + std_breath);
        f[FEAT_BREATHING_RATE_VARIABILITY]   = std_breath;
    }

    // ── 9. Scale and infer ─────────────────────────────────────────────────
    // apply_scaling() defined in scaler_audio.h (generated)
    double scaled[NUM_FEATURES];
    apply_scaling(f, scaled);

    // classifier.predict() defined in xgboost_audio.h (generated by micromlgen)
    // takes float*, returns int class (0=Wake, 1=NREM, 2=REM)
    float sf[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) sf[i] = (float)scaled[i];

    unsigned long t0 = micros();
    int prediction   = classifier.predict(sf);
    unsigned long t_ms = micros() - t0;

    // Update wake/sleep counter for next-epoch patient ratios
    if (prediction == 0) pat.n_wake++;
    else                  pat.n_sleep++;

    // ── 10. Output ──────────────────────────────────────────────────────────
    const char* stage[] = {"WAKE", "NREM", "REM "};
    Serial.print(epoch_num);
    Serial.print(" | ");
    Serial.print((millis() - sleep_start_ms) / 60000.0, 1);
    Serial.print(" min | ");
    Serial.print(total_mov, 4);
    Serial.print(" | ");
    Serial.print(stage[prediction]);
    Serial.print("  | ");
    Serial.print(t_ms / 1000.0, 2);
    Serial.println(" ms");

    // LED feedback
    blink_stage(prediction);
}

// ── LED FEEDBACK ──────────────────────────────────────────────────────────────

// Active-LOW helper: r/g/b = true → LED on.
static inline void set_rgb(bool r, bool g, bool b) {
    digitalWrite(LEDR, r ? LOW : HIGH);
    digitalWrite(LEDG, g ? LOW : HIGH);
    digitalWrite(LEDB, b ? LOW : HIGH);
}

// Show predicted sleep stage as a solid colour for 1.5 s, then off.
//   Wake  → Red
//   NREM  → Green
//   REM   → Blue
void blink_stage(int s) {
    if      (s == 0) set_rgb(true,  false, false);  // Red   — Wake
    else if (s == 1) set_rgb(false, true,  false);  // Green — NREM
    else             set_rgb(false, false, true );  // Blue  — REM
    delay(1500);
    set_rgb(false, false, false);
}

// ── RESET FUNCTIONS ───────────────────────────────────────────────────────────

void reset_imu() {
    for (int i = 0; i < 7; i++) imu_ch[i] = {0, 0, 0, 0, 0};
}

void reset_audio() {
    aud = {0};
}

// ── HELPERS ───────────────────────────────────────────────────────────────────

double chan_mean(ChanStats *s) {
    return (s->n > 0) ? s->sum / s->n : 0.0;
}

double chan_std(ChanStats *s) {
    if (s->n < 2) return 0.0;
    double m = chan_mean(s);
    return sqrt(max(0.0, s->sum_sq / s->n - m * m));
}
