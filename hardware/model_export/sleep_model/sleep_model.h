/*
 * Sleep Stage Classifier - Model Interface
 * Auto-generated - DO NOT EDIT
 *
 * Model: XGBoost (134 features, 50 trees)
 * Classes: 0=Wake, 1=NREM, 2=REM
 *
 * NOTE: This is a simplified interface. The actual XGBoost inference
 * is complex (~5000 lines of C code).
 *
 * For embedded deployment, you have two options:
 * 1. Use a pre-compiled XGBoost library (recommended)
 * 2. Generate full C code using emlearn or micromlgen libraries
 *
 * This header provides the data structures and interface.
 */

#ifndef SLEEP_MODEL_H
#define SLEEP_MODEL_H

#include <stdint.h>
#include <math.h>

#define NUM_FEATURES 134
#define NUM_CLASSES 3
#define NUM_TREES 50

// Feature vector (scaled)
typedef struct {
    double features[NUM_FEATURES];
} FeatureVector;

// Output probabilities
typedef struct {
    double prob_wake;    // class 0
    double prob_nrem;    // class 1
    double prob_rem;     // class 2
} SleepPrediction;

// Main inference function (implement based on your deployment method)
// This is a placeholder - actual implementation depends on your approach
void predict_sleep_stage(const double* features, double* output);

// Helper to get class from probabilities
static inline int get_predicted_class(const double* probs) {
    int pred_class = 0;
    double max_prob = probs[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            pred_class = i;
        }
    }
    return pred_class;
}

// Get class name
static inline const char* get_class_name(int class_id) {
    switch(class_id) {
        case 0: return "Wake";
        case 1: return "NREM";
        case 2: return "REM";
        default: return "Unknown";
    }
}

#endif // SLEEP_MODEL_H

/*
 * DEPLOYMENT OPTIONS:
 *
 * Option 1 (Recommended): Use TensorFlow Lite Micro or ONNX Runtime
 *   - Export XGBoost to ONNX format
 *   - Use onnxruntime-embedded on nRF52840
 *   - Most reliable for production
 *
 * Option 2: Use emlearn library
 *   pip install emlearn
 *   from emlearn.convert import convert_model
 *   c_code = convert_model(model, 'xgboost')
 *
 * Option 3: Manual tree implementation (shown below)
 *   - Simple but verbose (~2500 lines)
 *   - Good for learning/debugging
 */
