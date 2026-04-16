"""
Synchronize PSG and PillowClip data for sleep stage prediction

This script:
1. Reads PSG data (EDF files and EpochByEpoch.csv) to get ground truth sleep stages
2. Reads PillowClip IMU and audio data
3. Calculates time offset between devices for each patient
4. Creates synchronized dataset with:
   - 30-second epochs aligned with PSG
   - PillowClip features for each epoch
   - Sleep stage labels from PSG
"""

import os
import csv
import json
import datetime
import numpy as np
from pathlib import Path

class DataSynchronizer:
    def __init__(self, psg_dir, pillowclip_dir, output_dir):
        self.psg_dir = Path(psg_dir)
        self.pillowclip_dir = Path(pillowclip_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.sync_report = []

    def read_edf_header(self, edf_path):
        """Read EDF file header to get recording start time and duration"""
        with open(edf_path, 'rb') as f:
            header = f.read(256)

            start_date = header[168:176].decode('ascii', errors='ignore').strip()
            start_time = header[176:184].decode('ascii', errors='ignore').strip()
            num_records = int(header[236:244].decode('ascii', errors='ignore').strip())
            duration_per_record = float(header[244:252].decode('ascii', errors='ignore').strip())

            # Parse date and time
            day, month, year = start_date.split('.')
            hour, minute, second = start_time.split('.')
            year = '20' + year if int(year) < 50 else '19' + year

            start_datetime = datetime.datetime.strptime(
                f"{year}{month}{day} {hour}:{minute}:{second}",
                "%Y%m%d %H:%M:%S"
            )

            duration_seconds = num_records * duration_per_record

            return {
                'start_time': start_datetime,
                'duration_seconds': duration_seconds,
                'num_records': num_records
            }

    def read_pillowclip_timing(self, pillowclip_csv_path):
        """Extract timing info from PillowClip CSV file"""
        filename = os.path.basename(pillowclip_csv_path)
        parts = filename.split('_')

        # Filename format: PXXX_YYYYMMDD_HHMMSS_S01_pillow.csv
        date_str = parts[1]
        time_str = parts[2]

        device_start = datetime.datetime.strptime(
            f"{date_str} {time_str}",
            "%Y%m%d %H%M%S"
        )

        # Read first and last ts_ms
        with open(pillowclip_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
            first_ts_ms = int(first_row['ts_ms'])

            last_ts_ms = first_ts_ms
            for row in reader:
                last_ts_ms = int(row['ts_ms'])

        first_sample_time = device_start + datetime.timedelta(milliseconds=first_ts_ms)
        last_sample_time = device_start + datetime.timedelta(milliseconds=last_ts_ms)

        return {
            'device_start': device_start,
            'first_ts_ms': first_ts_ms,
            'last_ts_ms': last_ts_ms,
            'first_sample_time': first_sample_time,
            'last_sample_time': last_sample_time
        }

    def calculate_overlap(self, psg_info, pillowclip_info):
        """Calculate overlap period between PSG and PillowClip recordings"""
        psg_start = psg_info['start_time']
        psg_end = psg_start + datetime.timedelta(seconds=psg_info['duration_seconds'])

        pc_start = pillowclip_info['first_sample_time']
        pc_end = pillowclip_info['last_sample_time']

        # Check if there's overlap
        if pc_start > psg_end or pc_end < psg_start:
            return None

        overlap_start = max(psg_start, pc_start)
        overlap_end = min(psg_end, pc_end)
        overlap_duration = (overlap_end - overlap_start).total_seconds()

        # Calculate offset (PillowClip relative to PSG)
        offset_seconds = (pc_start - psg_start).total_seconds()

        return {
            'overlap_start': overlap_start,
            'overlap_end': overlap_end,
            'overlap_duration_seconds': overlap_duration,
            'overlap_epochs': int(overlap_duration / 30),  # 30-second epochs
            'offset_seconds': offset_seconds,
            'psg_start': psg_start,
            'psg_end': psg_end,
            'pc_start': pc_start,
            'pc_end': pc_end
        }

    def read_epoch_sleep_stages(self, epoch_csv_path):
        """Read sleep stage labels from EpochByEpoch.csv"""
        epochs = []
        with open(epoch_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append({
                    'epoch_num': int(row['Epoch #']),
                    'elapsed_time_sec': int(row['ElapsedTime(sec)']),
                    'stage_final': int(row['Stage final']) if row['Stage final'] else None,
                    'clock_time': row['ClockTime']
                })
        return epochs

    def align_pillowclip_to_epochs(self, pillowclip_csv_path, overlap_info, pillowclip_timing):
        """
        Align PillowClip samples to PSG 30-second epochs
        Returns dict of epoch_num -> PillowClip samples for that epoch
        """
        # Calculate which PillowClip samples correspond to each PSG epoch
        psg_start = overlap_info['psg_start']
        pc_first_ts_ms = pillowclip_timing['first_ts_ms']
        offset_seconds = overlap_info['offset_seconds']

        # Read all PillowClip data
        pillowclip_data = []
        with open(pillowclip_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_ms = int(row['ts_ms'])
                # Convert ts_ms to seconds relative to PSG start
                psg_relative_time = (ts_ms / 1000.0) - (pc_first_ts_ms / 1000.0) + offset_seconds

                pillowclip_data.append({
                    'psg_time': psg_relative_time,
                    'ts_ms': ts_ms,
                    'ax': float(row['ax']),
                    'ay': float(row['ay']),
                    'az': float(row['az']),
                    'gx': float(row['gx']),
                    'gy': float(row['gy']),
                    'gz': float(row['gz']),
                    'tempC': float(row['tempC']),
                    'mic_rms': float(row['mic_rms']),
                    'zcr': int(row['zcr'])
                })

        # Group samples by epoch (30-second windows)
        epoch_data = {}
        for sample in pillowclip_data:
            # Calculate which epoch this sample belongs to (0-indexed)
            epoch_idx = int(sample['psg_time'] / 30)

            if epoch_idx not in epoch_data:
                epoch_data[epoch_idx] = []

            epoch_data[epoch_idx].append(sample)

        return epoch_data

    def create_epoch_features(self, epoch_samples):
        """
        Create feature vector from PillowClip samples within an epoch
        Features: mean, std, min, max for each sensor
        """
        if not epoch_samples:
            return None

        features = {}
        sensors = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'tempC', 'mic_rms', 'zcr']

        for sensor in sensors:
            values = [s[sensor] for s in epoch_samples]
            features[f'{sensor}_mean'] = np.mean(values)
            features[f'{sensor}_std'] = np.std(values)
            features[f'{sensor}_min'] = np.min(values)
            features[f'{sensor}_max'] = np.max(values)

        features['num_samples'] = len(epoch_samples)

        return features

    def process_patient(self, patient_id):
        """Process a single patient"""
        print(f"\n{'='*60}")
        print(f"Processing {patient_id}")
        print(f"{'='*60}")

        # Find PSG files
        psg_patient_dir = self.psg_dir / f"{patient_id},{patient_id}"
        if not psg_patient_dir.exists():
            print(f"  ERROR: PSG directory not found")
            return None

        edf_files = list(psg_patient_dir.glob("*Export.edf"))
        epoch_files = list(psg_patient_dir.glob("*EpochByEpoch.csv"))

        if not edf_files or not epoch_files:
            print(f"  ERROR: Missing EDF or Epoch files")
            return None

        edf_path = edf_files[0]
        epoch_path = epoch_files[0]

        # Find PillowClip files
        pc_patient_dir = self.pillowclip_dir / patient_id
        if not pc_patient_dir.exists():
            print(f"  ERROR: PillowClip directory not found")
            return None

        pc_files = list(pc_patient_dir.glob("*pillow.csv"))
        if not pc_files:
            print(f"  ERROR: PillowClip CSV not found")
            return None

        pc_path = pc_files[0]

        # Read timing information
        print(f"  Reading PSG timing from EDF...")
        psg_info = self.read_edf_header(edf_path)

        print(f"  Reading PillowClip timing...")
        pc_info = self.read_pillowclip_timing(pc_path)

        # Calculate overlap
        print(f"  Calculating overlap...")
        overlap_info = self.calculate_overlap(psg_info, pc_info)

        if overlap_info is None:
            print(f"  ERROR: No overlap between PSG and PillowClip")
            return None

        print(f"  PSG:        {psg_info['start_time'].strftime('%H:%M:%S')} - "
              f"{(psg_info['start_time'] + datetime.timedelta(seconds=psg_info['duration_seconds'])).strftime('%H:%M:%S')} "
              f"({psg_info['duration_seconds']/60:.1f} min)")
        print(f"  PillowClip: {pc_info['first_sample_time'].strftime('%H:%M:%S')} - "
              f"{pc_info['last_sample_time'].strftime('%H:%M:%S')} "
              f"({(pc_info['last_ts_ms'] - pc_info['first_ts_ms'])/60000:.1f} min)")
        print(f"  Overlap:    {overlap_info['overlap_duration_seconds']/60:.1f} minutes "
              f"({overlap_info['overlap_epochs']} epochs)")
        print(f"  Offset:     {overlap_info['offset_seconds']:.1f} seconds "
              f"({overlap_info['offset_seconds']/60:.1f} minutes)")

        # Read sleep stages
        print(f"  Reading sleep stage labels...")
        epochs = self.read_epoch_sleep_stages(epoch_path)

        # Align PillowClip data to epochs
        print(f"  Aligning PillowClip data to epochs...")
        epoch_pc_data = self.align_pillowclip_to_epochs(pc_path, overlap_info, pc_info)

        # Create synchronized dataset
        print(f"  Creating synchronized dataset...")
        synchronized_data = []

        for epoch in epochs:
            epoch_num = epoch['epoch_num']

            # Only include epochs that have overlap
            if epoch_num in epoch_pc_data and len(epoch_pc_data[epoch_num]) > 0:
                features = self.create_epoch_features(epoch_pc_data[epoch_num])

                if features is not None:
                    synchronized_data.append({
                        'patient_id': patient_id,
                        'epoch_num': epoch_num,
                        'elapsed_time_sec': epoch['elapsed_time_sec'],
                        'sleep_stage': epoch['stage_final'],
                        **features
                    })

        # Save synchronized data
        output_file = self.output_dir / f"{patient_id}_synchronized.csv"
        if synchronized_data:
            keys = synchronized_data[0].keys()
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(synchronized_data)

            print(f"  Saved {len(synchronized_data)} synchronized epochs to {output_file.name}")

        # Add to sync report
        sync_info = {
            'patient_id': patient_id,
            'psg_start': psg_info['start_time'].isoformat(),
            'psg_duration_min': psg_info['duration_seconds'] / 60,
            'pc_start': pc_info['first_sample_time'].isoformat(),
            'pc_duration_min': (pc_info['last_ts_ms'] - pc_info['first_ts_ms']) / 60000,
            'overlap_min': overlap_info['overlap_duration_seconds'] / 60,
            'overlap_epochs': overlap_info['overlap_epochs'],
            'offset_seconds': overlap_info['offset_seconds'],
            'synchronized_epochs': len(synchronized_data)
        }

        self.sync_report.append(sync_info)

        return sync_info

    def process_all_patients(self):
        """Process all patients with both PSG and PillowClip data"""
        # Get list of patients
        patients = []
        for psg_dir in self.psg_dir.iterdir():
            if psg_dir.is_dir() and ',' in psg_dir.name:
                patient_id = psg_dir.name.split(',')[0]
                # Check if PillowClip data exists
                if (self.pillowclip_dir / patient_id).exists():
                    patients.append(patient_id)

        patients.sort()

        print(f"Found {len(patients)} patients with both PSG and PillowClip data")
        print(f"Patients: {', '.join(patients)}")

        # Process each patient
        for patient_id in patients:
            try:
                self.process_patient(patient_id)
            except Exception as e:
                print(f"  ERROR processing {patient_id}: {e}")
                import traceback
                traceback.print_exc()

        # Save synchronization report
        report_file = self.output_dir / "synchronization_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.sync_report, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Synchronization complete!")
        print(f"Report saved to: {report_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Paths
    psg_dir = "/Users/syed/Documents/University/Y3S2/FYP/Fresh_Start/PSG_Data"
    pillowclip_dir = "/Users/syed/Documents/University/Y3S2/FYP/Fresh_Start/PillowClip_Data"
    output_dir = "/Users/syed/Documents/University/Y3S2/FYP/Fresh_Start/Synchronized_Data"

    # Create synchronizer
    synchronizer = DataSynchronizer(psg_dir, pillowclip_dir, output_dir)

    # Process all patients
    synchronizer.process_all_patients()
