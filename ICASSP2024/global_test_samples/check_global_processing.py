import argparse
import os

import soundfile as sf
import numpy as np
import glob


if __name__ == '__main__':
    '''
    This script checks if the enhancing model contains a higher latency than allowed.
    The directory with the files processed with the enhancing model should be provided.
    The files must not have any naming modifications.
    
    Usage example: python3 check_global_processing.py --path path/to/enhanced/data 
    '''
    parser = argparse.ArgumentParser(description="Global processing checker.")
    parser.add_argument("--path", type=str, help="Path to the directory containing the enhanced audio files "
                                                 "(without any naming modification).")
    args = parser.parse_args()

    SAMPLE_RATE = 48_000  # official sample rate
    ALLOWED_LATENCY = 0.020 * SAMPLE_RATE  # total allowed latency: 20ms

    for path in glob.glob(os.path.join(args.path, "*1.wav")):
        sample1, sr1 = sf.read(path)
        sample2, sr2 = sf.read(path.replace("1.wav", "2.wav"))

        assert len(sample1) == len(sample2), f"The files ({os.path.basename(path)}, " \
                                             f"{os.path.basename(path.replace('1.wav', '2.wav'))}) had equal length " \
                                             f"and after processing the length is distinct. Please check!"
        assert sr1 == sr2 == SAMPLE_RATE, "Sample rate is changed. Please check that your algorithm provides 48kHz output!"

        diff = np.abs(sample1 - sample2)
        first_index = np.where(diff != 0.0)[0][0]

        if (len(sample1) // 2) - first_index > ALLOWED_LATENCY:
            print("Your algorithm has a longer latency than 20ms (the limit enforced by organizers)! "
                  "Please check it and solve the issue!")
            exit(1)

    print("Your algorithm's latency is in the allowed interval. Keep going! :)")

