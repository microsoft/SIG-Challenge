from pedalboard import Pedalboard, PitchShift, Gain, Reverb, Convolution, HighpassFilter, \
    LowpassFilter, Bitcrush, NoiseGate, Gain, Clipping, Resample, Phaser, Distortion, Invert, \
        Delay, GSMFullRateCompressor, LowShelfFilter, HighShelfFilter, PeakFilter, IIRFilter
import random
import numpy as np
import os
import soundfile as sf
import tempfile
from shutil import rmtree
from typing import List, Dict, Tuple, Any, Optional


class RIRFifo:
    """
    Simple FIFO-or-tempfile wrapper for writing RIRs to a file-like object and passing
    them to Pedalboard, because it insists on reading from a file specified by name.

    Uses a FIFO on systems that support it, otherwise a temporary file.
    """
    def __init__(self, audio_data: np.ndarray, sample_rate_hz: int):
        self.audio_data = audio_data
        self.sample_rate_hz = sample_rate_hz
        self.fifo_name = None
        self.temp_dir = None

    def __enter__(self):
        return self.__write_to_fifo(self.audio_data, self.sample_rate_hz)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_fifo()

    def __write_to_fifo(self, audio_data: np.ndarray, sample_rate_hz: int) -> str:
        if hasattr(os, 'mkfifo'):
            self.temp_dir = tempfile.mkdtemp()
            self.fifo_name = os.path.join(self.temp_dir, 'audio.fifo.wav')
            os.mkfifo(self.fifo_name)
            sf.write(self.fifo_name, audio_data, sample_rate_hz)
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            self.fifo_name = temp_file.name
            sf.write(self.fifo_name, audio_data, sample_rate_hz)
        return self.fifo_name

    def __delete_fifo(self):
        if hasattr(os, 'mkfifo') and self.temp_dir is not None:
            rmtree(self.temp_dir)
        else:
            os.remove(self.fifo_name)


class Distorter:
    def __init__(self, distortion_sampling_params: Dict[str, Any] = None, seed_mode: str = "random"):
        """
        Create a pedalboard-based distortion pipeline.
        
        Distortion sampling parameters are documented by example in load_default_config().
        
        Seed mode:
            * "random": Randomly generate parameters without any particular seed 
              beyond the global one
            * "fixed": Seed using a value derived from the input audio data, i.e. keep 
              the same parameters for the same input (but note that floating point 
              precision may cause deviation)
            * "fixed_iter": Seed using a value derived from the input audio data and 
              the iteration number, i.e. keep the same parameters for the same input 
            and iteration (but note that floating point precision may cause deviation)
        """
        self.distortion_sampling_params = distortion_sampling_params
        if self.distortion_sampling_params is None:
            self.load_default_config()
        self.seed_mode = seed_mode
            
    def load_default_config(self, impulse_responses: List[str] = []):
        """
        Load a default set of parameters.
        This is mostly included for testing and documentation purposes. In production,
        provide a configuration to the constructor that is loaded from some external
        config file.
        
        Set likelihood for any given distortion to 0 to disable it or to 1 to make it
        always active.
        """
        ir_likelihood = 0.5
        if len(impulse_responses) == 0:
            ir_likelihood = 0.0
        self.distortion_sampling_params = {
            # Speaker change augmentations
            "speaker_gain_delta_db": {"range": (-10, 10), "likelihood": 0.2},

            # Room and mic augmentations
            "allow_rir_and_generated": True,
            "room_freeverb_room_size": {"range": (0.01, 0.3), "likelihood": 0.1},
            "room_mic_impulse_response_file": {"range": impulse_responses, "likelihood": ir_likelihood},
            "room_bad_mic_response": {"likelihood": 0.5},
            "room_bad_mic_params": {
                "low_shelf_hz": (50, 700),
                "low_shelf_db": (-12, 12),
                "high_shelf_hz": (4500, 7500),
                "high_shelf_db": (-12, 12),
                "peak_count": (0, 7),
                "peak_hz": (600, 7400),
                "peak_db": (-6, 6)
            },

            # ADC augmentations
            "dac_highpass_cutoff_hz": {"range": (4, 100), "likelihood": 0.7},
            "dac_lowpass_cutoff_hz": {"range": (5000, 7900), "likelihood": 0.7},
            "dac_bitdepth": {"range": (8, 24), "likelihood": 0.1},

            # AGC augmentations
            "agc_expand_ratio": {"range": (1.4, 2.1), "likelihood": 0.1},
            "agc_gain_db": {"range": (5, 20), "likelihood": 0.1},

            # Preprocessing augmentations. Note: Resampling is only done once - either here, 
            # or by the GSM processor, which always resamples to 8kHz and takes precedence.
            "prepro_hardlimit_enabled": {"likelihood": 0.25},
            "prepro_postclip_gain_db": {"range": (-5, 0), "likelihood": 0.25},
            "prepro_sample_rate_hz": {"range": (3000, 32000), "likelihood": 0.4},
            "prepro_or_gsm_resample_quality": {"range": ["ZeroOrderHold", "Linear", "CatmullRom", "WindowedSinc"], "likelihood": 1.0}, # Best to always set this

            # Extra "weird distortions" augmentations
            "extra_phaser_rate_hz": {"range": (100, 500), "likelihood": 0.02},
            "extra_tanh_distortion_drive_db": {"range": (2, 10), "likelihood": 0.01},

            # Microaugmentations
            "microaug_shift_ms": {"range": (0.0, 2.0 / 16000.0), "likelihood": 0.0},

            # Transmission augmentation
            "gsm_transmission_enabled": {"likelihood": 0.25}
        }

    def generate_random_params(self, use_provided_rir: bool = False, seed: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate random parameters for a pedalboard based audio distortion pipeline
        """
        if not seed is None:
            random.seed(seed)
        new_params = {}
        distortion_sampling_params = self.distortion_sampling_params
        
        # Pick random values
        for param in distortion_sampling_params:
            # Bad mic params get special treatment below
            if param in ("room_bad_mic_params", "allow_rir_and_generated"):
                continue
            
            # Special case out not having impulse responses
            if param == "room_mic_impulse_response_file" and len(distortion_sampling_params[param]["range"]) == 0:
                if use_provided_rir and (random.random() < float(distortion_sampling_params[param]["likelihood"])):
                    new_params[param] = "PROVIDED_RIR"
                else:
                    continue
            
            # Figure out type
            param_type = bool
            if "range" in distortion_sampling_params[param]:
                param_type = type(distortion_sampling_params[param]["range"][0])
            
            # Generate a random number and if it's less than the likelihood, set the parameter
            if random.random() < float(distortion_sampling_params[param]["likelihood"]):
                if param_type == bool:  # Special case for booleans - if likelihood test passes, always True
                    new_params[param] = True
                elif param_type == str:  # For string types, the range should be a list of possible strings
                    new_params[param] = random.choice(distortion_sampling_params[param]["range"])
                else:  # For other numerical types
                    range_low = float(distortion_sampling_params[param]["range"][0])
                    range_high = float(distortion_sampling_params[param]["range"][1])
                    new_params[param] = random.uniform(range_low, range_high)
        
            # If the module has a wet/dry mix range specified, pick that also
            if "range_mix" in distortion_sampling_params[param]:
                range_low = float(distortion_sampling_params[param]["range_mix"][0])
                range_high = float(distortion_sampling_params[param]["range_mix"][1])
                new_params[param + "_mix"] = random.uniform(range_low, range_high)

        # Some special conditioning for certain parameters
        # GSM transmission disables resample
        if new_params.get("gsm_transmission_enabled") is not None and new_params["gsm_transmission_enabled"] == True:
            if new_params.get("prepro_sample_rate_hz") is not None:
                del new_params["prepro_sample_rate_hz"]

        # AGC expansion on enables gain and increases gain a bit
        if new_params.get("agc_expand_ratio") is not None:
            range_low = float(distortion_sampling_params["agc_gain_db"]["range"][0])
            range_high = float(distortion_sampling_params["agc_gain_db"]["range"][1])
            new_params["agc_gain_db"] = random.uniform(range_low, range_high) + 6.0

        # Bad mic response generation can if desired disable RIR, and requires addition parameter picking
        if new_params.get("room_bad_mic_response") is not None and new_params["room_bad_mic_response"] == True:
            if "allow_rir_and_generated" in distortion_sampling_params:
                if not distortion_sampling_params["allow_rir_and_generated"]:
                    if new_params.get("room_mic_impulse_response_file") is not None:
                        del new_params["room_mic_impulse_response_file"]
            bad_mic_params = distortion_sampling_params["room_bad_mic_params"]
            peak_count = random.randint(*bad_mic_params["peak_count"])
            for mic_param in bad_mic_params:
                bad_mic_params[mic_param] = tuple([float(x) for x in bad_mic_params[mic_param]])
            new_params["room_bad_mic_params"] = {
                "low_shelf_hz": random.uniform(*bad_mic_params["low_shelf_hz"]),
                "low_shelf_db": random.uniform(*bad_mic_params["low_shelf_db"]),
                "high_shelf_hz": random.uniform(*bad_mic_params["high_shelf_hz"]),
                "high_shelf_db": random.uniform(*bad_mic_params["high_shelf_db"]),
                "peak_hz": [random.uniform(*bad_mic_params["peak_hz"]) for _ in range(peak_count)],
                "peak_db": [random.uniform(*bad_mic_params["peak_db"]) for _ in range(peak_count)],
            }
        return new_params

    # @lru_cache(maxsize=500)
    def get_impulse_response_conv(self, filename: str) -> Convolution:
        """
        Impulse response loader. Possibly with caching (to be evaluated)
        """
        return Convolution(impulse_response_filename = filename)

    def generate_bad_mic_response(self, bad_mic_params: Dict[str, Any]) -> List[IIRFilter]:
        """
        Generate a filter with a very not flat response by combining a low shelf, high shelf, and a few peaks/notches
        """
        response_modules = [
            LowShelfFilter(cutoff_frequency_hz = bad_mic_params["low_shelf_hz"], gain_db = bad_mic_params["low_shelf_db"]),
            HighShelfFilter(cutoff_frequency_hz = bad_mic_params["high_shelf_hz"], gain_db = bad_mic_params["high_shelf_db"]),
        ]
        for cutoff, gain in zip(bad_mic_params["peak_hz"], bad_mic_params["peak_db"]):
            response_modules.append(PeakFilter(cutoff_frequency_hz = cutoff, gain_db = gain, q = 5.0))
        return response_modules

    def pedalboard_process(self, audio: np.ndarray, sample_rate_hz: int, pedalboard_params: Dict[str, Any], provided_rir: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Set up a pedalboard processing chain and process audio
        """
        # List for modules
        module_list = []
        
        # Speaker augmentations
        if pedalboard_params.get("speaker_gain_delta_db") is not None:
            speaker_gain = Gain(gain_db = pedalboard_params["speaker_gain_delta_db"])
            module_list.append(speaker_gain)
        
        # Room + Mic
        if pedalboard_params.get("room_freeverb_room_size") is not None:
            wet_level = 0.33  # Default value if not randomized
            dry_level = 0.4  # Default value if not randomized
            if "room_freeverb_room_size_mix" in pedalboard_params:
                wet_level = pedalboard_params["room_freeverb_room_size_mix"]                
                dry_level = 1.0 - pedalboard_params["room_freeverb_room_size_mix"]
                wet_level = wet_level / 0.5 * 0.33
                dry_level = dry_level / 0.5 * 0.4
            room_reverb = Reverb(room_size = pedalboard_params["room_freeverb_room_size"], wet_level = wet_level, dry_level = dry_level)
            module_list.append(room_reverb)
        if pedalboard_params.get("room_bad_mic_response") is not None and pedalboard_params["room_bad_mic_response"] == True:
            module_list.extend(self.generate_bad_mic_response(pedalboard_params["room_bad_mic_params"]))
        if pedalboard_params.get("room_mic_impulse_response_file") is not None:
            if pedalboard_params["room_mic_impulse_response_file"] == "PROVIDED_RIR":
                with RIRFifo(provided_rir["data"], provided_rir["sample_rate"]) as rir_fifo_name:
                    room_mic_impulse = Convolution(impulse_response_filename = rir_fifo_name)
            else:
                room_mic_impulse = self.get_impulse_response_conv(pedalboard_params["room_mic_impulse_response_file"])
            module_list.append(room_mic_impulse)
        
        # Analog-digital-conversion
        if pedalboard_params.get("dac_highpass_cutoff_hz") is not None:
            dac_highpass = HighpassFilter(cutoff_frequency_hz = pedalboard_params["dac_highpass_cutoff_hz"])
            module_list.append(dac_highpass)
        if pedalboard_params.get("dac_lowpass_cutoff_hz") is not None:
            dac_lowpass = LowpassFilter(cutoff_frequency_hz = pedalboard_params["dac_lowpass_cutoff_hz"])
            module_list.append(dac_lowpass)
        if pedalboard_params.get("dac_bitdepth") is not None:
            dac_bitdepth = Bitcrush(bit_depth = pedalboard_params["dac_bitdepth"])
            module_list.append(dac_bitdepth)
        
        # "Anti-AGC"
        if pedalboard_params.get("agc_expand_ratio") is not None:
            threshold_db = 0.01
            if "agc_expand_ratio_mix" in pedalboard_params:
                threshold_db = pedalboard_params["agc_expand_ratio_mix"]
            agc_expander = NoiseGate(threshold_db = threshold_db, ratio = pedalboard_params["agc_expand_ratio"])
            module_list.append(agc_expander)    
        if pedalboard_params.get("agc_gain_db") is not None:
            agc_gain = Gain(gain_db = pedalboard_params["agc_gain_db"])
            module_list.append(agc_gain)
        
        # Pre-transmission processing
        if pedalboard_params.get("prepro_hardlimit_enabled") is not None and pedalboard_params["prepro_hardlimit_enabled"] == True:
            prepro_hardlimit = Clipping()
            module_list.append(prepro_hardlimit)
        if pedalboard_params.get("prepro_postclip_gain_db") is not None:
            prepro_gain = Gain(gain_db = pedalboard_params["prepro_postclip_gain_db"])
            module_list.append(prepro_gain)    
        if pedalboard_params.get("prepro_sample_rate_hz") is not None:
            if pedalboard_params.get("gsm_transmission_enabled") is not None and pedalboard_params["gsm_transmission_enabled"] == True:
                assert False, "Choose either resampling on its own, or as part of GSM transmission"
            if pedalboard_params.get("prepro_or_gsm_resample_quality") is None:
                assert False, "Missing prepro_or_gsm_resample_quality"
            prepro_resample = Resample(target_sample_rate = pedalboard_params["prepro_sample_rate_hz"], quality = getattr(Resample.Quality, pedalboard_params["prepro_or_gsm_resample_quality"])) 
            module_list.append(prepro_resample)
        
        # "Other weird distortions"
        if pedalboard_params.get("extra_phaser_rate_hz") is not None: 
            mix = pedalboard_params.get("extra_phaser_rate_hz_mix", 1.0)
            extra_phasedistortion = Phaser(rate_hz = pedalboard_params["extra_phaser_rate_hz"], mix = mix)
            module_list.append(extra_phasedistortion)
        if pedalboard_params.get("extra_tanh_distortion_drive_db") is not None: 
            extra_nonlinear = Distortion(drive_db = pedalboard_params["extra_tanh_distortion_drive_db"])
            module_list.append(extra_nonlinear)
        
        # Micro-augmentations
        if pedalboard_params.get("microaug_shift_ms") is not None:
            mix = pedalboard_params.get("microaug_shift_mix", 1.0)
            microaug_shift = Delay(delay_seconds = pedalboard_params["microaug_shift_ms"] / 1000.0, mix = mix)
            module_list.append(microaug_shift)
        
        # Transmission
        if pedalboard_params.get("gsm_transmission_enabled") is not None and pedalboard_params["gsm_transmission_enabled"] == True:
            if pedalboard_params.get("prepro_or_gsm_resample_quality") is None:
                assert False, "Missing prepro_or_gsm_resample_quality"
            transmission_gsm = GSMFullRateCompressor(quality = getattr(Resample.Quality, pedalboard_params["prepro_or_gsm_resample_quality"]))
            module_list.append(transmission_gsm)

        # Final hardlimit is always enabled
        final_hardlimit = Clipping(threshold_db=0.0)
        module_list.append(final_hardlimit)
        
        # Process and return
        board = Pedalboard(module_list)
        return board(audio, sample_rate_hz)
    
    def apply_distortions(self, 
                          audio: np.ndarray,
                          sample_rate_hz: int,
                          params: Optional[Dict[str, Any]] = None, 
                          provided_rir: Optional[Dict[str, Any]] = None, 
                          seed_iter: Optional[int] = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply distortions to an audio file, potentially picking random parameters according to config.
        
        params, if provided, overrides random parameter selection and just uses the exact parameters provided.
        
        provided_rir can be used to provide a RIR to use in case one is needed (i.e. when the param generator 
        chooses to use one) instead of the module randomly choosing one from a list. This is not supported
        in fixed modes, since RIRs are not cached. It must be a dictionary with keys "data" (rir as an ndarray) 
        and "sample_rate" (in hz).
        
        Returns the processed audio and the parameters used.
        """
        # Random set of parameters according to config, if none provided
        if params is None:
            if self.seed_mode == "random":
                params = self.generate_random_params(use_provided_rir=(provided_rir is not None))
            elif self.seed_mode.startswith("fixed"):
                seed = np.sum(np.abs(audio))
                if self.seed_mode == "fixed_iter":
                    if seed_iter is None:
                        assert False, "seed_iter is required for fixed_iter mode"
                    seed = seed + seed_iter
                params = self.generate_random_params(use_provided_rir=(provided_rir is not None), seed=seed)
            else:
                assert False, "Unknown seed mode"
            
        # Process audio and return
        return self.pedalboard_process(audio, sample_rate_hz, params, provided_rir), params
    