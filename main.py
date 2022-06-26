import os
import datetime
import argparse
from asteroid.separate import file_separate
from asteroid.models import ConvTasNet, BaseModel
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from pydub import AudioSegment
from typing import Iterator, Optional
import soundfile as sf
import torch
import warnings
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, HubertForCTC
from deepmultilingualpunctuation import PunctuationModel

torch.cuda.is_available()

punctuation_model = PunctuationModel()
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
hb_model = HubertForCTC.from_pretrained("facebook/hubert-xlarge-ls960-ft").to("cuda")

# device = torch.device("cuda")

# cleaning_model_name = 'hugggof/ConvTasNet_Libri1Mix_enhsignle_16k'

cleaning_model_name = "mhu-coder/ConvTasNet_Libri1Mix_enhsingle"
cleaning_model = ConvTasNet.from_pretrained(cleaning_model_name)
# cleaning_model.load_state_dict(conf.state_dict())
cleaning_model.cuda()
continuous_nnet = LambdaOverlapAdd(
    nnet=cleaning_model,  # function to apply to each segment.
    n_src=1,  # number of sources in the output of nnet
    window_size=16000,  # Size of segmenting window
    hop_size=None,  # segmentation hop size
    window="blackmanharris",  # Type of the window (see scipy.signal.get_window
    reorder_chunks=True,  # Whether to reorder each consecutive segment.
    enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
)
continuous_nnet.cuda()


def mp3_to_clean_wav(input_mp3: str) -> str:
    sound = AudioSegment.from_mp3(input_mp3)
    main_wav = input_mp3.replace(".mp3", ".wav")
    sound.export(main_wav, format="wav")
    # This will output the clean wav file with _est1.wav
    out_file, _ = os.path.splitext(main_wav)
    file_separate(continuous_nnet, main_wav, force_overwrite=True, resample=True)
    os.remove(main_wav)
    os.rename(f'{out_file}_est1.wav', main_wav)
    return main_wav


def wav_to_wavs(input_wav: str, segment_seconds: int = 20, overlap_seconds: int = None) -> list[str]:
    sound = AudioSegment.from_wav(input_wav)
    wav_dir = input_wav.replace(".wav", "")
    if not os.path.exists(wav_dir):
        os.mkdir(wav_dir)
    parts = []
    slices = []
    if overlap_seconds is not None:
        length_s = len(sound)
        i = 0
        while i < length_s:
            end = i+(segment_seconds*1000)
            if end > length_s:
                end = length_s
            extra_end = end + (overlap_seconds*1000)
            if extra_end > length_s:
                extra_end = length_s
            chunk = sound[i:extra_end]
            if len(chunk) > 0:
                slices.append(chunk)
            i = end
    else:
        slices = sound[::(segment_seconds * 1000)]
    for count, sound_slice in enumerate(slices):
        output = os.path.join(wav_dir, f'{count}.wav')
        sound_slice.export(output, format="wav")
        parts.append(output)
    return parts


def load_wav(path: str) -> (torch.Tensor, int):
    mixture, fs = sf.read(path, dtype="float32", always_2d=True)
    # Soundfile returns the mixture as shape (time, channels), and Asteroid expects (batch, channels, time)
    mixture = mixture.transpose()
    mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])
    return torch.from_numpy(mixture), fs


def walk_for_chapter(path: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith("wav"):
                paths.append(os.path.join(root, file))
    return paths


def get_waveforms(paths: list[str], sampling_rate: Optional[int] = 16000) -> list[np.ndarray]:
    waveforms = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in paths:
            waveform, sr = librosa.load(path, sr=sampling_rate)
            waveforms.append(waveform)

    return waveforms


def hb_transcribe(paths: list[str], sep: str = "\r\n") -> str:
    input_values = processor(get_waveforms(paths), padding=True, do_normalize=True, return_tensors="pt").input_values  # Batch size 1
    logits = hb_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    document = []
    for id in predicted_ids:
        document.append(str(processor.decode(id)))
    return sep.join(document).lower()



# from huggingsound import SpeechRecognitionModel
# w2v2_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device="cuda")


# def w2v2_transcribe(paths: list[str], sep: str = "\r\n") -> str:
#     document = []
#     transcriptions = w2v2_model.transcribe(paths)
#     for chunk in transcriptions:
#         document.append(chunk.get('transcription'))
#     return sep.join(document)


# def process_mp3(input_mp3: str):
#     t1 = datetime.datetime.now()
#     outs = mp3_to_clean_wav(
#         input_mp3)
#     wav_parts = wav_to_wavs(outs, 38)
#     # print(wav_parts)
#     transcription = w2v2_transcribe(wav_parts, sep=' ')
#
#     target_path, _ = os.path.splitext(input_mp3)
#     with open(f'{target_path}-transcription.txt', 'w') as f:
#         f.write(transcription)
#     print(f'{target_path} took {datetime.datetime.now() - t1} to process')


def process_long_wav_from_mp3(input_mp3: str, sample_rate: int = 16000, device: str = "cuda"):
    t1 = datetime.datetime.now()
    outs = mp3_to_clean_wav(
        input_mp3)
    audio, _ = librosa.load(outs, sr=sample_rate)
    chunk_duration = 18  # sec
    padding_duration = 2  # sec

    chunk_len = chunk_duration * sample_rate
    input_padding_len = int(padding_duration * sample_rate)
    output_padding_len = hb_model._get_feat_extract_output_lengths(input_padding_len)

    all_preds = []
    for start in range(input_padding_len, len(audio) - input_padding_len, chunk_len):
        chunk = audio[start - input_padding_len:start + chunk_len + input_padding_len]

        input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values
        with torch.no_grad():
            logits = hb_model(input_values.to(device)).logits[0]
            logits = logits[output_padding_len:len(logits) - output_padding_len]

            predicted_ids = torch.argmax(logits, dim=-1)
            all_preds.append(predicted_ids.cuda())

    transcript = processor.decode(torch.cat(all_preds))
    # The whole text is too long, but half should be fine.
    split_transcription = transcript.split()
    halfway = int(len(split_transcription)/2)
    transcription = []
    for transcript_chunk in [" ".join(split_transcription[:halfway]), " ".join(split_transcription[halfway:])]:
        transcription.append(punctuation_model.restore_punctuation(transcript_chunk))
    transcription_with_punctuation = " ".join(transcription)

    target_path, _ = os.path.splitext(input_mp3)
    with open(f'{target_path}-transcription.txt', 'w') as f:
        f.write(str(transcription_with_punctuation).lower())
    print(f'{target_path} took {datetime.datetime.now() - t1} to process')


def process_long_wav(input_wav: str, sample_rate: int = 16000, device: str = "cuda"):
    t1 = datetime.datetime.now()
    audio, _ = librosa.load(input_wav, sr=sample_rate)
    chunk_duration = 18  # sec
    padding_duration = 2  # sec

    chunk_len = chunk_duration * sample_rate
    input_padding_len = int(padding_duration * sample_rate)
    output_padding_len = hb_model._get_feat_extract_output_lengths(input_padding_len)

    all_preds = []
    for start in range(input_padding_len, len(audio) - input_padding_len, chunk_len):
        chunk = audio[start - input_padding_len:start + chunk_len + input_padding_len]

        input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values
        with torch.no_grad():
            logits = hb_model(input_values.to(device)).logits[0]
            logits = logits[output_padding_len:len(logits) - output_padding_len]

            predicted_ids = torch.argmax(logits, dim=-1)
            all_preds.append(predicted_ids.cpu())

    transcription = processor.decode(torch.cat(all_preds))

    target_path, _ = os.path.splitext(input_wav)
    with open(f'{target_path}-transcription.txt', 'w') as f:
        f.write(str(transcription).lower())
    print(f'{target_path} took {datetime.datetime.now() - t1} to process')


parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
args = parser.parse_args()


if __name__ == '__main__':
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".mp3"):
                # process_mp3(os.path.join(root, file))
                process_long_wav_from_mp3(os.path.join(root, file))
