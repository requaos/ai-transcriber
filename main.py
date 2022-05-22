import os
import datetime
import argparse
from huggingsound import SpeechRecognitionModel
from asteroid.separate import file_separate
from asteroid.models import ConvTasNet, BaseModel
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from pydub import AudioSegment
import soundfile as sf
import torch

torch.cuda.is_available()

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
    file_separate(continuous_nnet, main_wav, force_overwrite=True)
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


model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device="cuda")


def walk_for_chapter(path: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith("wav"):
                paths.append(os.path.join(root, file))
    return paths


def transcribe(paths: list[str], sep: str = "\r\n") -> str:
    document = []
    transcriptions = model.transcribe(paths)
    for chunk in transcriptions:
        document.append(chunk.get('transcription'))
    return sep.join(document)


def process_mp3(input_mp3: str):
    t1 = datetime.datetime.now()
    outs = mp3_to_clean_wav(
        input_mp3)
    wav_parts = wav_to_wavs(outs, 38)
    # print(wav_parts)
    transcription = transcribe(wav_parts, sep=' ')

    target_path, _ = os.path.splitext(input_mp3)
    with open(f'{target_path}-transcription.txt', 'w') as f:
        f.write(transcription)
    print(f'{target_path} took {datetime.datetime.now() - t1} to process')


parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
args = parser.parse_args()


if __name__ == '__main__':
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".mp3"):
                process_mp3(os.path.join(root, file))
