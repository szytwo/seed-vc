import argparse
import os
import time
from math import log10

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio
import uvicorn
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from pydub.effects import normalize
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

from custom.TextProcessor import TextProcessor
from custom.file_utils import logging, delete_old_files_and_folders
from hf_utils import load_custom_model_from_hf
from modules.commons import build_model, load_checkpoint, recursive_munch

# Load model and configuration
# cuda:<index> 来指定特定的 GPU，其中 <index> 是显卡的编号，例如："cuda:0"
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_dir = 'results'


def load_models(f0_condition):
    if not f0_condition:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                                         "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        rmvpe = None
    else:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                                                                         "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        # f0 extractor
        from modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        rmvpe = RMVPE(model_path, is_half=False, device=device)

    config = yaml.safe_load(open(dit_config_path, 'r'))
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, stage='DiT')
    hop_length = config['preprocess_params']['spect_params']['hop_length']
    sr = config['preprocess_params']['sr']

    # Load checkpoints
    model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                     load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location=device))
    campplus_model.eval()
    campplus_model.to(device)

    from modules.bigvgan import bigvgan

    if not f0_condition:
        bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)
    else:
        bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)

    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec", 'pytorch_model.bin', 'config.yml')

    codec_config = yaml.safe_load(open(config_path))
    codec_model_params = recursive_munch(codec_config['model_params'])
    codec_encoder = build_model(codec_model_params, stage="codec")

    ckpt_params = torch.load(ckpt_path, map_location=device)

    for key in codec_encoder:
        codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
    _ = [codec_encoder[key].eval() for key in codec_encoder]
    _ = [codec_encoder[key].to(device) for key in codec_encoder]

    speechtokenizer_set = ('facodec', codec_encoder, None)

    # whisper
    from transformers import AutoFeatureExtractor, WhisperModel

    whisper_name = model_params.speech_tokenizer.whisper_name if hasattr(model_params.speech_tokenizer,
                                                                         'whisper_name') else "openai/whisper-small"
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": 0,
        "fmax": None,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return model, to_mel, bigvgan_model, sr, hop_length, whisper_feature_extractor, whisper_model, campplus_model, rmvpe, speechtokenizer_set


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def webui():
    description = (
        "Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
        "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
        "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
        "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
        "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。")
    inputs = [
        gr.Audio(type="filepath", label="Source Audio / 源音频"),
        gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
        gr.Slider(minimum=1, maximum=200, value=10, step=1, label="Diffusion Steps / 扩散步数",
                  info="10 by default, 50~100 for best quality / 默认为 10，50~100 为最佳质量"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust / 长度调整",
                  info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, label="Inference CFG Rate",
                  info="has subtle influence / 有微小影响"),
        gr.Checkbox(label="Use F0 conditioned model / 启用F0输入", value=False,
                    info="Must set to true for singing voice conversion / 歌声转换时必须勾选"),
        gr.Checkbox(label="Auto F0 adjust / 自动F0调整", value=True,
                    info="Roughly adjust F0 to match target voice. Only works when F0 conditioned model is used. / 粗略调整 F0 以匹配目标音色，仅在勾选 '启用F0输入' 时生效"),
        gr.Slider(label='Pitch shift / 音调变换', minimum=-24, maximum=24, step=1, value=0,
                  info="Pitch shift in semitones, only works when F0 conditioned model is used / 半音数的音高变换，仅在勾选 '启用F0输入' 时生效"),
    ]

    examples = [["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 25, 1.0, 0.7, False, True, 0],
                ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 25, 1.0, 0.7, True, True, 0],
                ["examples/source/Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav",
                 "examples/reference/teio_0.wav", 100, 1.0, 0.7, True, False, 0],
                ["examples/source/TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav",
                 "examples/reference/trump_0.wav", 50, 1.0, 0.7, True, False, -12],
                ]

    outputs = [gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format='mp3'),
               gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format='wav')]

    gr.Interface(fn=voice_conversion,
                 description=description,
                 inputs=inputs,
                 outputs=outputs,
                 title="Seed Voice Conversion",
                 examples=examples,
                 cache_examples=False,
                 ).launch()


# 定义一个函数进行显存清理
def clear_cuda_cache():
    """
    清理PyTorch的显存和系统内存缓存。
    """
    if torch.cuda.is_available():
        logging.info("Clearing GPU memory...")
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # 打印显存日志
        logging.info(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        logging.info(f"[GPU Memory] Max Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")

        # 重置统计信息
        torch.cuda.reset_peak_memory_stats()


# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。

app = FastAPI(docs_url=None)
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  # 允许跨域的headers，可以用来鉴别来源等作用。
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# 使用本地的 Swagger UI 静态资源
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    logging.info("Custom Swagger UI endpoint hit")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/5.9.0/swagger-ui.css",
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.get("/test")
async def test():
    return PlainTextResponse('success')


@app.post('/do')
async def do(source: UploadFile = File(...)
             , target: UploadFile = File(...)
             , diffusion_steps: int = 60
             , length_adjust: float = 1.0
             , inference_cfg_rate: float = 0.7
             , f0_condition: bool = False
             , auto_f0_adjust: bool = False
             , pitch_shift: int = 0):
    # 记录开始时间
    start_time = time.time()

    timestamp = time.time()
    sext = source.filename.split('.')[-1]
    text = target.filename.split('.')[-1]
    source_path = f"{result_dir}/s{timestamp}.{sext}"
    target_path = f"{result_dir}/t{timestamp}.{text}"
    output_path = f"{result_dir}/o{timestamp}.wav"
    with open(source_path, "wb") as f:
        f.write(await source.read())
    with open(target_path, "wb") as f:
        f.write(await target.read())

    voice_conversion_save(
        source=source_path,
        target=target_path,
        output=output_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift
    )

    # 计算耗时
    elapsed = time.time() - start_time
    logging.info(f"生成完成，用时: {elapsed}")

    # return FileResponse(path=f'{output_path}', filename=f'o{timestamp}.wav', media_type='application/octet-stream')
    return PlainTextResponse(f'o{timestamp}.wav')


@app.get('/download')
async def download(name: str):
    return FileResponse(path=f'{result_dir}/{name}', filename=name, media_type='application/octet-stream')


@torch.no_grad()
@torch.inference_mode()
def voice_conversion(
        source,
        target,
        diffusion_steps,
        length_adjust,
        inference_cfg_rate,
        f0_condition,
        auto_f0_adjust,
        pitch_shift
):
    logging.info(
        f"Source: {source}, Target: {target}, Diffusion Steps: {diffusion_steps}, "
        f"Length Adjust: {length_adjust}, Inference CFG Rate: {inference_cfg_rate}, "
        f"F0 Condition: {f0_condition}, Auto F0 Adjust: {auto_f0_adjust}, "
        f"Pitch Shift: {pitch_shift}"
    )

    (
        inference_module,
        mel_fn,
        bigvgan_fn,
        sr_fn,
        hop_length_fn,
        whisper_feature_extractor,
        whisper_model,
        campplus_model,
        rmvpe,
        speechtokenizer_set
    ) = load_models(f0_condition)

    # streaming and chunk processing related params
    max_context_window = sr_fn // hop_length_fn * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length_fn
    # Load audio
    source_audio = librosa.load(source, sr=sr_fn)[0]
    ref_audio = librosa.load(target, sr=sr_fn)[0]

    logging.info(f"source_audio最大幅度1: {source_audio.max()}")
    logging.info(f"ref_audio最大幅度1: {ref_audio.max()}")
    # 归一化音频幅度
    source_audio = source_audio / max(abs(source_audio))  # 归一化到[-1, 1]范围
    ref_audio = ref_audio / max(abs(ref_audio))  # 归一化到[-1, 1]范围

    logging.info(f"source_audio最大幅度2: {source_audio.max()}")
    logging.info(f"ref_audio最大幅度2: {ref_audio.max()}")
    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr_fn * 25]).unsqueeze(0).float().to(device)

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr_fn, 16000)

    logging.info(f"ref_waves_16k最大幅度1: {ref_waves_16k.max()}")
    # 归一化音频幅度
    # ref_waves_16k = ref_waves_16k / ref_waves_16k.abs().max()
    # logging.info(f"ref_waves_16k最大幅度2: {ref_waves_16k.max()}")

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr_fn, 16000)
    logging.info(f"converted_waves_16k最大幅度1: {converted_waves_16k.max()}")

    # 归一化音频幅度
    # converted_waves_16k = converted_waves_16k / converted_waves_16k.abs().max()
    # logging.info(f"converted_waves_16k最大幅度2: {converted_waves_16k.max()}")

    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        alt_inputs = whisper_feature_extractor([converted_waves_16k.squeeze(0).cpu().numpy()],
                                               return_tensors="pt",
                                               return_attention_mask=True,
                                               sampling_rate=16000)
        alt_input_features = whisper_model._mask_input_features(
            alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
        alt_outputs = whisper_model.encoder(
            alt_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        S_alt = alt_outputs.last_hidden_state.to(torch.float32)
        S_alt = S_alt[:, :converted_waves_16k.size(-1) // 320 + 1]
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            alt_inputs = whisper_feature_extractor([chunk.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   sampling_rate=16000)
            alt_input_features = whisper_model._mask_input_features(
                alt_inputs.input_features, attention_mask=alt_inputs.attention_mask).to(device)
            alt_outputs = whisper_model.encoder(
                alt_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            S_alt = alt_outputs.last_hidden_state.to(torch.float32)
            S_alt = S_alt[:, :chunk.size(-1) // 320 + 1]
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr_fn, 16000)
    ori_inputs = whisper_feature_extractor([ori_waves_16k.squeeze(0).cpu().numpy()],
                                           return_tensors="pt",
                                           return_attention_mask=True)
    ori_input_features = whisper_model._mask_input_features(
        ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
    with torch.no_grad():
        ori_outputs = whisper_model.encoder(
            ori_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    S_ori = ori_outputs.last_hidden_state.to(torch.float32)
    S_ori = S_ori[:, :ori_waves_16k.size(-1) // 320 + 1]

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.5)
        F0_alt = rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.5)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    # Length regulation
    (cond,
     _,
     codes,
     commitment_loss,
     codebook_loss) = inference_module.length_regulator(
        S_alt, ylens=target_lengths,
        n_quantizers=3,
        f0=shifted_f0_alt
    )

    (prompt_condition,
     _,
     codes,
     commitment_loss,
     codebook_loss) = inference_module.length_regulator(
        S_ori,
        ylens=target2_lengths,
        n_quantizers=3,
        f0=F0_ori
    )

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    logging.info(f"generate chunk by chunk and stream the output...")

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        # Voice Conversion
        vc_target = inference_module.cfm.inference(cat_condition,
                                                   torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                   mel2, style2, None, diffusion_steps,
                                                   inference_cfg_rate=inference_cfg_rate)
        vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = bigvgan_fn(vc_target)[0]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                    overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    return generated_wave_chunks, sr_fn


def increase_volume_safely(audio, volume_multiplier=1.0):
    if volume_multiplier != 1.0:
        # 1. 归一化音频到最大范围，确保音频峰值不超过 0 dB
        audio = normalize(audio)
        # 2. 根据倍数计算增益的分贝值
        gain_in_db = 20 * log10(volume_multiplier)  # 按倍数计算增益
        # 3. 增加音量
        audio = audio.apply_gain(gain_in_db)

    return audio


def voice_conversion_save(
        source,
        target,
        output,
        diffusion_steps,
        length_adjust,
        inference_cfg_rate,
        f0_condition,
        auto_f0_adjust,
        pitch_shift
):
    try:
        # 检查文件是否存在，若存在则删除
        if os.path.exists(output):
            os.remove(output)

        generated_wave_chunks, sr_fn = voice_conversion(
            source=source,
            target=target,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift
        )

        # 如果已经生成了全部音频，返回拼接后的音频
        if len(generated_wave_chunks) > 0:
            complete_wave = np.concatenate(generated_wave_chunks)
            # 检查并替换 NaN 和 Inf
            complete_wave = np.nan_to_num(complete_wave, nan=0.0, posinf=0.0, neginf=0.0)
            # 归一化到 [-1, 1]
            # complete_wave = complete_wave / np.abs(complete_wave).max()

            if complete_wave.dtype != np.int16:
                complete_wave = (complete_wave * 32768.0).astype(np.int16)

            # 此处可以在生成完音频后，返回拼接的完整音频文件
            # 输出拼接后的音频文件到目标路径
            audio_segment = AudioSegment(
                complete_wave.tobytes(), frame_rate=sr_fn,
                sample_width=complete_wave.dtype.itemsize, channels=1
            )
            # 设置要增加的音量倍数
            volume_multiplier = 1.0  # 音量倍数
            # 安全地增加音量
            audio_with_increased_volume = increase_volume_safely(audio_segment, volume_multiplier)
            audio_with_increased_volume.export(output, format="wav")

        logging.info(f"write: {output}")
    except Exception as e:
        TextProcessor.log_error(e)
        errmsg = f"音频生成失败，错误信息：{str(e)}"
        logging.error(errmsg)
    finally:
        # 删除过期文件
        delete_old_files_and_folders(result_dir, 1)
        clear_cuda_cache()


def get_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./examples/source/source_s1.wav")
    parser.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
    parser.add_argument("--output", type=str, default="./results/audio.wav")
    parser.add_argument("--diffusion-steps", type=int, default=60)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=bool, default=False)
    parser.add_argument("--auto-f0-adjust", type=bool, default=False)
    parser.add_argument("--pitch_shift", type=int, default=0)
    parser.add_argument("--api", type=bool, default=False)
    parser.add_argument("--port", type=int, default=7869)
    # parser.add_argument("--webui",type=bool,default=False)

    return parser.parse_args()  # ✅ 每次调用都解析参数


if __name__ == "__main__":
    try:
        argsMain = get_main_args()

        if argsMain.api:
            uvicorn.run(app, host="0.0.0.0", port=argsMain.port)
        else:
            voice_conversion_save(
                source=argsMain.source,
                target=argsMain.target,
                output=argsMain.output,
                diffusion_steps=argsMain.diffusion_steps,
                length_adjust=argsMain.length_adjust,
                inference_cfg_rate=argsMain.inference_cfg_rate,
                f0_condition=argsMain.f0_condition,
                auto_f0_adjust=argsMain.auto_f0_adjust,
                pitch_shift=argsMain.pitch_shift
            )

    except Exception as ex:
        logging.error(ex)
        exit(0)
