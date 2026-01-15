"""
数字人视频生成器 V4
功能：根据 JSON 配置，在教学视频上叠加动画数字人

核心特点：
- 直接从视频提取音频（ffmpeg）
- 与 V1 保持一致的口型和眨眼逻辑
- 支持 lecture（5张素材）和 coding（3张素材）两种场景
- 支持静默片段删除

处理流程：
1. 读取 JSON 配置，获取片段信息
2. 从 MP4 提取音频（VAD 用 16kHz，音量用原始采样率）
3. VAD 人声检测 → speech_mask
4. 音量计算 → volume_per_frame
5. 眨眼生成 → blink_mask
6. 长时间静默检测 → long_silence_mask
7. OpenCV 逐帧处理，叠加数字人
8. 删除静默片段，合并音频
9. 输出最终视频
"""

import os
import json
import glob
import random
import subprocess
import numpy as np
import cv2
import torch
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 【路径配置】
# ============================================================

# 获取脚本所在目录的父目录作为项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VIDEO_FOLDER = os.path.join(PROJECT_ROOT, "source", "lecture")
JSON_FOLDER = os.path.join(PROJECT_ROOT, "resourceProject", "geminiGen")
AVATAR_FOLDER = os.path.join(PROJECT_ROOT, "source", "characterStyle")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "source", "output")

# ============================================================
# 【效果参数】
# ============================================================

# 视频帧率（会从视频中读取实际值）
DEFAULT_FPS = 25

# VAD 人声检测阈值 (0-1)
# - 值越大，越不容易被噪音触发
# - 值越小，越容易检测到人声
VAD_THRESHOLD = 0.25

# 人声前后缓冲 (毫秒)
PADDING_MS = 250

# 嘴巴张合模式: "volume", "fixed", "hybrid"
# - "volume": 根据音量决定张嘴/闭嘴
# - "fixed": 固定频率交替张合
# - "hybrid": 混合模式（推荐）- 音量超过阈值时用固定频率张合
MOUTH_MODE = "hybrid"

# 固定频率模式的参数
MOUTH_OPEN_FRAMES = 1   # 张嘴持续帧数
MOUTH_CLOSE_FRAMES = 1  # 闭嘴持续帧数

# 混合模式的音量阈值 - 只有音量超过此值时才启用固定频率张合
# 低于此值时保持闭嘴
MOUTH_VOLUME_THRESHOLD = 0.15

# 眨眼频率范围 (秒)
BLINK_MIN_INTERVAL = 3.0
BLINK_MAX_INTERVAL = 6.0

# 眨眼持续时间范围 (毫秒)
BLINK_MIN_DURATION_MS = 100
BLINK_MAX_DURATION_MS = 150

# 长时间静默阈值 (秒) - 用于 lecture 场景切换姿态
LONG_SILENCE_THRESHOLD = 5.0

# 数字人高度
AVATAR_HEIGHT_LECTURE = 450
AVATAR_HEIGHT_CODING = 320

# ============================================================
# VAD 模型（全局加载）
# ============================================================

_vad_model = None
_vad_utils = None


def load_vad_model():
    """加载 silero-vad 模型"""
    global _vad_model, _vad_utils
    if _vad_model is None:
        print("加载 VAD 人声检测模型...")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    return _vad_model, _vad_utils


# ============================================================
# 音频处理模块
# ============================================================


def extract_audio(video_path: str, output_path: str, sample_rate: int = None) -> bool:
    """
    使用 ffmpeg 从视频提取音频
    
    Args:
        video_path: 输入视频路径
        output_path: 输出音频路径 (.wav)
        sample_rate: 目标采样率，None 表示保持原始采样率
    
    Returns:
        是否成功
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn",  # 不要视频
        "-acodec", "pcm_s16le",  # PCM 格式
        "-ac", "1",  # 单声道
    ]
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(output_path)
    
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def get_speech_mask(audio_path: str, fps: float, total_frames: int,
                    vad_threshold: float = VAD_THRESHOLD,
                    padding_ms: int = PADDING_MS) -> np.ndarray:
    """
    使用 VAD 检测人声，返回 speech_mask
    
    Args:
        audio_path: 16kHz 采样率的音频文件
        fps: 视频帧率
        total_frames: 总帧数
        vad_threshold: VAD 阈值
        padding_ms: 人声前后的 padding 毫秒数
    
    Returns:
        speech_mask: bool 数组，长度为 total_frames
    """
    model, utils = load_vad_model()
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    # 读取音频（VAD 需要 16kHz）
    wav = read_audio(audio_path, sampling_rate=16000)
    
    # 获取人声时间段
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=vad_threshold,
        sampling_rate=16000,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
    )
    
    print(f"    VAD 检测到 {len(speech_timestamps)} 个人声片段")
    
    # 创建掩码
    speech_mask = np.zeros(total_frames, dtype=bool)
    
    # 计算 padding 的帧数
    padding_frames = int(padding_ms * fps / 1000)
    
    # 标记人声帧
    for ts in speech_timestamps:
        start_sec = ts["start"] / 16000
        end_sec = ts["end"] / 16000
        start_frame = max(0, int(start_sec * fps) - padding_frames)
        end_frame = min(total_frames, int(end_sec * fps) + padding_frames)
        speech_mask[start_frame:end_frame] = True
    
    return speech_mask


def get_volume_per_frame(audio_path: str, fps: float, total_frames: int) -> np.ndarray:
    """
    计算每帧的归一化音量
    
    Args:
        audio_path: 原始采样率的音频文件
        fps: 视频帧率
        total_frames: 总帧数
    
    Returns:
        volume_per_frame: float 数组，值范围 0-1
    """
    # 读取音频（保持原始采样率）
    audio_data, sr = sf.read(audio_path)
    
    # 如果是立体声，转为单声道
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 计算每帧对应的采样点数
    samples_per_frame = int(sr / fps)
    volume_per_frame = np.zeros(total_frames)
    
    for i in range(total_frames):
        start_sample = i * samples_per_frame
        end_sample = min(len(audio_data), start_sample + samples_per_frame)
        
        if start_sample < len(audio_data):
            # 计算这一帧的 RMS 音量
            frame_audio = audio_data[start_sample:end_sample]
            rms = np.sqrt(np.mean(frame_audio ** 2))
            volume_per_frame[i] = rms
    
    # 归一化到 0-1（使用最大值）
    max_vol = np.max(volume_per_frame)
    if max_vol > 0:
        volume_per_frame = volume_per_frame / max_vol
    
    return volume_per_frame


# ============================================================
# 眨眼和静默检测模块
# ============================================================


def generate_blink_frames(total_frames: int, fps: float,
                          min_interval: float = BLINK_MIN_INTERVAL,
                          max_interval: float = BLINK_MAX_INTERVAL,
                          min_duration_ms: float = BLINK_MIN_DURATION_MS,
                          max_duration_ms: float = BLINK_MAX_DURATION_MS) -> np.ndarray:
    """
    生成眨眼帧标记
    
    注意：不依赖 speech_mask，任何时候都可以眨眼（与 V1 保持一致）
    
    Args:
        total_frames: 总帧数
        fps: 帧率
        min_interval: 最小眨眼间隔（秒）
        max_interval: 最大眨眼间隔（秒）
        min_duration_ms: 最小眨眼时长（毫秒）
        max_duration_ms: 最大眨眼时长（毫秒）
    
    Returns:
        blink_mask: bool 数组，长度为 total_frames
    """
    blink_mask = np.zeros(total_frames, dtype=bool)
    current_frame = 0
    
    while current_frame < total_frames:
        # 随机下一次眨眼的间隔
        interval_sec = random.uniform(min_interval, max_interval)
        interval_frames = int(interval_sec * fps)
        
        blink_start = current_frame + interval_frames
        
        if blink_start >= total_frames:
            break
        
        # 随机眨眼持续时间
        blink_duration_ms = random.uniform(min_duration_ms, max_duration_ms)
        blink_duration_frames = max(1, int(blink_duration_ms * fps / 1000))
        
        blink_end = min(total_frames, blink_start + blink_duration_frames)
        
        # 标记眨眼帧（不检查 speech_mask，任何时候都可以眨眼）
        blink_mask[blink_start:blink_end] = True
        
        current_frame = blink_end
    
    return blink_mask


def get_long_silence_mask(speech_mask: np.ndarray, fps: float,
                          threshold_sec: float = LONG_SILENCE_THRESHOLD) -> np.ndarray:
    """
    计算每帧往后看是否长时间静默
    
    用于 lecture 场景判断是否切换到静默姿态（1.png, 2.png）
    
    Args:
        speech_mask: 人声掩码
        fps: 帧率
        threshold_sec: 静默阈值（秒）
    
    Returns:
        long_silence_mask: bool 数组
    """
    total_frames = len(speech_mask)
    threshold_frames = int(threshold_sec * fps)
    long_silence_mask = np.zeros(total_frames, dtype=bool)
    
    for i in range(total_frames):
        end_check = min(total_frames, i + threshold_frames)
        # 如果从当前帧往后 threshold_frames 帧内都没有人声，则标记为长时间静默
        if not np.any(speech_mask[i:end_check]):
            long_silence_mask[i] = True
    
    return long_silence_mask


def generate_mouth_mask(speech_mask: np.ndarray, volume_per_frame: np.ndarray,
                        mode: str = MOUTH_MODE,
                        open_frames: int = MOUTH_OPEN_FRAMES,
                        close_frames: int = MOUTH_CLOSE_FRAMES,
                        volume_threshold: float = MOUTH_VOLUME_THRESHOLD) -> np.ndarray:
    """
    生成嘴巴张合 mask
    
    Args:
        speech_mask: 人声掩码
        volume_per_frame: 每帧音量
        mode: "fixed" 或 "volume"
        open_frames: 固定模式下张嘴持续帧数
        close_frames: 固定模式下闭嘴持续帧数
        volume_threshold: 音量模式下的阈值
    
    Returns:
        mouth_open_mask: bool 数组，True 表示张嘴
    """
    total_frames = len(speech_mask)
    mouth_open_mask = np.zeros(total_frames, dtype=bool)
    
    if mode == "fixed":
        # 固定频率模式：在说话时以固定频率交替张合
        cycle_length = open_frames + close_frames
        in_speech = False
        speech_start_frame = 0
        
        for i in range(total_frames):
            if speech_mask[i]:
                if not in_speech:
                    # 刚开始说话，记录起始帧
                    in_speech = True
                    speech_start_frame = i
                
                # 计算在当前说话段中的相对位置
                relative_frame = i - speech_start_frame
                position_in_cycle = relative_frame % cycle_length
                
                # 前 open_frames 帧张嘴，后 close_frames 帧闭嘴
                if position_in_cycle < open_frames:
                    mouth_open_mask[i] = True
            else:
                in_speech = False
    
    elif mode == "hybrid":
        # 混合模式：音量超过阈值时用固定频率张合，否则闭嘴
        cycle_length = open_frames + close_frames
        in_speaking = False  # 是否在"有效说话"中（音量超过阈值）
        speaking_start_frame = 0
        
        for i in range(total_frames):
            if speech_mask[i] and volume_per_frame[i] > volume_threshold:
                # 音量超过阈值，启用固定频率张合
                if not in_speaking:
                    in_speaking = True
                    speaking_start_frame = i
                
                relative_frame = i - speaking_start_frame
                position_in_cycle = relative_frame % cycle_length
                
                if position_in_cycle < open_frames:
                    mouth_open_mask[i] = True
            else:
                # 音量低于阈值或不在说话，保持闭嘴
                in_speaking = False
    
    else:
        # 音量模式：根据音量决定张嘴/闭嘴
        for i in range(total_frames):
            if speech_mask[i] and volume_per_frame[i] > volume_threshold:
                mouth_open_mask[i] = True
    
    return mouth_open_mask


# ============================================================
# 帧选择逻辑模块
# ============================================================


def select_avatar_frame_lecture(is_speaking: bool, is_mouth_open: bool,
                                 is_blinking: bool, is_long_silence: bool) -> str:
    """
    lecture 场景的帧选择逻辑
    
    素材映射（根据 status_info）：
    - 1.png (silent_open): 长时间不说话，睁开眼睛
    - 2.png (silent_blink): 长时间不说话，闭上眼睛
    - 3.png (speak_closed): 说话时，睁开眼睛，嘴巴闭上
    - 4.png (speak_open): 说话时，睁开眼睛，嘴巴张开
    - 5.png (speak_blink): 说话时，闭上眼睛，嘴巴闭上
    
    状态组合：
    - 说话 + 音量高 + 不眨眼 → speak_open (4.png)
    - 说话 + 音量高 + 眨眼 → speak_blink (5.png)
    - 说话 + 音量低 + 不眨眼 → speak_closed (3.png)
    - 说话 + 音量低 + 眨眼 → speak_blink (5.png)
    - 短暂停顿 + 不眨眼 → speak_closed (3.png) - 保持说话姿态
    - 短暂停顿 + 眨眼 → speak_blink (5.png)
    - 长时间静默 + 不眨眼 → silent_open (1.png)
    - 长时间静默 + 眨眼 → silent_blink (2.png)
    
    Returns:
        图片 key
    """
    if is_long_silence:
        # 长时间静默 → 使用静默姿态 (1.png, 2.png)
        if is_blinking:
            return "silent_blink"  # 2.png
        else:
            return "silent_open"   # 1.png
    else:
        # 说话中或短暂停顿 → 使用说话姿态 (3.png, 4.png, 5.png)
        if is_blinking:
            return "speak_blink"   # 5.png - 眨眼时闭嘴
        elif is_speaking and is_mouth_open:
            return "speak_open"    # 4.png - 说话且音量高
        else:
            return "speak_closed"  # 3.png - 说话但音量低，或短暂停顿


def select_avatar_frame_coding(is_speaking: bool, is_mouth_open: bool,
                                is_blinking: bool) -> str:
    """
    coding 场景的帧选择逻辑
    
    素材映射（根据 status_info）：
    - 1.png (speak_closed): 睁开眼睛，嘴巴闭上
    - 2.png (speak_open): 睁开眼睛，嘴巴张开
    - 3.png (speak_blink): 闭上眼睛，嘴巴闭上
    
    coding 场景只有一种姿态，不区分长时间静默
    
    状态组合：
    - 说话 + 音量高 + 不眨眼 → speak_open (2.png)
    - 说话 + 音量高 + 眨眼 → speak_blink (3.png)
    - 说话 + 音量低 + 不眨眼 → speak_closed (1.png)
    - 说话 + 音量低 + 眨眼 → speak_blink (3.png)
    - 不说话 + 不眨眼 → speak_closed (1.png)
    - 不说话 + 眨眼 → speak_blink (3.png)
    
    Returns:
        图片 key
    """
    if is_blinking:
        return "speak_blink"   # 3.png - 眨眼时闭嘴
    elif is_speaking and is_mouth_open:
        return "speak_open"    # 2.png - 说话且音量高
    else:
        return "speak_closed"  # 1.png - 其他情况


# ============================================================
# 素材加载和叠加模块
# ============================================================


def load_avatar_images(avatar_type: str) -> dict:
    """
    加载数字人素材
    
    Args:
        avatar_type: "lecture" 或 "coding"
    
    Returns:
        素材字典 {"silent_open": img, "silent_blink": img, ...}
    """
    folder = os.path.join(AVATAR_FOLDER, avatar_type)
    images = {}
    
    # 尝试不同的图片扩展名
    for ext in ["png", "jpg", "jpeg"]:
        test_path = os.path.join(folder, f"1.{ext}")
        if os.path.exists(test_path):
            if avatar_type == "lecture":
                # lecture: 5 张素材
                images["silent_open"] = cv2.imread(os.path.join(folder, f"1.{ext}"), cv2.IMREAD_UNCHANGED)
                images["silent_blink"] = cv2.imread(os.path.join(folder, f"2.{ext}"), cv2.IMREAD_UNCHANGED)
                images["speak_closed"] = cv2.imread(os.path.join(folder, f"3.{ext}"), cv2.IMREAD_UNCHANGED)
                images["speak_open"] = cv2.imread(os.path.join(folder, f"4.{ext}"), cv2.IMREAD_UNCHANGED)
                images["speak_blink"] = cv2.imread(os.path.join(folder, f"5.{ext}"), cv2.IMREAD_UNCHANGED)
            else:  # coding
                # coding: 3 张素材
                images["speak_closed"] = cv2.imread(os.path.join(folder, f"1.{ext}"), cv2.IMREAD_UNCHANGED)
                images["speak_open"] = cv2.imread(os.path.join(folder, f"2.{ext}"), cv2.IMREAD_UNCHANGED)
                images["speak_blink"] = cv2.imread(os.path.join(folder, f"3.{ext}"), cv2.IMREAD_UNCHANGED)
                # coding 场景复用素材
                images["silent_open"] = images["speak_closed"]
                images["silent_blink"] = images["speak_blink"]
            break
    
    # 验证所有图片都加载成功
    if images:
        for key, img in images.items():
            if img is None:
                print(f"    警告: {avatar_type}/{key} 加载失败")
                return None
    
    return images if images else None


def resize_avatar(img: np.ndarray, target_height: int) -> np.ndarray:
    """缩放单张图片"""
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def resize_avatar_dict(images: dict, target_height: int) -> dict:
    """缩放所有图片"""
    return {key: resize_avatar(img, target_height) for key, img in images.items()}


def flip_avatar_dict(images: dict) -> dict:
    """水平翻转所有图片（用于左侧位置）"""
    return {key: cv2.flip(img, 1) if img is not None else None for key, img in images.items()}


def get_position_coords(position: str, video_w: int, video_h: int,
                        avatar_w: int, avatar_h: int) -> tuple:
    """
    根据位置字符串计算坐标
    
    Args:
        position: "bottom-right", "bottom-left", "top-right", "top-left"
        video_w, video_h: 视频尺寸
        avatar_w, avatar_h: 数字人尺寸
    
    Returns:
        (x, y) 左上角坐标
    """
    margin = 20
    if position == "bottom-right":
        return video_w - avatar_w - margin, video_h - avatar_h - margin
    elif position == "bottom-left":
        return margin, video_h - avatar_h - margin
    elif position == "top-right":
        return video_w - avatar_w - margin, margin
    elif position == "top-left":
        return margin, margin
    else:
        # 默认右下角
        return video_w - avatar_w - margin, video_h - avatar_h - margin


def overlay_avatar(frame: np.ndarray, avatar: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    将数字人图片叠加到视频帧
    
    Args:
        frame: BGR 格式的视频帧
        avatar: BGRA 格式的数字人图片（带 alpha 通道）
        x, y: 叠加位置（左上角坐标）
    
    Returns:
        叠加后的视频帧
    """
    h, w = avatar.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    
    # 边界检查
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return frame
    
    if avatar.shape[2] == 4:
        # 有 alpha 通道，进行透明叠加
        alpha = avatar[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                alpha * avatar[:, :, c] +
                (1 - alpha) * frame[y:y+h, x:x+w, c]
            )
    else:
        # 无 alpha 通道，直接覆盖
        frame[y:y+h, x:x+w] = avatar
    
    return frame


# ============================================================
# 主视频处理模块
# ============================================================


def process_video(video_path: str, json_path: str, output_path: str) -> bool:
    """
    处理单个视频
    
    Args:
        video_path: 输入视频路径
        json_path: JSON 配置文件路径
        output_path: 输出视频路径
    
    Returns:
        是否成功
    """
    print(f"\n处理视频: {os.path.basename(video_path)}")
    
    # 1. 读取 JSON 配置
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        segments = config["segments"]
        print(f"  片段数: {len(segments)}")
    except Exception as e:
        print(f"  错误: 无法读取 JSON 配置 - {e}")
        return False
    
    # 2. 提取音频
    temp_audio_vad = output_path + ".temp_vad.wav"
    temp_audio_vol = output_path + ".temp_vol.wav"
    
    print("  提取音频...")
    if not extract_audio(video_path, temp_audio_vad, sample_rate=16000):
        print("  错误: 无法提取 VAD 音频")
        return False
    if not extract_audio(video_path, temp_audio_vol, sample_rate=None):
        print("  错误: 无法提取音量音频")
        return False
    
    # 3. 打开视频获取信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  错误: 无法打开视频")
        return False
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_video_frames / video_fps
    print(f"  视频信息: {video_w}x{video_h}, {video_fps:.2f}fps, {total_video_frames}帧, {video_duration:.1f}秒")
    
    # 4. 分析音频
    print("  分析音频...")
    speech_mask = get_speech_mask(temp_audio_vad, video_fps, total_video_frames)
    volume_per_frame = get_volume_per_frame(temp_audio_vol, video_fps, total_video_frames)
    blink_mask = generate_blink_frames(total_video_frames, video_fps)
    long_silence_mask = get_long_silence_mask(speech_mask, video_fps)
    
    # 生成嘴巴张合 mask
    mouth_open_mask = generate_mouth_mask(speech_mask, volume_per_frame)
    
    # 打印统计信息
    speech_ratio = np.sum(speech_mask) / len(speech_mask)
    blink_count = np.sum(np.diff(blink_mask.astype(int)) == 1)
    print(f"  人声占比: {speech_ratio*100:.1f}%")
    print(f"  眨眼次数: {blink_count}")
    
    # 嘴巴开闭统计
    speech_frames = np.where(speech_mask)[0]
    if len(speech_frames) > 0:
        mouth_open_count = np.sum(mouth_open_mask[speech_frames])
        mouth_open_ratio = mouth_open_count / len(speech_frames)
        print(f"  说话帧数: {len(speech_frames)}")
        print(f"  张嘴帧数: {mouth_open_count} ({mouth_open_ratio*100:.1f}%)")
        print(f"  嘴巴模式: {MOUTH_MODE}")
        if MOUTH_MODE == "fixed":
            print(f"  张合周期: 张{MOUTH_OPEN_FRAMES}帧 + 闭{MOUTH_CLOSE_FRAMES}帧")
    
    # 调试：打印开头帧的状态
    print(f"  开头10帧 speech_mask: {speech_mask[:10].tolist()}")
    print(f"  开头10帧 mouth_open: {mouth_open_mask[:10].tolist()}")
    
    # 5. 加载素材
    print("  加载素材...")
    lecture_imgs = load_avatar_images("lecture")
    coding_imgs = load_avatar_images("coding")
    
    if lecture_imgs:
        lecture_imgs = resize_avatar_dict(lecture_imgs, AVATAR_HEIGHT_LECTURE)
        lecture_imgs_flipped = flip_avatar_dict(lecture_imgs)
    else:
        lecture_imgs_flipped = None
        
    if coding_imgs:
        coding_imgs = resize_avatar_dict(coding_imgs, AVATAR_HEIGHT_CODING)
        coding_imgs_flipped = flip_avatar_dict(coding_imgs)
    else:
        coding_imgs_flipped = None
    
    # 6. 构建帧处理计划
    frame_plan = []  # [(keep, seg_type, position), ...]
    
    for frame_idx in range(total_video_frames):
        frame_time = frame_idx / video_fps
        
        # 找到这一帧属于哪个片段
        seg_type = None
        position = None
        for seg in segments:
            if seg["start"] <= frame_time < seg["end"]:
                seg_type = seg["type"]
                position = seg.get("position", "bottom-right")
                break
        
        if seg_type == "silent":
            frame_plan.append((False, None, None))  # 不保留
        elif seg_type in ["lecture", "coding"]:
            frame_plan.append((True, seg_type, position))
        elif seg_type == "hardware":
            frame_plan.append((True, "hardware", None))  # 保留但不叠加
        else:
            # 不在任何片段中，默认保留但不叠加
            frame_plan.append((True, None, None))
    
    # 7. 逐帧处理
    print("  逐帧处理...")
    temp_video = output_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, video_fps, (video_w, video_h))
    
    state_counts = {
        "speak_open": 0, "speak_closed": 0, "speak_blink": 0,
        "silent_open": 0, "silent_blink": 0,
        "no_overlay": 0, "skipped": 0
    }
    
    frame_idx = 0
    kept_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        keep, seg_type, position = frame_plan[frame_idx]
        
        if not keep:
            state_counts["skipped"] += 1
            frame_idx += 1
            continue
        
        if seg_type in ["lecture", "coding"]:
            # 选择素材集
            if seg_type == "lecture":
                if position in ["bottom-left", "top-left"]:
                    imgs = lecture_imgs_flipped
                else:
                    imgs = lecture_imgs
            else:  # coding
                if position in ["bottom-left", "top-left"]:
                    imgs = coding_imgs_flipped
                else:
                    imgs = coding_imgs
            
            if imgs:
                # 获取当前帧的状态
                is_speaking = bool(speech_mask[frame_idx])
                is_mouth_open = bool(mouth_open_mask[frame_idx])  # 使用预计算的 mask
                is_blinking = bool(blink_mask[frame_idx])
                is_long_silence = bool(long_silence_mask[frame_idx])
                
                # 选择图片
                if seg_type == "lecture":
                    img_key = select_avatar_frame_lecture(
                        is_speaking, is_mouth_open, is_blinking, is_long_silence
                    )
                else:
                    img_key = select_avatar_frame_coding(
                        is_speaking, is_mouth_open, is_blinking
                    )
                
                avatar = imgs[img_key]
                
                # 计算位置
                avatar_h, avatar_w = avatar.shape[:2]
                x, y = get_position_coords(position, video_w, video_h, avatar_w, avatar_h)
                
                # 叠加
                frame = overlay_avatar(frame, avatar, x, y)
                
                state_counts[img_key] += 1
                
                # 调试输出（每 100 帧输出一次）
                if frame_idx % 100 == 0:
                    print(f"    [帧{frame_idx}] speaking={is_speaking}, mouth_open={is_mouth_open}, "
                          f"blink={is_blinking}, long_silence={is_long_silence} → {img_key}")
            else:
                state_counts["no_overlay"] += 1
        else:
            state_counts["no_overlay"] += 1
        
        out.write(frame)
        kept_frames += 1
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"  保留帧数: {kept_frames}/{total_video_frames}")
    print(f"  [状态统计]")
    for state, count in state_counts.items():
        if count > 0:
            print(f"    {state}: {count}")
    
    # 8. 处理音频（删除静默片段）
    print("  处理音频...")
    
    keep_segments = []
    for seg in segments:
        if seg["type"] != "silent":
            keep_segments.append((seg["start"], seg["end"]))
    
    if keep_segments:
        temp_audio_final = output_path + ".temp_audio.wav"
        
        # 使用 ffmpeg 的 atrim 和 concat 滤镜
        filter_parts = []
        for i, (start, end) in enumerate(keep_segments):
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
        
        concat_inputs = "".join([f"[a{i}]" for i in range(len(keep_segments))])
        filter_complex = "".join(filter_parts) + f"{concat_inputs}concat=n={len(keep_segments)}:v=0:a=1[outa]"
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outa]",
            temp_audio_final
        ]
        subprocess.run(cmd)
        
        # 合并视频和音频
        print("  合并视频和音频...")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video,
            "-i", temp_audio_final,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd)
        
        if os.path.exists(temp_audio_final):
            os.remove(temp_audio_final)
    else:
        print("  警告: 没有保留的片段")
    
    # 9. 清理临时文件
    for f in [temp_video, temp_audio_vad, temp_audio_vol]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"  完成: {output_path}")
    return True


# ============================================================
# 主程序入口
# ============================================================


def main():
    """主函数"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 获取所有视频文件
    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    
    if not video_files:
        print(f"未找到视频文件: {VIDEO_FOLDER}")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"输出目录: {OUTPUT_FOLDER}")
    print("-" * 50)
    
    # 预加载 VAD 模型
    load_vad_model()
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(JSON_FOLDER, f"{video_name}.json")
        output_path = os.path.join(OUTPUT_FOLDER, f"{video_name}.mp4")
        
        # 检查 JSON 是否存在
        if not os.path.exists(json_path):
            print(f"\n跳过 {video_name}: 未找到对应的 JSON 文件")
            continue
        
        # 检查是否已处理
        if os.path.exists(output_path):
            print(f"\n跳过 {video_name}: 输出文件已存在")
            continue
        
        try:
            process_video(video_path, json_path, output_path)
        except Exception as e:
            import traceback
            print(f"\n处理 {video_name} 失败: {e}")
            traceback.print_exc()
    
    print("-" * 50)
    print("全部处理完成！")


if __name__ == "__main__":
    main()
