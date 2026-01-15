# 数字人视频生成器 (Digital Avatar Video Generator)

一个基于 Python 的教学视频数字人叠加工具，可以根据 JSON 配置在视频上自动叠加动画数字人，支持口型同步、眨眼动画和多种场景切换。

## 功能特点

- **智能口型同步**：基于 VAD 人声检测和音量分析，实现自然的口型动画
- **自然眨眼动画**：随机间隔的眨眼效果，让数字人更生动
- **多场景支持**：
  - `lecture`：讲课场景（站姿，5张素材）
  - `coding`：编程场景（坐姿，3张素材）
  - `hardware`：硬件展示（不叠加数字人）
  - `silent`：静默片段（自动删除）
- **灵活位置**：支持四角定位（bottom-right, bottom-left, top-right, top-left）
- **Gemini AI 集成**：可使用 Gemini API 自动分析视频并生成配置

## 项目结构

```
├── onlyPyV4/                    # 主程序
│   ├── character_video_v4.py    # 数字人视频生成器核心代码
│   └── 开始生成.bat             # Windows 启动脚本
├── testGeminiAPI/               # Gemini API 测试
│   ├── testApi.py               # API 调用示例
│   └── global_const.py          # API Key 配置
├── resourceProject/
│   ├── prompt/                  # Gemini 提示词模板
│   └── geminiGen/               # 生成的 JSON 配置文件
└── source/
    ├── lecture/                 # 输入视频目录
    ├── characterStyle/          # 数字人素材
    │   ├── lecture/             # 讲课素材 (1-5.png)
    │   └── coding/              # 编程素材 (1-3.png)
    └── output/                  # 输出视频目录
```

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 FFmpeg

FFmpeg 是必需的外部依赖，用于音视频处理。

**Windows:**
```bash
# 使用 Chocolatey
choco install ffmpeg

# 或使用 Scoop
scoop install ffmpeg

# 或手动下载: https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

### 3. 配置 Gemini API（可选）

如需使用 AI 自动分析视频，在 `testGeminiAPI/global_const.py` 中配置 API Key：

```python
api_key = "your-gemini-api-key"
```

## 使用方法

### 1. 准备素材

将数字人图片放入 `source/characterStyle/` 目录：
- `lecture/`: 1.png ~ 5.png（静默睁眼、静默闭眼、说话闭嘴、说话张嘴、说话眨眼）
- `coding/`: 1.png ~ 3.png（闭嘴、张嘴、眨眼）

### 2. 准备视频和配置

- 将视频放入 `source/lecture/` 目录
- 在 `resourceProject/geminiGen/` 创建同名 JSON 配置文件

### 3. 运行生成

```bash
cd onlyPyV4
python character_video_v4.py
```

或双击 `开始生成.bat`

## JSON 配置格式

```json
{
  "video_name": "example.mp4",
  "total_duration": 300,
  "segments": [
    {"start": 0, "end": 45, "type": "lecture", "position": "bottom-right", "note": "PPT介绍"},
    {"start": 45, "end": 150, "type": "coding", "position": "bottom-left", "note": "写代码"},
    {"start": 150, "end": 200, "type": "hardware", "position": null, "note": "硬件演示"},
    {"start": 200, "end": 215, "type": "silent", "position": null, "note": "静默删除"}
  ]
}
```

## 参数调整

在 `character_video_v4.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VAD_THRESHOLD` | 0.25 | VAD 人声检测阈值 |
| `MOUTH_MODE` | "hybrid" | 口型模式 (volume/fixed/hybrid) |
| `BLINK_MIN_INTERVAL` | 3.0 | 最小眨眼间隔（秒） |
| `AVATAR_HEIGHT_LECTURE` | 450 | 讲课数字人高度 |
| `AVATAR_HEIGHT_CODING` | 320 | 编程数字人高度 |

## 依赖说明

- **numpy**: 数值计算
- **opencv-python**: 视频处理
- **soundfile**: 音频读取
- **torch/torchaudio**: VAD 人声检测（silero-vad）
- **ffmpeg**: 音视频编解码（外部依赖）
- **google-genai**: Gemini API（可选）

## License

MIT
