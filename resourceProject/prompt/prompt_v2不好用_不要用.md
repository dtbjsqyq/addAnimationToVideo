# 视频分析 Prompt V2（含字幕） 这个不好用，不要用！！！

## 任务

分析这个 1920x1080 的教学录屏视频，完成两件事：
1. 按状态变化划分片段
2. **转录视频中的语音为中文字幕**

输出 JSON。

## 数字人素材说明

我有两套数字人素材：
- **standing（站姿）**：用于讲课，尺寸约 350x450 像素
- **sitting（坐姿）**：用于写代码，尺寸约 250x320 像素，优先放底部，实在不行可放顶部或左上角、右上角

## 片段类型

| type | 画面特征 | avatar_state | 处理方式 |
|------|----------|--------------|----------|
| `lecture` | PPT、PDF、网页文档、口头讲解 | `standing` | 叠加站姿数字人 |
| `coding` | IDE 编辑器、终端、写代码、调试 | `sitting` | 叠加坐姿数字人 |
| `hardware` | **摄像头实拍**：展示硬件、手部操作、接线 | `none` | 保留视频，不加数字人 |
| `silent` | 超过 10 秒无语音且无操作 | `none` | 删除这段视频 |

**注意：**
- PPT 里显示硬件图片仍算 `lecture`，只有摄像头实拍才算 `hardware`
- 说话或有操作都不算 `silent`

## 位置规则

| position | 适用场景 |
|----------|----------|
| `bottom-right` | 默认首选 |
| `bottom-left` | 右下角有重要内容时 |
| `top-right` | 底部有重要内容时 |
| `top-left` | 底部和右上都有重要内容时 |

**规则：**
- `sitting` 状态优先放底部，底部实在放不下可放顶部
- `hardware` 和 `silent` 的 position 为 `null`
- 不要遮挡代码、配置、关键文字
- video_name 为我上传的视频名字

## 字幕规则（重要！）

1. **转录所有语音**：将视频中的语音完整转录为中文文字
2. **按句子分段**：每条字幕是一个完整的句子或短语，不要太长
3. **时间精确**：start 和 end 精确到小数点后 1 位（秒）
4. **字幕长度**：每条字幕不超过 20 个中文字符，太长要拆分
5. **跳过静默**：`silent` 类型片段内没有字幕
6. **保留口语**：保留"嗯"、"那个"等口语词，但可以适当精简

## 输出格式

只输出 JSON，不要其他文字：

```json
{
  "video_name": "文件名.mp4",
  "total_duration": 300,
  "segments": [
    {"start": 0, "end": 45, "type": "lecture", "avatar_state": "standing", "position": "bottom-right", "note": "PPT介绍"},
    {"start": 45, "end": 150, "type": "coding", "avatar_state": "sitting", "position": "bottom-left", "note": "VS Code写代码"},
    {"start": 150, "end": 200, "type": "hardware", "avatar_state": "none", "position": null, "note": "摄像头演示Arduino连线"},
    {"start": 200, "end": 215, "type": "silent", "avatar_state": "none", "position": null, "note": "等待无操作"},
    {"start": 215, "end": 300, "type": "lecture", "avatar_state": "standing", "position": "bottom-right", "note": "总结"}
  ],
  "subtitles": [
    {"start": 0.0, "end": 2.5, "text": "大家好，今天我们来学习Arduino"},
    {"start": 2.5, "end": 5.0, "text": "首先打开这个PDF文档"},
    {"start": 5.0, "end": 8.2, "text": "这里介绍了Arduino的基本概念"},
    {"start": 45.0, "end": 47.5, "text": "现在我们打开VS Code"},
    {"start": 47.5, "end": 50.0, "text": "开始写第一个程序"},
    {"start": 150.0, "end": 153.0, "text": "这是Arduino开发板"},
    {"start": 153.0, "end": 156.5, "text": "我们把这根线接到这里"},
    {"start": 215.0, "end": 218.0, "text": "好，今天的内容就到这里"},
    {"start": 218.0, "end": 220.5, "text": "下节课我们继续"}
  ]
}
```

## 要求

1. `video_name` 必须与上传的视频文件名完全一致
2. segments 时间连续，覆盖整个视频
3. 相邻同类型同位置的片段要合并
4. note 简短描述（用于人工核对）
5. **subtitles 必须覆盖所有有语音的部分**
6. **subtitles 的时间范围必须在非 silent 片段内**
7. **字幕文字必须是中文**

请分析视频并输出 JSON。
