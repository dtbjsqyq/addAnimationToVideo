# 视频分析 Prompt

## 任务

分析这个 1920x1080 的教学录屏视频，按状态变化划分片段，输出 JSON。

## 数字人素材说明

我有两套数字人素材：
- **standing（站姿）**：用于讲课，尺寸约 350x450 像素
- **sitting（坐姿）**：用于写代码，尺寸约 250x320 像素，优先放底部，实在不行可放顶部 实在不行也可以放到左上角或者右上角

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
  ]
}
```

## 要求

1. `video_name` 必须与上传的视频文件名完全一致
2. 时间连续，覆盖整个视频
3. 相邻同类型同位置的片段要合并
4. note 简短描述（用于人工核对）

请分析视频并输出 JSON。
