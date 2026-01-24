# 从 ChatTTS 迁移到 Qwen-TTS

## v0.2.0 版本的重大变化

tts-cli v0.2.0 版本将底层引擎从 ChatTTS 升级到 Qwen3-TTS，带来了以下改进：

- ✅ **声音克隆**: 从 3 秒参考音频克隆任意声音
- ✅ **9 种高级预设音色**: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- ✅ **自然语言声音设计**: 用自然语言描述想要的声音特质
- ✅ **更长文本支持**: 单次处理长度从 500 增加到 1000 字符
- ✅ **更广泛的字符支持**: 原生支持数字、标点符号等（无需转换）
- ✅ **多语言支持**: 支持 10 种语言（中、英、日、韩、德、法、俄、葡、西、意）

### ⚠️ 破坏性变更

#### 1. 说话人音色文件格式变更 (.pt → .qwen-voice)

**问题**: ChatTTS 的说话人文件 (`.pt`) **不兼容** Qwen-TTS。

**迁移方案**: 使用声音克隆功能重新创建音色文件。

##### 步骤 1: 准备参考音频

为每个旧的 `.pt` 音色准备：
- 3-10 秒的参考音频文件（WAV/MP3/FLAC 格式）
- 参考音频的准确文本转录

**音频质量要求**:
- 清晰、背景噪音少
- 单人说话
- 转录文本需要完全匹配音频内容

##### 步骤 2: 克隆声音并保存

```bash
# 使用参考音频克隆声音，并保存为 .qwen-voice 文件
tts-cli --mode clone \
  --reference-audio reference.wav \
  --reference-text "这是参考音频的文本转录" \
  --text "测试克隆的声音" \
  --save-speaker my_voice.qwen-voice \
  --output test.wav
```

##### 步骤 3: 使用新的音色文件

```bash
# 旧方式（v0.1.x，不再支持）
# tts-cli --text "你好" --speaker voice.pt

# 新方式（v0.2.0+）
tts-cli --text "你好" --speaker my_voice.qwen-voice --output output.wav
```

#### 2. 文本处理变更

**ChatTTS (v0.1.x)**: 需要激进的文本规范化
- 数字转为文字 (3 → three)
- 有限的标点符号支持
- 严格的字符过滤

**Qwen-TTS (v0.2.0+)**: 最小化规范化
- 原生支持数字
- 完整的标点符号支持
- 广泛的字符集支持

**迁移影响**: 现有文本无需修改，会自动获得更好的质量。

#### 3. 默认参数变更

| 参数 | v0.1.x (ChatTTS) | v0.2.0+ (Qwen-TTS) |
|------|------------------|-------------------|
| `--max-length` | 500 | 1000 |
| `--max-batch` | 1 | 4 |
| `--speaker` | 随机生成 | Ryan (英文男声) |

## 新功能使用指南

### 1. 使用 9 种预设音色

```bash
# 中文女声
tts-cli --text "你好，世界" --speaker Vivian --output hello.wav

# 英文男声
tts-cli --text "Hello, world" --speaker Ryan --output hello.wav

# 日语女声
tts-cli --text "こんにちは、世界" --speaker Ono_Anna --output hello.wav
```

**可用音色列表**:

| 音色名称 | 语言 | 性别 | 特点 |
|---------|------|------|------|
| Vivian | 中文 | 女 | 清晰、专业 |
| Serena | 中文 | 女 | 温暖、友好 |
| Uncle_Fu | 中文 | 男 | 成熟、权威 |
| Dylan | 中文（北京） | 男 | 年轻、有活力 |
| Eric | 中文（四川） | 男 | 独特方言 |
| Ryan | 英语 | 男 | 清晰、中性 |
| Aiden | 英语 | 男 | 专业、温暖 |
| Ono_Anna | 日语 | 女 | 清晰、专业 |
| Sohee | 韩语 | 女 | 清晰、友好 |

### 2. 声音克隆工作流

#### 方案 A: 从头创建克隆声音

```bash
# 步骤 1: 录制或准备 3-10 秒的参考音频
# - 格式: WAV/MP3/FLAC
# - 质量: 清晰、无背景噪音
# - 内容: 单人说话

# 步骤 2: 转录参考音频文本
REFERENCE_TEXT="这是我的声音样本，用于克隆声音"

# 步骤 3: 创建并保存克隆的声音
tts-cli --mode clone \
  --reference-audio my_voice_sample.wav \
  --reference-text "$REFERENCE_TEXT" \
  --text "这是测试内容" \
  --save-speaker my_voice.qwen-voice \
  --output test.wav

# 步骤 4: 在后续使用中重用克隆的声音
tts-cli --speaker my_voice.qwen-voice \
  --file long_article.txt \
  --output article.wav
```

#### 方案 B: 从现有音频提取声音（用于替换旧 .pt 文件）

```bash
# 如果你有之前用 ChatTTS 生成的音频，可以从中提取一段作为参考

# 步骤 1: 从旧音频中截取 3-10 秒片段
ffmpeg -i old_audio.wav -ss 0 -t 5 reference.wav

# 步骤 2: 手动转录这 5 秒音频的文本
REFERENCE_TEXT="（音频中的确切文本）"

# 步骤 3: 克隆声音
tts-cli --mode clone \
  --reference-audio reference.wav \
  --reference-text "$REFERENCE_TEXT" \
  --text "测试新声音" \
  --save-speaker cloned_voice.qwen-voice \
  --output test.wav
```

### 3. 声音设计（自然语言描述）

```bash
# 使用自然语言描述想要的声音特质
tts-cli --mode design \
  --voice-description "温暖的老年男性声音，带有轻微的英国口音" \
  --text "晚上好，女士们先生们" \
  --output speech.wav

# 其他示例
tts-cli --mode design \
  --voice-description "年轻有活力的女性声音" \
  --text "大家好！" \
  --output hello.wav
```

### 4. 风格控制

```bash
# 使用 --instruct 参数控制说话风格
tts-cli --text "我太兴奋了！" \
  --speaker Aiden \
  --instruct "speak with enthusiasm" \
  --output excited.wav

# 其他风格示例
tts-cli --text "请仔细听" \
  --speaker Ryan \
  --instruct "speak slowly and clearly" \
  --output clear.wav
```

## 批量迁移脚本

如果你有多个旧的 `.pt` 文件需要迁移，可以使用以下脚本批量处理：

```bash
#!/bin/bash

# batch_migrate.sh - 批量迁移 .pt 文件到 .qwen-voice

# 使用方法:
# 1. 为每个 .pt 文件准备对应的参考音频和转录文本
# 2. 在下方数组中填写信息
# 3. 运行脚本: bash batch_migrate.sh

# 配置迁移列表（根据你的实际情况修改）
declare -A MIGRATIONS
MIGRATIONS["narrator.pt"]="narrator_ref.wav|这是旁白的声音样本"
MIGRATIONS["character1.pt"]="char1_ref.wav|这是角色一的声音"
MIGRATIONS["character2.pt"]="char2_ref.wav|这是角色二的声音"

for pt_file in "${!MIGRATIONS[@]}"; do
    info="${MIGRATIONS[$pt_file]}"
    ref_audio=$(echo $info | cut -d'|' -f1)
    ref_text=$(echo $info | cut -d'|' -f2)

    # 生成新的 .qwen-voice 文件名
    qwen_file="${pt_file%.pt}.qwen-voice"

    echo "Migrating $pt_file -> $qwen_file"

    tts-cli --mode clone \
        --reference-audio "$ref_audio" \
        --reference-text "$ref_text" \
        --text "测试" \
        --save-speaker "$qwen_file" \
        --output "test_${qwen_file%.qwen-voice}.wav" \
        --quiet

    if [ $? -eq 0 ]; then
        echo "✓ Successfully migrated $pt_file"
    else
        echo "✗ Failed to migrate $pt_file"
    fi
done

echo "Migration complete!"
```

## 常见问题

### Q: 我的 .pt 文件无法使用，报错怎么办？

A: ChatTTS 的 `.pt` 文件**不兼容** Qwen-TTS。请参考上述"声音克隆工作流"重新创建音色文件。

错误信息示例：
```
ValueError: ChatTTS speaker files (.pt) are not compatible with Qwen-TTS.

To clone a voice:
  tts-cli --mode clone --reference-audio <file.wav> \
          --reference-text '<transcript>' \
          --save-speaker voice.qwen-voice
```

### Q: 克隆出的声音质量不好怎么办？

A: 改善克隆质量的技巧：
- ✅ 使用 3-10 秒的清晰参考音频（不要太短或太长）
- ✅ 确保参考音频背景噪音少
- ✅ 转录文本必须**完全匹配**参考音频内容
- ✅ 使用高质量音频格式（推荐 WAV）
- ✅ 参考音频中只有单人说话

### Q: 多语言支持如何使用？

A: Qwen-TTS 支持 10 种语言，使用 `--language` 参数指定：

```bash
# 中文
tts-cli --text "你好" --speaker Vivian --language zh --output zh.wav

# 英语
tts-cli --text "Hello" --speaker Ryan --language en --output en.wav

# 日语
tts-cli --text "こんにちは" --speaker Ono_Anna --language ja --output ja.wav

# 法语
tts-cli --text "Bonjour" --speaker Ryan --language fr --output fr.wav

# 自动检测（默认）
tts-cli --text "Hello world" --speaker Ryan --language auto --output auto.wav
```

支持的语言: `zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`, `auto`

### Q: 如果我不想克隆声音，有默认音色吗？

A: 有！v0.2.0 提供 9 种高质量预设音色：

```bash
# 直接使用预设音色（无需克隆）
tts-cli --text "Hello" --speaker Ryan --output hello.wav
```

参考上面的"可用音色列表"选择合适的音色。

### Q: 升级到 v0.2.0 后，我的脚本需要修改吗？

A: 大部分脚本**不需要修改**，但建议：

1. **如果使用 .pt 文件**: 需要替换为 .qwen-voice 文件
2. **如果依赖随机音色**: 现在默认使用 Ryan，可能需要指定 `--speaker`
3. **如果调整 max-length**: 新默认值是 1000，可能需要调整

示例迁移：

```bash
# 旧脚本（v0.1.x）
tts-cli --file input.txt --speaker voice.pt --max-length 500

# 新脚本（v0.2.0+）
tts-cli --file input.txt --speaker voice.qwen-voice --max-length 1000
```

## 获取帮助

如遇到迁移问题，请：

1. 查看完整文档: [README.md](README.md)
2. 运行 `tts-cli --help` 查看所有可用参数
3. 提交 Issue: [GitHub Issues](https://github.com/yahaa/tts-cli/issues)
