---
title: vers
emoji: 🔄☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Vertex AI to OpenAI Adapter

OpenAI API 格式到 Gemini/Vertex AI 的适配器服务。

## 新功能：竞速模式 (Race Mode)

竞速模式可以并发发送多个相同的 LLM 请求，然后使用第一个成功返回的结果，自动取消其他请求。这可以显著提高服务的可靠性和响应速度。

### 启用竞速模式

通过环境变量启用：

```bash
RACE_MODE=true                  # 启用竞速模式
RACE_CONCURRENT_COUNT=3         # 并发请求数量（默认 3）

# 配置多个 API Key（用逗号分隔），竞速时会自动轮换使用
EXPRESS=key1,key2,key3          # 多个 Express API Key
# 或使用多个 SA 凭证文件，系统会自动轮换
```

### 工作原理

1.  **流式请求 (Streaming)**: 并发发送多个**真流式**请求，选择**第一个返回数据块最快**的请求，并持续输出其后续内容，其他请求将被取消。这确保了最快的“第一反应速度”。
2.  **非流式请求 (Non-Streaming)**: 并发发送多个请求，并**等待所有请求完成**，然后选择返回内容**字符串最长**的那个结果。这牺牲了速度以换取更丰富的内容。
3.  **自动Key轮换**：每个并发请求使用不同的 API Key 或 SA 凭证，避免单个key的速率限制，提高成功率。

### 使用示例

#### Docker Compose

```yaml
services:
  adapter:
    image: your-image
    environment:
      - RACE_MODE=true
      - RACE_CONCURRENT_COUNT=3
      - PASSWORD=your_password
      # 配置多个 Express API Key（推荐）
      - EXPRESS=key1,key2,key3
      # 或使用多个 SA 凭证
      # - JSON=cred1_json,cred2_json,cred3_json
```

#### Docker 命令行

```bash
docker run \
  -e RACE_MODE=true \
  -e RACE_CONCURRENT_COUNT=3 \
  -e EXPRESS=key1,key2,key3 \
  -e PASSWORD=your_password \
  your-image
```

### 日志输出

启用后，您会在日志中看到：

```
INFO: Race mode ENABLED - will make 3 concurrent requests and use first successful one
INFO: [ClientFactory] Created Gemini Express client (key index: 0)
INFO: [ClientFactory] Created Gemini Express client (key index: 1)
INFO: [ClientFactory] Created Gemini Express client (key index: 2)
INFO: Race mode - Got successful result, cancelling 2 pending tasks
```

可以看到每个并发请求使用了不同的 key（index: 0, 1, 2）。

### 注意事项

- 竞速模式会增加 API 调用次数，请确保您的配额充足
- **建议配置多个 API Key**：并发数应该 ≤ 可用 Key 数量，这样每个请求都能用不同的 key，避免速率限制
- 建议并发数设置为 2-5 之间，过高可能造成资源浪费
- 自动模式（auto mode）下不启用竞速，避免重复尝试
- 流式请求会自动切换到假流式模式以支持竞速
- **Key 轮换**：系统会自动轮换使用不同的 key，每个并发请求用不同的 key

### 适用场景

- API 不稳定，经常出现超时或错误
- 需要更快的首次响应时间
- **拥有多个服务账号或 API Key**（强烈推荐，可充分发挥竞速优势）
- 对响应可靠性要求高的生产环境
- 单个 key 有速率限制，需要分散请求

### 最佳实践

如果你有 3 个 Express API Key：

```bash
# 设置 3 个并发请求，每个用不同的 key
RACE_MODE=true
RACE_CONCURRENT_COUNT=3
EXPRESS=key1,key2,key3
```

这样：
- ✅ 每个请求用不同的 key，避免单个 key 速率限制
- ✅ 3 个请求同时发出，谁先成功用谁的
- ✅ 成功率大幅提升
- ✅ 响应速度更快

