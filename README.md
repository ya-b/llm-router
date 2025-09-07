# LLM Router

一个用于路由openai compatible api请求的服务器，支持多个模型提供商和动态配置重载。

## 功能特性

- 支持 OpenAI 兼容的 API 接口
- 动态配置重载，无需重启服务
- 模型别名分组功能
- 简单的认证机制


## 配置

### 配置文件

项目使用 YAML 格式的配置文件 `config.yaml`，包含以下部分：

```yaml
model_list:
  - model_name: your_model_name
    llm_params:
      model: model
      api_base: https://llm.api:8000/v1
      api_key: sk-1234
router_settings:
  model_group_alias: '{"model_name_alias": ["your_model_name"]}'
```

`model_list`每个模型配置包含：
- `model_name`: 在路由中使用的模型名称
- `llm_params`: 模型的实际参数
  - `model`: 提供商处的实际模型名称
  - `api_base`: API 基础 URL
  - `api_key`: API 密钥

`model_group_alias` 是一个 JSON 字符串，定义了模型别名到实际模型名称的映射。当请求使用别名时，路由器会将其映射到对应的实际模型。

### 配置重载

配置文件支持热重载。当 `config.yaml` 文件被修改时，服务会自动检测并重新加载配置，无需重启服务。

## 命令行参数

程序支持以下命令行参数：

```bash
Usage: llm-router.exe [OPTIONS]

Options:
  -i, --ip <IP>                [default: 0.0.0.0]
  -p, --port <PORT>            [default: 4000]
  -c, --config <CONFIG>        [default: config.yaml]
  -t, --token <TOKEN>
  -l, --log-level <LOG_LEVEL>  [default: warn]
  -h, --help                   Print help
```

### 使用示例

```bash
llm-router --ip 0.0.0.0 --port 4000 --config config.yaml --token your-secret-token
```


## API 使用

发送 POST 请求到 `/v1/chat/completions` 端点：

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "model": "model_name_alias",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

### 健康检查

发送 GET 请求到 `/health` 端点：

```bash
curl http://localhost:4000/health
```
