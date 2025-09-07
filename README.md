# LLM Router

一个用于路由openai compatible api请求的服务器，支持动态配置重载。

## 功能特性

- 支持 OpenAI、Anthropic 兼容的 API 接口
- 支持 OpenAI、Anthropic 转换
- 动态配置重载，无需重启服务
- 模型别名分组功能
- 简单的认证机制


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

```bash
curl -X GET http://localhost:4000/v1/models -H "Authorization: Bearer your-secret-token"


curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "model": "gpt_models",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'


curl http://localhost:4000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-token" \
  -d '{
    "model": "gpt_models",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```


## 配置

### 配置文件

项目使用 YAML 格式的配置文件 `config.yaml`，包含以下部分：

```yaml
model_list:
  - model_name: model1
    llm_params:
      api_type: openai # openai, anthropic
      model: model
      api_base: https://llm.api:8000/v1
      api_key: sk-1234
  - model_name: model2
    llm_params:
      api_type: anthropic # openai, anthropic
      model: model
      api_base: https://llm.api:8000
      api_key: sk-1234

router_settings:
  strategy: roundrobin # roundrobin(平滑加权轮询),random(随机)
  model_groups:
    - name: gpt_models # 调用api的时候使用的名称
      models:
        - name: model1 # model_list中定义的model_name。weight默认100
        - name: model2
          weight: 100
    - name: gpt_models2
      models:
        - name: model1
```

`model_list`每个模型配置包含：
- `model_name`: router_settings使用的模型名称
- `llm_params`: 模型的实际参数
  - `api_type`: 接口类型 # openai, anthropic
  - `model`: 提供商处的实际模型名称
  - `api_base`: API 基础 URL
  - `api_key`: API 密钥

`router_settings` 定义路由策略。请求的时候模型名称使用router_settings中定义的name

### 配置重载

配置文件支持热重载。当 `config.yaml` 文件被修改时，服务会自动检测并重新加载配置，无需重启服务。