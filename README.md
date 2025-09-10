# LLM Router

一个用于路由openai compatible api请求的服务器，支持动态配置重载。

## 功能特性

- 支持 OpenAI、Anthropic 兼容的 API 接口
- 支持 OpenAI、Anthropic 互相转换
- 动态配置重载，无需重启服务


## 命令行参数

程序支持以下命令行参数：

```bash
Usage: llm-router [OPTIONS]

Options:
  -i, --ip <IP>                [default: 0.0.0.0]
  -p, --port <PORT>            [default: 8000]
  -c, --config <CONFIG>        Path to config file [default: config.yaml]
  -t, --token <TOKEN>          
  -l, --log-level <LOG_LEVEL>  trace, debug, info, warn, error [default: warn]
      --proxy <PROXY>          socks and http proxy, example: socks5://192.168.0.2:10080
  -h, --help                   Print help
```

### 使用示例

```bash
llm-router --ip 0.0.0.0 --port 8000 --config config.yaml --token your-secret-token
```


## API 使用

```bash
curl -X GET http://localhost:8000/v1/models -H "Authorization: Bearer your-secret-token"


curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "model": "gpt_models",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'


curl http://localhost:8000/v1/messages \
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

调用接口的时候，使用**router_settings.model_groups**中的**name**字段的值

项目使用 YAML 格式的配置文件 `config.yaml`，包含以下部分：


```yaml
model_list:
  - model_name: model1
    llm_params:
      api_type: openai  # openai, anthropic
      model: qwen3-8b
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: sk-1234
      rewrite_body: '{"enable_thinking": false, "max_tokens": 8192}' # 非必填
      
  - model_name: model2
    llm_params:
      api_type: openai
      model: glm-4.5-flash
      api_base: https://open.bigmodel.cn/api/paas/v4
      api_key: sk-1234
      rewrite_body: '{"metadata": null}'  # claude code通过openai接口调用glm的时候，metadata设为null

  - model_name: model3
    llm_params:
      api_type: anthropic # openai, anthropic
      model: glm-4.5-flash
      api_base: https://open.bigmodel.cn/api/anthropic
      api_key: sk-1234
      
router_settings:
  strategy: roundrobin  # roundrobin(平滑加权轮询),random(加权随机)
  model_groups:
    - name: gpt_models # 调用api的时候使用的名称
      models:
        - name: model1 # model_list中定义的model_name。weight默认100
        - name: model2 
          weight: 100

    - name: gpt_models2
      models:
        - name: model1
        - name: model3
```

`model_list`每个模型配置包含：
- `model_name`: router_settings使用的模型名称
- `llm_params`: 模型的实际参数
  - `api_type`: 接口类型 # openai, anthropic
  - `model`: 提供商处的实际模型名称
  - `api_base`: API 基础 URL
  - `api_key`: API 密钥
  - `rewrite_body`: 可选，会覆盖请求体里的参数

`router_settings` 定义路由策略。请求的时候模型名称使用router_settings中定义的name

### 配置重载

配置文件支持热重载。当 `config.yaml` 文件被修改时，服务会自动检测并重新加载配置，无需重启服务。