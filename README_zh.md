# LLM Router

[English README](README.md)

一个用于路由openai compatible api请求的服务器。

## 功能特性

- 支持 OpenAI、Anthropic、Gemini 兼容的 API 接口
- 支持 OpenAI、Anthropic、Gemini 互相转换
- 基于jq表达式选择模型，支持jq语法

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
      --log-file <PATH>        同时将日志写入该文件（最大 10MB）
      --proxy <PROXY>          socks and http proxy, example: socks5://192.168.0.2:10080
      --check                  Check all models in config and exit
  -h, --help                   Print help
```

### 使用示例

```bash
llm-router --ip 0.0.0.0 --port 8000 --config config.yaml --token your-secret-token

# 检查配置中所有模型的可用性（不启动服务）
llm-router --config config.yaml --check
```


## API 使用

```bash
curl -X GET http://localhost:8000/v1/models -H "Authorization: Bearer your-secret-token"


curl "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "model": "gpt_models",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'


curl "http://localhost:8000/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-token" \
  -d '{
    "model": "gpt_models",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'


curl "http://localhost:8000/v1beta/models/gpt_models:generateContent?alt=sse&key=your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {
            "text": "Hello"
          }
        ]
      }
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
      api_type: openai  # openai, anthropic, gemini
      model: qwen3-8b
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: sk-1234
      rewrite_header: '{"X-Request-ID": "12345"}' # 非必填
      rewrite_body: '{"enable_thinking": false, "max_tokens": 8192}' # 非必填

  - model_name: model2
    llm_params:
      api_type: anthropic
      model: glm-4.5-flash
      api_base: https://open.bigmodel.cn/api/anthropic
      api_key: sk-1234

  - model_name: model3
    llm_params:
      api_type: gemini
      model: gemini-2.5-pro
      api_base: https://generativelanguage.googleapis.com/v1beta
      api_key: sk-1234

router_settings:
  strategy: roundrobin  # roundrobin,random,leastconn
  model_groups:
    - name: gpt_models # 调用api的时候使用的名称
      models:
        - name: model1 # model_list中定义的model_name。weight默认100
          selector: '.tools | length == 0'
        - name: model2
          weight: 100
          selector: '.tools | length > 0' # jq表达式返回true时才可能会选择该模型，规则参考https://jqlang.org/manual/

    - name: gpt_models2
      models:
        - name: model1
        - name: model3
```

`router_settings` 定义路由策略。请求的时候模型名称使用router_settings中定义的name
roundrobin,random,leastconn 这三种策略都使用weight加权。每次请求失败，weight降低1/2，weight为0时，除非仅剩当前1个模型，否则该模型将不会被使用。

selector 为空时会选择该模型。不为空时：根据jq表达式匹配请求体中内容，仅当结果为true时才会选择该模型。其他任何值都不会选择该模型。
