# LLM Router

[中文说明 (Chinese README)](README_zh.md)

A router server for OpenAI‑compatible API requests.

## Features

- Supports OpenAI, Anthropic, and Gemini compatible API endpoints
- Converts requests/responses across OpenAI, Anthropic, and Gemini
- Model selection via jq expressions (full jq syntax supported)

## CLI

```
Usage: llm-router [OPTIONS]

Options:
  -i, --ip <IP>                [default: 0.0.0.0]
  -p, --port <PORT>            [default: 8000]
  -c, --config <CONFIG>        Path to config file [default: config.yaml]
  -t, --token <TOKEN>
  -l, --log-level <LOG_LEVEL>  trace, debug, info, warn, error [default: warn]
      --log-file <PATH>        Also write logs to this file (max 10MB)
      --proxy <PROXY>          socks and http proxy, e.g. socks5://192.168.0.2:10080
      --check                  Check all models in config and exit
  -h, --help                   Print help
```

### Examples

```
llm-router --ip 0.0.0.0 --port 8000 --config config.yaml --token your-secret-token

# Check availability of all models (without starting the server)
llm-router --config config.yaml --check
```

## API Usage

```
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
          { "text": "Hello" }
        ]
      }
    ]
  }'
```

## Configuration

### Config file

When calling APIs, use the value of the `name` field under `router_settings.model_groups` as the model name.

The project uses a YAML config file `config.yaml` with the following structure:

```yaml
model_list:
  - model_name: model1
    llm_params:
      api_type: openai  # openai, anthropic, gemini
      model: qwen3-8b
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: sk-1234
      rewrite_header: '{"X-Request-ID": "12345"}' # optional
      rewrite_body: '{"enable_thinking": false, "max_tokens": 8192}' # optional

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
  strategy: roundrobin  # roundrobin, random, leastconn
  model_groups:
    - name: gpt_models # the name used when calling APIs
      models:
        - name: model1 # refers to model_list.model_name; weight defaults to 100
          selector: '.tools | length == 0'
        - name: model2
          weight: 100
          selector: '.tools | length > 0' # select only when jq evaluates to true; see https://jqlang.org/manual/

    - name: gpt_models2
      models:
        - name: model1
        - name: model3
```

`router_settings` defines routing strategies. When making requests, use the `name` defined under `router_settings.model_groups` as the model name.

For `roundrobin`, `random`, and `leastconn`, weights are applied. On each failure, a model’s weight is halved. When a model’s weight reaches 0, it will not be selected unless it’s the only remaining model.

If `selector` is empty, the model is eligible for selection. If set, the jq expression is evaluated against the request body; the model is only eligible when the result is `true`. Any other result excludes the model.
