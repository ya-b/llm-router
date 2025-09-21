##############################
# Builder stage (Alpine 3.22)
##############################
FROM alpine:3.22 AS builder

# Install build dependencies and Rust toolchain
RUN apk add --no-cache \
    build-base \
    rust \
    cargo \
    git \
    pkgconfig \
    openssl-dev

WORKDIR /app

# Leverage build cache: build deps first
COPY . /app
RUN cargo build --release --locked


##############################
# Runtime stage (Alpine 3.22)
##############################
FROM alpine:3.22 AS runtime

# Runtime dependencies: SSL certs and GCC unwind library
RUN apk add --no-cache \
    ca-certificates \
    libgcc

WORKDIR /app

# Copy binary
COPY --from=builder /app/target/release/llm-router /usr/local/bin/llm-router

# Copy default config (can be overridden by bind mount)
COPY config.yaml /app/config.yaml

EXPOSE 8000

ENTRYPOINT ["llm-router"]
CMD ["--ip","0.0.0.0","--port","8000","--config","/app/config.yaml"]
