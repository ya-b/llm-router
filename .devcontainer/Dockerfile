FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt update && \
    apt install -y --no-install-recommends apt-transport-https ca-certificates git curl build-essential

ENV RUSTUP_UPDATE_ROOT=https://mirrors.aliyun.com/rustup/rustup \
    RUSTUP_DIST_SERVER=https://mirrors.aliyun.com/rustup \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN mkdir -p ~/.cargo/ && \
    echo '[source.crates-io]\nreplace-with = "aliyun"\n[source.aliyun]\nregistry = "sparse+https://mirrors.aliyun.com/crates.io-index/"' > ~/.cargo/config.toml
RUN curl -sSf https://mirrors.aliyun.com/repo/rust/rustup-init.sh | bash -s -- -y --no-modify-path && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME && \
    rustup --version && \
    cargo --version && \
    rustc --version

CMD ["sleep", "infinity"]