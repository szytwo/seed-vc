# 使用 PyTorch 官方 CUDA 12.1 运行时镜像
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 设置容器内工作目录为 /workspace
WORKDIR /workspace

# 替换软件源为清华镜像
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 防止交互式安装，完全不交互，使用默认值
ENV DEBIAN_FRONTEND=noninteractive
# 设置时区
ENV TZ=Asia/Shanghai

# 更新源并安装依赖
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc g++ make \
    xz-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# RUN gcc --version

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 安装 Python 3.10.16 到自定义路径
# 使用 update-alternatives 设置 Python 3.10.16 为默认 Python 版本
# https://mirrors.aliyun.com/python-release/
COPY wheels/linux/Python-3.10.16.tgz .

RUN tar -xzf Python-3.10.16.tgz \
    && cd Python-3.10.16 \
    && ./configure --prefix=/usr/local/python3.10.16 --enable-optimizations \
    && make -j$(nproc) && make altinstall \
    && cd .. \
    && rm -rf Python-3.10.16 Python-3.10.16.tgz \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/python3.10.16/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/python3.10.16/bin/pip3.10 1

# 验证 Python 和 pip 版本
# RUN python --version && pip --version

# 下载并解压 FFmpeg
# https://www.johnvansickle.com/ffmpeg
COPY wheels/linux/ffmpeg-6.0.1-amd64-static.tar.xz .

RUN tar -xJf ffmpeg-6.0.1-amd64-static.tar.xz -C /usr/local \
    && mv /usr/local/ffmpeg-* /usr/local/ffmpeg \
    && rm ffmpeg-6.0.1-amd64-static.tar.xz

# 设置 FFmpeg 到环境变量
ENV PATH="/usr/local/ffmpeg:${PATH}"

# RUN ffmpeg -version

# 设置容器内工作目录为 /code
WORKDIR /code

# 将项目源代码复制到容器中
COPY . /code

# 升级 pip 并安装 Python 依赖：
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && rm -rf /wheels

# 暴露容器端口
EXPOSE 22
EXPOSE 80
EXPOSE 7869

# 容器启动时执行 api.py
# CMD ["python", "api.py"]
