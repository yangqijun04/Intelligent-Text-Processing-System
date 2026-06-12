# Dify 智能文本处理系统

基于 Dify 1.13.3 + Ollama AI 的全离线智能文本分析平台，解压即用。

---

## 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `images.tar` | 13GB | 13 个 Docker 镜像（Dify 全套 + Ollama + bridge） |
| `dify-1.13.3.tar.gz` | 5GB | Dify 项目目录（含 AI 模型、数据库、配置） |

---

## 一、环境要求

### 硬件

| 要求 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Ubuntu 20.04+ | Ubuntu 22.04+ |
| 内存 | 16GB | 32GB+ |
| 磁盘空闲 | 30GB | 50GB+ |
| GPU（可选） | NVIDIA 6GB 显存 | NVIDIA 12GB+ 显存 |

### 软件

| 要求 | 版本 |
|------|------|
| Docker | 24.0+ |
| Docker Compose | v2（docker内置） |
| GPU 驱动 | NVIDIA 535+ |
| NVIDIA Container Toolkit | 1.14+ |

如未安装 NVIDIA Container Toolkit ，具体安装方法可参见附录。

---

## 二、部署步骤

```bash
# 1. 解压 dify-1.13.3 文件夹
tar -xzf dify-1.13.3.tar.gz

# 2. 修复文件权限
sudo chown -R 1001:1001 dify-1.13.3/docker/volumes/app/storage/

cd dify-1.13.3/docker

# 3. 导入所有 Docker 镜像
docker load -i ../../images.tar

# 4. 启动全部服务
docker compose --profile postgresql --profile weaviate up -d

# 5. 等待服务就绪
docker compose ps
```

也可使用一键部署脚本：将 `images.tar`、`dify-1.13.3.tar.gz`、`deploy.sh` 放在同一目录，然后在该目录下执行 `bash deploy.sh`一条指令即可。

---

## 三、访问系统

| 地址 | 用途 |
|------|------|
| `http://localhost` | Dify 控制台（管理 workflow、模型） |
| **`http://localhost:8088`** | **智能处理前端（上传文件、AI 分析）** |


### 首次使用

**第一步：登录 Dify 控制台**

打开 `http://localhost`，使用管理员账号登录：

| 字段 | 值 |
|------|-----|
| 邮箱 | `yangqijun23@nudt.edu.cn` |
| 密码 | `admin123` |

**第二步：开始使用智能文本处理系统**

打开 `http://localhost:8088`：
1. 上传文档（支持 .txt / .docx / .pdf / .md 等）或文件夹
2. 输入处理指令或点击上方快捷按钮
3. 点击 **「启动智能处理」** 等待结果

如需修改 workflow 或模型配置，在已登录的 `http://localhost` 控制台操作即可。

---


## 四、无 GPU 用户须知

如果你的机器**没有 NVIDIA GPU**，启动前必须去掉 GPU 配置：

编辑 `/dify-intelligence-pack/dify-1.13.3/docker/docker-compose.override.yaml`，**删除**以下 5 行：

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

删除后 Ollama 自动切换为 CPU 推理（会慢一些，但功能正常）。

---


## 五、常见问题

### Q: 端口被占用

**检测端口占用**：
```bash
sudo lsof -i :80      # 检测 Nginx 端口
sudo lsof -i :8088    # 检测 bridge 端口
sudo lsof -i :11434   # 检测 Ollama 端口
```

> 有输出即被占用。`docker-pr` 表示被 Docker 容器占用（正常），其他进程名表示被宿主机程序占用（需停止或换端口）。

**修改 Nginx 端口**：编辑 `.env`，改 `EXPOSE_NGINX_PORT=8080` 

**修改 bridge 端口**：编辑 `docker-compose.override.yaml`，改 `8088:8088` 为 `9090:8088` 

**修改 Ollama 端口**：执行 `sudo systemctl stop ollama.service` 命令



**修改后重启服务**：
```bash
cd dify-1.13.3/docker
docker compose --profile postgresql --profile weaviate down
docker compose --profile postgresql --profile weaviate up -d
```

**修改端口后访问地址会随之变化**：
| 地址 | 用途 |
|------|------|
| `http://localhost：8080` | Dify 控制台 |
| `http://localhost:9090` | 智能处理前端（上传文件、AI 分析） |

### Q: 首次上传文件失败 / 提示权限错误

执行以下命令后重启服务：

```bash
sudo chown -R 1001:1001 dify-1.13.3/docker/volumes/app/storage/
chmod -R 755 dify-1.13.3/docker/volumes/app/storage/
cd dify-1.13.3/docker
docker compose --profile postgresql --profile weaviate down
docker compose --profile postgresql --profile weaviate up -d
```

### Q: 如何停止 / 重启

```bash
cd dify-1.13.3/docker

# 停止
docker compose --profile postgresql --profile weaviate down

# 重启
docker compose --profile postgresql --profile weaviate up -d

# 查看日志
docker compose logs -f ollama     # AI 推理日志
docker compose logs -f bridge     # 前端后台日志
```

---

## 附录：安装 NVIDIA Container Toolkit

如果使用 GPU 推理，必须先安装此工具。未安装时 Ollama 自动切换为 CPU 推理。

官方安装指南：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Ubuntu 快速安装：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证安装成功：

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
