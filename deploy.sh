#!/bin/bash
set -e

echo "=== Dify 智能文本处理系统 一键部署 ==="
echo ""

echo "[1/4] 解压项目…"
tar -xzf dify-1.13.3.tar.gz

echo "[2/4] 修复文件权限（请输入 sudo 密码）…"
sudo chown -R 1001:1001 dify-1.13.3/docker/volumes/app/storage/

echo "[3/4] 导入 Docker 镜像（约 2-5 分钟）…"
docker load -i images.tar

echo "[4/4] 启动服务（约 1-2 分钟）…"
cd dify-1.13.3/docker
docker compose --profile postgresql --profile weaviate up -d

sleep 8
docker compose ps

echo ""
echo "============================================"
echo "  部署完成！"
echo ""
echo "  Dify 控制台:     http://localhost"
echo "  智能处理前端:    http://localhost:8088"
echo ""
echo "  登录邮箱:        yangqijun23@nudt.edu.cn"
echo "  登录密码:        admin123"
echo "============================================"
