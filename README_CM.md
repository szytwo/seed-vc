## 安装

```

docker build -t seed-vc:latest .  # 构建镜像
docker load -i seed-vc.tar # 导入镜像
docker save -o seed-vc.tar seed-vc:latest # 导出镜像
docker-compose up -d # 后台运行容器

```