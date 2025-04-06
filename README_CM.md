## 安装

```

docker build -t seed-vc:1.0 .  # 构建镜像
docker load -i seed-vc-1.0.tar # 导入镜像
docker save -o seed-vc-1.0.tar seed-vc:1.0 # 导出镜像
docker-compose up -d # 后台运行容器

```