# 博客日常维护手册

这是我的博客管理备忘录。以后每次更新文章，按照这个流程操作即可。

## 一、 日常更新流程 (4步法)

### 第 1 步：创作 (Write)
在 `docs` 文件夹下新建 `.md` 文件，使用 Markdown 语法编写内容。

### 第 2 步：预览 (Preview)
在 PyCharm 的 Terminal 中运行以下命令，启动本地服务器：

```bash
mkdocs serve
```

* 打开浏览器访问：[http://127.0.0.1:8000]
* 检查排版和内容，确认无误后按 `Ctrl + C` 停止服务。

### 第 3 步：发布 (Deploy)

将生成的静态网页推送到 GitHub Pages，让网站上线：

```bash
mkdocs gh-deploy
```

### 第 4 步：备份源码 (Backup)

**重要！** 这一步是为了防止源码丢失。将本地的 Markdown 原稿同步到 GitHub 仓库：

```bash
git add .
git commit -m "日常更新：新增文章"
git push
```

---

## 二、 进阶配置：管理菜单栏

如果要调整文章在左侧导航栏的顺序，需要修改根目录下的 `mkdocs.yml` 文件。

找到 `nav` 配置项（如果没有就自己加上）：

```yaml
site_name: 我的 Python 学习笔记
theme:
  name: material

# 导航栏配置
nav:
  - 首页: index.md
  - 博客指南: guide.md  # <--- 这就是我现在这篇文章
  - Python笔记:
      - 变量与类型: python_basic.md
  - 随想: thoughts.md
```

> **注意**：修改 `mkdocs.yml` 之后，通常需要重启 `mkdocs serve` 才能看到效果。