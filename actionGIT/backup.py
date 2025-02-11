name: Sync with Upstream

on:
  schedule:
    - cron: '0 0 * * *'  # 每天 00:00 运行（UTC 时间）
  workflow_dispatch:  # 允许手动触发

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout the repository
        uses: actions/checkout@v3
        with:
          persist-credentials: false

      - name: Configure Git
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"

      - name: Add Upstream and Fetch
        run: |
          git remote add upstream https://github.com/原仓库的所有者/原仓库.git
          git fetch upstream

      - name: Merge upstream changes
        run: |
          git checkout main
          git merge upstream/main
          git push origin main
