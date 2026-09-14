# SteelRailWay 测试

模块一测试脚本与运行方式见仓库根目录 [README.md](../../README.md)。

正式交付文档：

- [测试用例清单](../../docs/module1/测试用例清单.xlsx)
- [缺陷清单](../../docs/module1/缺陷清单.docx)
- [测试报告](../../docs/module1/测试报告.docx)

```bash
cd SteelRailWay
pip install -r tests/requirements-test.txt
pytest tests/manual/phase1/ -v
```

本目录下其余 Markdown（`TEST_REPORT.md`、`BUG_FIX_REPORT.md` 等）是整理前的工作笔记，以 `docs/module1/` 三份正式文档为准。
