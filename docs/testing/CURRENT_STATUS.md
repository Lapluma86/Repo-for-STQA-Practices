# 当前测试状态（2026-09-18）

目录清理前后均执行全部64项测试，结果一致：**44通过、20失败、0跳过、0执行错误**，用例身份和每项通过/失败结果完全一致。此次未修改被测源码、测试断言或冻结指纹。

| 模块 | 用例数 | 通过 | 失败 | 跳过 | 执行错误 |
| --- | --- | --- | --- | --- | --- |
| module1 | 48 | 38 | 10 | 0 | 0 |
| module2 | 16 | 6 | 10 | 0 | 0 |

使用AI_common指定解释器，清理后运行命令：

```powershell
$testTemp = Join-Path $env:TEMP ('stqa-tests-' + [guid]::NewGuid().ToString('N'))
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/ -p no:cacheprovider --basetemp=$testTemp --junitxml=docs/evidence/cleanup-check/tests.xml
```

本机默认pytest临时目录以及受限权限下新建的临时目录曾出现拒绝访问，前置尝试未完成有效测试。上述有效前后对比在获准的隔离限制外使用专用临时目录运行，退出码均为1（已存在的测试失败）。

[清理后日志](../evidence/cleanup-check/tests.log) · [JUnit](../evidence/cleanup-check/tests.xml) · [清理前JUnit](../evidence/cleanup-check/before.xml) · [逐项对比摘要](../evidence/cleanup-check/comparison.json)

三个重点缺陷仍未修复，模块二也保留现有失败；参见[缺陷说明](DEFECTS.md)。119个源码、脚本、配置及测试文件逐字节未变，完整源码冻结校验通过。

覆盖率未在本次重测；2026-09-15模块一证据中的70.28%仅对应三个指定模块，不代表本次全64项或整个项目。完整训练、真实数据精度、GPU性能和办公附件重生成不在此次结构清理范围内。
