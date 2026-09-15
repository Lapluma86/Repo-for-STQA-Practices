# Rail Lab：单页演示工具

## 启动

在 PowerShell 中运行（终端保持打开）：

```powershell
Set-Location 'D:\projects\Repo-for-STQA-Practices'
& 'D:\Anaconda\envs\AI_common\python.exe' -X utf8 tools/rail_demo/server.py
```

浏览器打开 http://127.0.0.1:8765 。Ctrl+C 停止服务。如果端口被占用，命令末尾加 `--port 8766`，浏览器也改为对应端口。不需要 npm、额外 Web 框架、模型权重或真实数据。依赖沿用项目 AI_common 环境。

## 操作

1. 默认选择异常样本、view_id=1、patch_size=160、stride=120、输出256、zscore。页面自动读取第1个patch。
2. 点击右箭头切到第2个patch：裁剪框的 y 从0变为120，可看到第一处合成缺陷；三列下方图像来自数据集实际返回的张量。
3. 切换 patch 边长、步长、归一化后，点击“读取并展示”。切换参数会把旧结果变灰，重新读取后回到第1个patch。
4. 切换正常样本：label=0，无GT文件，数据集返回全零GT。
5. 输入view_id=0或9：显示被测类真实的ValueError。恢复1重新读取。
6. 选择深度TIFF损坏：先构造并扫描有效数据，再损坏临时TIFF，调用__getitem__，显示真实FileNotFoundError。切回异常样本即可恢复。

原图320×640。源图裁剪框用于解释位置；裁剪、resize、归一化和GT读取由 SteelRailWay/datasets/rail_dataset.py 中的 RailDualModalDataset 完成。

RGB输出为了观看而逆转ImageNet标准化；深度预览对每幅图独立拉伸到0–255，因此归一化差异应通过下方真实数值比较。GT预设标注不代表模型预测，正常样本的全零GT来自缺省行为。label为整帧标签，即使某个patch没有缺陷像素，异常帧的label仍为1。

当前被测类test扫描硬编码cam1–6，view_id=7/8虽然通过参数验证但得到空集，页面如实说明。此工具不修复这个行为，也不将其自动计作新课程缺陷。

所有输入为合成数据，每次请求在本目录已忽略的.runtime中建立独立临时样本并自动清理，不修改用户数据。工具仅监听127.0.0.1，不上传内容。页面不运行训练、推理或pytest。

## 约6分钟录屏脚本

建议OBS屏幕捕获＋麦克风，1080p、30fps，浏览器最大化；先试录20秒检查字号和声音。PPT只用于开头或结尾。准备一个单独终端执行测试，避免关闭服务终端。

| 时间 | 操作 | 讲解示例 |
|---|---|---|
| 0:00–0:25 | 展示页面标题与输入区 | 本次选择钢轨项目的数据处理等模块开展测试；这里用合成样本直观展示真实数据加载接口。 |
| 0:25–1:30 | 默认样本，切到第2、第3个patch | 三路输入共用坐标系；垂直滑窗、水平居中，GT使用最近邻缩放。裁剪结果来自实际被测类。 |
| 1:30–2:00 | 切换minmax，重新读取并展示张量区 | 预览灰度经过显示映射；真实归一化范围看这里的min/max。 |
| 2:00–2:30 | 正常样本，再输入view_id=9，恢复1 | 这是正常场景和非法边界输入；展示预期错误，不代表测试失败。 |
| 2:30–3:15 | 选择损坏TIFF，再恢复异常样本 | 原缺陷会触发AttributeError，修复后给出包含路径的FileNotFoundError；页面展示当前修复后的行为。 |
| 3:15–4:30 | 切换终端执行模块一测试 | 展示命令、执行过程、通过/失败/跳过数量，并说明CUDA跳过不能算通过。 |
| 4:30–5:30 | 打开本次HTML报告、历史修复前日志 | 对应输入、断言和结果。历史日志标注为修复前证据，当前页面不是修复前版本。 |
| 5:30–6:00 | 总结 | 覆盖率仅针对三个指定模块；本次不证明完整TRD精度和端到端验收。 |

## 录制时生成独立测试证据

下面先创建带时间戳的独立目录，再运行模块一，避免覆盖历史证据。测试结果以当次实际输出为准，不在页面硬编码历史通过数。

```powershell
Set-Location 'D:\projects\Repo-for-STQA-Practices'
$demoEvidence = Join-Path 'tests/artifacts' ('video-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $demoEvidence | Out-Null
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/ -v --cov --cov-config=.coveragerc --cov-report=term-missing "--html=$demoEvidence/report.html" --self-contained-html "--junitxml=$demoEvidence/results.xml" 2>&1 | Tee-Object -FilePath "$demoEvidence/run.log"
Start-Process (Join-Path $demoEvidence 'report.html')
```

可先演练单个缺陷回归：

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/blackbox/test_regression_depth.py -v
```

修复前证据可查 tests/logs/fixes-20260915/before.log；录制前核对日志版本与当前代码。AI辅助工具及测试成果来源应据实说明，不包装成人工独立完成。

## 工具自检

服务启动后运行；使用8766时将末尾参数改为8766：

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -X utf8 tools/rail_demo/check_demo.py 8765
```

检查5个patch的GT逐像素对齐、三种归一化、9种裁剪组合、正常样本、非法输入、空集、损坏文件及异常恢复。这些工具验证不计入课程模块一用例数。

Codex本机沙箱已知存在Windows临时目录ACL限制；若在Codex内遇到WinError 5，可在普通用户PowerShell中执行同一启动命令，无需更改系统权限或使用管理员账号。
