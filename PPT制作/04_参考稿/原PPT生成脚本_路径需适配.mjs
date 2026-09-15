import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation} from 'file:///C:/Users/sundowner/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='D:/projects/Repo-for-STQA-Practices'; const build=root+'/tests/artifacts/delivery-build';
const p=Presentation.create({slideSize:{width:1280,height:720}});
const content=[
['钢轨缺陷检测模块测试','软件测试与质量保证 模块一\n成员A  成员B  成员C\n2026年9月15日'],
['被测对象与边界','RGB与深度双模态数据进入特征重建检测流程\n本次选择数据加载、蒸馏损失、工程指标\n使用合成图像验证代码行为\n完整模型精度和真实数据训练不在本次范围'],
['测试策略','参数有效域采用等价类和边界值\n采样隔离、错误恢复与空间对齐采用场景法\n损失与指标使用白盒分析及确定数值断言\n每个执行节点关联输入、预期和实际结果'],
['自动化工程','根目录pytest入口限定模块一收集范围\n临时双模态数据和固定种子保证可复现\n用例强制验证非空数据，避免空执行通过\nDataLoader与小型模型验证最小集成流程'],
['模块一独立执行结果','199项收集\n188项通过  0项失败  11项CUDA跳过\nCPU执行项通过率100%\n历史CUDA记录同样为跳过，后续补测'],
['指定模块覆盖率','语句与分支综合统计\n数据集75.09%  工程指标89.66%  损失函数96.15%\n三个指定模块合计79.43%\n覆盖率不等同于完整系统质量证明'],
['典型缺陷  短时延计时','触发条件：小模型与很短的测量循环\n原实现使用墙钟，可能返回0毫秒\n改为高分辨率单调时钟\n可控时钟前进2毫秒，4次运行预期0.5毫秒'],
['缺陷闭环与测试改进','深度文件读取失败时，先检查None再转换类型\n测量次数0、负数、浮点与布尔值统一拒绝\n7个回归执行项修复前失败，修复后通过\n历史仅增加注释的记录不计入有效缺陷'],
['Git记录与成员代号','A对应EdwinNull，B对应sundowner，C暂无提交\n现存已提交Python行归属：A 100%  B 0%  C 0%\nB有清理及合并工作，行归属不代表总工作量\n未提交的AI辅助改动不纳入个人贡献'],
['交付与后续安排','交付Excel用例、Word缺陷清单及测试报告\n保留模块一日志、覆盖率和源码指纹\nCUDA由小组后续补充\n录制视频与现场演练形式待讨论']
];
for(let i=0;i<content.length;i++){
 const s=p.slides.add(); s.background.fill='#FFFFFF';
 const add=(text,left,top,width,height,size,bold=false,color='#172B4D')=>{const t=s.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});t.text=text;t.text.style={typeface:'Microsoft YaHei',fontSize:size,bold,color,autoFit:'none'};return t;};
 add(content[i][0],72,64,1120,92,i===0?52:42,true);
 add(content[i][1],76,208,1100,352,i===0?32:30,false,'#303943');
 add('SteelRailWay  模块一',76,651,700,30,16,false,'#687787');
 add(String(i+1).padStart(2,'0'),1150,650,60,35,18);
 s.speakerNotes.textFrame.setText('依据：deliverables/module1/evidence/module1.log、case-index.json、Git记录。建议讲解约40秒。'+(i===8?'代码比例仅为HEAD现存Python行归属，不代表总工作量。':''));
 const img=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await img.arrayBuffer()));
}
await(await PresentationFile.exportPptx(p)).save(build+'/candidate.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:root+'/deliverables/module1/成果汇报.pptx',pythonExecutable:'D:/Anaconda/envs/AI_common/python.exe',integrityValidatorPath:'C:/Users/sundowner/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:'C:/Users/sundowner/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],verifyArtifactToolImport:true,receiptPath:build+'/ppt-validation.json'});

