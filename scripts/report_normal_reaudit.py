"""Publish manually inspected evidence; no automated visual verdicts."""
import html
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit/normal_random_review'
NOTES=[
('REVIEW','两个人的远近变化大体对应；第119帧第二人相对主角的位置、遮挡有差异。参考图第二人位置也需放大核对。'),
('COARSE','人物跑动、第二人离开画面和机库方向大致对应。参考图人物和地面方向对应，地表纹理未建模。'),
('REVIEW','飞机行进视角大致对应，但白模只有车辆包围框，飞机主体和机场地面设施缺失。参考图同样如此；这不是已证实的时间错配。'),
('COARSE','跑道两人前后关系、跑动转向和机库轮廓大致对应；参考图两人相对位置大致对应。'),
('COARSE','远处第二人接近主角的过程可对应。参考图车内人物透出属于线框车辆/遮挡表达局限。'),
('COARSE','车辆驶入、驶离与人物转身顺序大致对应；港口结构和参考图方向可对应。'),
('COARSE','第二人进入画面并停在右前方可对应；参考图倒地人物与车辆位置可对应，植被表示很粗。'),
('REVIEW','机场人物移动大致对应；飞机、登机梯等重要背景缺失，第119帧附近的白模大面遮挡了RGB中可见区域。参考图也缺少飞机主体。'),
('COARSE','山坡两人相对移动和坡面方向大致对应；参考图也对应，装饰性大型雕塑未表示。'),
('REVIEW','车旁/车顶两人的移动顺序大体相似，但角色高度、车体接触和相对位置需核对；参考图仅能确认停车场布局。'),
('COARSE','机场草地转身、第二人进出画面大致对应；参考图两人和机库方向可对应。'),
('COARSE','车辆沿道路前进和转向大致对应；参考图道路方向可对应。车内人形站立透出是粗代理表示，不能当作姿态正确。'),
('COARSE','人物穿过警车群的路径与运镜大致对应；倒地NPC有粗表示，参考图道路方向对应。'),
('COARSE','两人近距离交互后分开，位置与镜头大致对应；攻击姿态在代理中缺失，参考图两人布局对应。'),
('COARSE','树旁两人相对移动大致对应；参考图可找到车辆与倒地角色，肢体动作不具备对应精度。'),
('REVIEW','第60/90/119帧第二人在RGB中清晰可见，在白模中只露出窄条，近处背景几何遮挡不同；参考图也存在此现象。'),
('COARSE','路边转向、走向垃圾桶及跳起的大致顺序可对应；参考图人物与垃圾桶布局对应。'),
('REVIEW','游泳段码头支撑结构大幅简化；第0/30帧白模第二角色在高处，RGB水面附近人物与其对应关系可疑。参考图爬梯姿态无法由代理确认。需连续帧核对。'),
('REVIEW','墙间攀爬的背景通道和角色高度/接触关系有明显表达差异，白模直立角色无法验证攀爬阶段；参考图同样不能证明接触对应。'),
('COARSE','第二人从远处跑近并到主角身边，抽样位置大致对应；参考图道路及两人布局对应，植被轮廓粗糙。'),
('COARSE','静止俯视画面的两人位置、墙边界大致对应；参考图相似。静止片段不能检验运动时间对齐。'),
('COARSE','夜间路上转身往返和第二人远近关系大致对应；参考图相机方向对应。'),
('REVIEW','楼梯处上下车，第60帧RGB主角在车侧，白模角色仍在车框内，另一角色高度也需核对；参考图车辆区域有对应。疑似交互时序/姿态问题，尚未定因。'),
('REVIEW','两人围绕倒地NPC移动的整体场景对应，但第60/90/119帧倒地者相对主角的位置和姿态偏差明显；参考图车辆/道路对应。需区分布娃娃简化与时间错配。'),
]

def main():
    selection=json.loads((ROOT/'selection.json').read_text(encoding='utf-8'))
    records=[]
    for i,(sample,(status,note)) in enumerate(zip(selection['records'],NOTES,strict=True)):
        records.append({'number':i+1,'sample_id':sample['sample_id'],'task':sample['task'],'status':status,'finding':note,'target_frames':[0,30,60,90,119],'target_evidence':f'targets_{i//4:02d}.jpg','reference_evidence':f'references_{i//4:02d}.jpg'})
    result={'selection_method':selection['method'],'eligible':selection['eligible'],'seed':selection['seed'],'counts':dict(Counter(r['status'] for r in records)),'scope':'Direct visual inspection of five target frame pairs and one historical pair per record; not continuous video or frame-accurate validation','paid_api_calls':0,'records':records}
    (ROOT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 程序正常样本的随机抽检','',f"从 {selection['eligible']} 条无告警且未人工检查的记录中，固定种子随机抽取24条。不是按好坏挑选。覆盖7个episode、4类任务，23条train、1条test；未覆盖validation。",'',
          '每条直接查看目标第0、30、60、90、119帧的RGB/白模，以及1对历史参考帧，共120对目标画面、24对历史参考。未调用付费API。','',
          '结果：15条在这些画面中粗位置/运镜基本对应；9条存在需深入检查的问题。后者包括背景资产缺失、遮挡差异、特殊动作表达不足和疑似时空错位，不能全部算作帧匹配错误。前者也不等于正式验收通过。','',
          '这次没有证明“大多数帧配错了”，也不支持“程序正常就能直接训练”。人物/相机的粗对应比背景与交互状态更可靠。对于背景记忆训练，飞机、码头、近处结构的缺失和错误遮挡值得优先处理。','',
          '抽样单位为训练记录，不是独立场景；存在同源片段相关性。本次不推算全包合格率。','',
          '下表COARSE表示抽样画面粗对应，REVIEW表示需要进一步核验，不是最终合格/不合格标签。','',
          '|编号|结论|样本|观察|','|---|---|---|---|']
    for r in records:
        text.append(f"|{r['number']}|{r['status']}|{r['sample_id']}|{r['finding']}|")
    (ROOT/'抽检报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    body=['<!doctype html><meta charset="utf-8"><title>程序正常样本抽检</title><style>body{max-width:1320px;margin:24px auto;font:17px sans-serif}img{max-width:100%}p{line-height:1.6}</style><h1>程序正常样本随机抽检</h1>',
          '<p>24条：15条抽样画面粗对应，9条需深入检查。查看了每条5对目标帧和1对历史帧；不等于逐帧验收。没有调用付费API。</p>']
    for page in range(6):
        body.append(f'<h2>样本 {page*4+1}–{page*4+4}</h2>')
        for r in records[page*4:page*4+4]:
            body.append(f'<p><b>{r["number"]} {r["status"]}</b> {html.escape(r["sample_id"])}<br>{html.escape(r["finding"])}</p>')
        body.append(f'<img src="targets_{page:02d}.jpg"><h3>对应历史参考帧</h3><img src="references_{page:02d}.jpg">')
    (ROOT/'review.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(result['counts']))

if __name__=='__main__':
    main()
