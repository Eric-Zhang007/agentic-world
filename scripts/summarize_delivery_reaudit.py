"""Join actual audit evidence into one per-record ledger and local review page."""
import csv
import html
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit'


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def rows(path):
    if not path.exists():
        return []
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def main():
    metadata=rows(ROOT/'metadata/records.jsonl')
    media={r['sample_id']:r for r in rows(ROOT/'media/samples.jsonl')}
    images={r['sample_id']:r for r in rows(ROOT/'images/samples.jsonl')}
    visual={}
    reference_review={}
    reference_index={(r['page'],r['row']):r for r in read(ROOT/'reference_contacts/index.json',[])}
    for r in read(ROOT/'reference_contacts/review.json',{}).get('records',[]):
        item=reference_index[(r['page'],r['row'])]
        reference_review[item['sample_id']]={**r,'reference_index':item['reference_index']}
    for path in (ROOT/'direct_visual_review.json',ROOT/'priority_visual/review.json'):
        for record in read(path,{}).get('records',[]):
            visual[record['sample_id']]=record
    affected={}
    for r in rows(ROOT/'metadata/source_clock_rates.jsonl'):
        if r.get('flagged'):
            for sid in r['sample_ids']:
                affected[sid]=r
    ledger=[]
    for r in metadata:
        sid=r['sample_id']
        m=media.get(sid,{})
        im=images.get(sid,{})
        v=visual.get(sid,{})
        reference=reference_review.get(sid)
        reasons=[]
        if sid in affected:
            reasons.append('source_clock_rate_flagged')
        reasons.extend(sorted({i['code'] for i in r['issues']}))
        if m.get('errors'):
            reasons.append('encoded_media_timing_or_decode_issue')
        if im.get('errors'):
            reasons.append('image_mask_or_embedding_issue')
        if v.get('status','').startswith('CONFIRMED_'):
            reasons.append(v['status'])
        if reference and reference['status'] in ('REFERENCE_SPATIAL_MISMATCH','REFERENCE_CAMERA_MISMATCH'):
            reasons.append(reference['status'])
        media_details={name:{k:value.get(k) for k in ('decoded_frames','errors','nonmonotonic_timestamps',
            'duplicate_timestamps','backwards_timestamps','corrupt_frames')} for name,value in m.get('media',{}).items()}
        ledger.append({'sample_id':sid,'split':r['split'],'task':r['task'],'source':r.get('source'),
            'review_status':'ISSUE_OR_REVIEW_FLAG' if reasons else 'NO_MECHANICAL_FLAG_NOT_VISUALLY_ACCEPTED',
            'reasons':reasons,'metadata_checked':r['audit_completed'],
            'media_checked':bool(m),'media_status':m.get('status','PENDING'),
            'media_details':media_details,'images_complete':bool(im.get('complete')),
            'visual_review':v or {'status':'NOT_DIRECTLY_REVIEWED'},
            'reference_visual_review':reference or {'status':'NOT_DIRECTLY_REVIEWED'},
            'training_acceptance':'NOT_GRANTED_BY_THIS_REAUDIT'})
    with (ROOT/'review_ledger.jsonl').open('w',encoding='utf-8') as f:
        for row in ledger:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (ROOT/'review_ledger.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['sample_id','split','task','review_status','reasons','media_status','images_complete','visual_status'])
        for r in ledger:
            writer.writerow([r['sample_id'],r['split'],r['task'],r['review_status'],';'.join(r['reasons']),
                r['media_status'],r['images_complete'],r['visual_review']['status']])
    summary={'records':len(ledger),'metadata_completed':sum(r['metadata_checked'] for r in ledger),
        'media_completed':sum(r['media_checked'] for r in ledger),
        'images_completed':sum(r['images_complete'] for r in ledger),
        'directly_reviewed_samples':len(visual),'direct_review_scope':'Only documented sample frames; not full videos or every reference',
        'directly_reviewed_reference_pairs':len(reference_review),
        'issue_or_review_flag_records':sum(bool(r['reasons']) for r in ledger),
        'reason_counts':dict(Counter(reason for r in ledger for reason in r['reasons'])),
        'full_visual_acceptance':False,'archive_modified':False,'paid_api_calls':0}
    motion=read(ROOT/'flagged_motion/summary.json',{}).get('results',[])
    strong=[]
    for r in motion:
        fit=r.get('affine_timing',{})
        train=fit.get('train_correlation')
        holdout=fit.get('holdout_correlation')
        zero=fit.get('holdout_zero_lag_correlation')
        if all(isinstance(v,(int,float)) for v in (train,holdout,zero)) and train>.8 and holdout>.6 and holdout-zero>.2:
            strong.append(r)
    summary['motion_diagnostics']={'source_segments_measured':len(motion),'strong_temporal_evidence_segments':len(strong),
        'records_in_those_segments':sum(len(r['sample_ids']) for r in strong),
        'reporting_criteria':'train correlation > .8; held-out correlation > .6; held-out improvement > .2. Diagnostic grouping, not acceptance.',
        'controls_measured':len(read(ROOT/'control_motion/summary.json',{}).get('results',[]))}
    summary['media_findings']={'samples_with_timing_or_decode_errors':sum(bool(r.get('errors')) for r in media.values()),
        'samples_with_rgb_proxy_timestamp_differences':sum(r.get('rgb_proxy_timestamps_match') is False for r in media.values()),
        'fully_decoded_streams':sum(bool(v.get('complete')) for r in media.values() for v in r.get('media',{}).values()),
        'stream_error_counts':dict(Counter(e for r in media.values() for v in r.get('media',{}).values() for e in v.get('errors',[])))}
    (ROOT/'overall_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    cards=[]
    for sid,v in visual.items():
        evidence=v.get('evidence',f'priority_visual/{sid}.jpg')
        if not (ROOT/evidence).exists():
            continue
        findings=v.get('findings',[v.get('finding','')])
        cards.append(f'<article><h2>{html.escape(sid)}</h2><p>{html.escape(v["status"])}</p>'
                     f'<a href="{html.escape(evidence)}"><img loading="lazy" src="{html.escape(evidence)}" alt="RGB and proxy comparison"></a>'
                     +''.join(f'<p>{html.escape(s)}</p>' for s in findings)+'</article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GTA 交付包复核</title><style>body{margin:24px auto;padding:0 20px;max-width:1400px;font:16px/1.6 system-ui;background:#f5f6f7;color:#20252a}h1{font-size:28px}h2{font-size:15px;overflow-wrap:anywhere}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:24px}article{border-top:2px solid #496d69;padding-top:12px}img{width:100%;height:auto}a{color:#166880}p{overflow-wrap:anywhere}pre{white-space:pre-wrap;background:white;padding:16px}</style>
<h1>GTA 交付包复核</h1><p>原始文件未改动。左 RGB，右白模；连续时间修正候选图有第三列。下方仅展示已经直接查看并记录结论的样本。</p>
<p><a href="README.md">调查说明</a> · <a href="review_ledger.csv">全部样本清单 CSV</a> · <a href="metadata/REPORT.md">元数据报告</a> · <a href="media/progress.json">视频解码进度</a></p>'''
    page+='<pre>'+html.escape(json.dumps(summary,indent=2,ensure_ascii=False))+'</pre><section>'+''.join(cards)+'</section></html>'
    refs=''.join(f'<article><a href="reference_contacts/{p.name}"><img loading="lazy" src="reference_contacts/{p.name}" alt="Historical reference RGB and whitebox pairs"></a></article>'
                 for p in sorted((ROOT/'reference_contacts').glob('references_*.jpg')))
    page=page.replace('</html>','<h1>历史参考图对</h1><p>这 18 对参考来自 18 段异常源轨迹，全部已直接查看。逐项结论见 reference_contacts/review.json。</p><section>'+refs+'</section></html>')
    (ROOT/'review.html').write_text(page,encoding='utf-8')
    temporal_lines=[]
    for r in motion:
        fit=r.get('affine_timing',{})
        if fit.get('scale') is None:
            continue
        temporal_lines.append('| '+' / '.join(r['source'])+f" | {fit['scale']:.4f} | {fit['offset']} | {fit['holdout_zero_lag_correlation']:.3f} | {fit['holdout_correlation']:.3f} |")
    report=f'''# GTA 交付包复核结果

结论：交付内容确实存在 RGB 与白模错位，历史参考图对也受影响。不能继续把当前整包称为已经完成对齐验收的正式训练集。

## 这次查了什么

- 元数据：{summary['metadata_completed']} / {len(ledger)} 条。
- 实际视频完整解码：{summary['media_completed']} / {len(ledger)} 条，每条 RGB 与白模两段。
- 图片、mask、embedding 的读取与尺寸连接：{summary['images_completed']} / {len(ledger)} 条。
- 我直接看了 {len(visual)} 条样本的标记帧，以及 18 对历史参考图片；这不等于逐帧目视验收了全部视频。
- 对 18 段时间异常源轨迹检查了连续运动，另用 3 段非异常轨迹做对照。

全部检查使用本地程序与直接查看图片，没有付费视觉 API 调用。原 ZIP、原世界数据和 HF 文件没有改动。

## 为什么会错

最初指出的街道样本已定位到原始时钟拟合：同步记录发生约 4.39 秒偏移跳变后，导出程序将前后记录放入同一个线性拟合，得到了 0.9058927 的时间比例。视频过去约 20.094 秒，映射状态只推进约 18.164 秒。近处主角一直居中，掩盖了远处人物和运镜的错位。这个错误早于我们的清洗，但此前的审核漏掉了它。

只用前半段估计时间对应，在未参与拟合的后半段仍能显著改善相机运动对应。18 段中有 {len(strong)} 段符合本次强证据统计条件，涉及 {summary['motion_diagnostics']['records_in_those_segments']} 条交付记录。这些是需要修复的源轨迹范围，不是已经修好的样本。其余拟合证据弱的轨迹不能自动挪帧。

另有封装时间问题：有些视频虽然 120 帧、平均 24 fps，内部出现重复 PTS，个别 RGB 与白模的时间序列不同。它与上述源时钟错误独立，不能混作一种原因。

本次完成的媒体检查中，{summary['media_findings']['samples_with_timing_or_decode_errors']} 条有封装时间告警，{summary['media_findings']['samples_with_rgb_proxy_timestamp_differences']} 条 RGB/白模时间戳序列不同；共 {summary['media_findings']['fully_decoded_streams']} 个视频完整解码。具体逐帧异常与独立 ffprobe 复现见 media/。

## 已做的修复验证

对 w003 和 w006 两个样本进行了隔离的无损 CFR 修复，四个视频均保留 120 帧，逐帧像素与原视频完全一致，时间戳恢复为 n/24。它只修封装时间，不修原始内容同步。

原打包检查现已加入逐帧 CFR 验证，旧的通过缓存会失效。真实错误视频被拒绝，修复候选通过；14 项相关测试通过。没有批量覆盖训练数据。

## 怎么处理现有数据

1. 封装 PTS 异常：按实际解码顺序修复，要求修复前后逐帧内容一致，再打包。
2. 源时钟异常：先重建帧到状态的时间对应，再重渲染白模；目标视频、参考白模图、状态索引必须一起更新。不能只挪目标视频。
3. 背景几何与交互差异：从时间错误中分离出来验收。固定粗人体可以没有精细动作，但人物根位置、相机、持有/乘坐状态不能因此被默认放过。
4. 清单中未出现程序告警的样本，并不自动获得视觉通过。当前直接复核覆盖范围在 JSON 中逐项记录。

## 连续运动证据

下表的候选对应为 proxy_index = scale × rgb_index + offset。它仅用于诊断，不直接写回数据。

| 源轨迹 | scale | offset | 原始后半段相关性 | 候选后半段相关性 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(temporal_lines)+'''

## 文件

- `review.html`：可直接打开的图文复核页。
- `review_ledger.csv` / `review_ledger.jsonl`：所有样本逐项检查结果。
- `metadata/REPORT.md`：全包元数据统计与时间告警。
- `media/`：实际视频解码结果及每条样本的对照图。
- `images/REPORT.md`：所有图片、mask 和 embedding 检查结果。
- `reference_contacts/`：18 对历史参考的对照与人工结论。
- `flagged_motion/`、`control_motion/`：异常轨迹与对照轨迹的连续运动测量。
- `pts_candidate/`：两条样本的隔离修复实验与逐像素验证记录。

最终覆盖数字以 overall_summary.json 以及各子任务 summary.json 为准。
'''
    (ROOT/'完整复核报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
