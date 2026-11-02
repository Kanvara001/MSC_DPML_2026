import base64, io, os
from PIL import Image

D = os.path.dirname(os.path.abspath(__file__))
PG = os.path.join(D, 'pg')
OUT = os.environ.get('DECK_OUT', os.path.join(os.path.dirname(D), 'ffdbackbone-crispdm-deck.html'))

def img(name, w, crop=None):
    im = Image.open(os.path.join(PG, name)).convert('RGB')
    if crop: im = im.crop(crop)
    if im.width > w: im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

FIG3 = img('fig3.png', 1800)
FIG3TOP = img('fig3.png', 1800, (0, 40, 1616, 228))
FIG4 = img('fig4.png', 1200)
FIG5 = img('fig5.png', 1000)
AUGS = [img(f'aug_{n}.png', 224) for n in ['0orig','1rgb','2hsv','3bc','4jpg','5flip','6gray','7cut']]

# ---------- chart helpers (all marks placed with one linear scale) ----------
def dotplot(rows, lo, hi, ticks, W=620, label_w=210, row_h=44, top=14, fmt='{:.2f}', axis_title=''):
    """rows: (label, value, kind) kind in {'base','mid','hi'}"""
    x0, x1 = label_w, W - 60
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    H = top + row_h * len(rows) + 46
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    gb = top + row_h * len(rows)
    for t in ticks:
        x = sx(t)
        o.append(f'<line class="g" x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{gb}"/>')
        o.append(f'<text class="tk" x="{x:.1f}" y="{gb+20}" text-anchor="middle">{t:g}</text>')
    if axis_title:
        o.append(f'<text class="tk" x="{(x0+x1)/2:.1f}" y="{gb+40}" text-anchor="middle">{axis_title}</text>')
    for k, (lab, v, kind) in enumerate(rows):
        y = top + row_h * k + row_h / 2
        x = sx(v)
        cls = {'base': 'm0', 'mid': 'm1', 'hi': 'm2'}[kind]
        wt = ' font-weight="600"' if kind == 'hi' else ''
        o.append(f'<text class="lb" x="0" y="{y+5:.1f}"{wt}>{lab}</text>')
        o.append(f'<line class="{cls}" x1="{x0}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke-width="{4 if kind=="hi" else 2.5}"/>')
        o.append(f'<circle class="{cls}f" cx="{x:.1f}" cy="{y:.1f}" r="{8 if kind=="hi" else 6.5}"/>')
        o.append(f'<text class="vl" x="{x+14:.1f}" y="{y+5:.1f}"{wt}>{fmt.format(v)}</text>')
    o.append('</svg>')
    return ''.join(o)

def stackbar(segs, W=620, h=44, label=''):
    """segs: (label, count, cls)"""
    tot = sum(c for _, c, _ in segs)
    o = [f'<svg viewBox="0 0 {W} {h+44}" width="100%" role="img" class="chart">']
    x = 0
    for lab, c, cls in segs:
        w = c / tot * W
        o.append(f'<rect class="{cls}" x="{x:.1f}" y="0" width="{w-2:.1f}" height="{h}"/>')
        o.append(f'<text class="sb" x="{x+w/2-1:.1f}" y="{h/2+5:.1f}" text-anchor="middle">{c:,}</text>')
        o.append(f'<text class="tk" x="{x+w/2-1:.1f}" y="{h+22:.1f}" text-anchor="middle">{lab}</text>')
        x += w
    o.append('</svg>')
    return ''.join(o)

# Table I ของบทความ (Avg. AUC ของ CDF/DFDC/FFIW) · ตรวจกับ PDF หน้า 9236 แล้ว
PRE = dotplot([
    ('ViT-B · supervised 1K', 77.11, 'base'),
    ('CLIP (VLM)', 81.92, 'base'),
    ('BEiT v2 · SSL 1K', 82.70, 'mid'),
    ('BEiT · หน้าล้วน', 83.77, 'mid'),
    ('FB · 1K → หน้า', 86.56, 'hi'),
], 75, 90, [75, 80, 85, 90], W=560, label_w=190, row_h=50, axis_title='Avg. cross-set AUC (%) · Table I')

SOTA = dotplot([
    ('CADDM', 79.07, 'base'), ('FTCN', 81.34, 'base'), ('SBI', 84.22, 'base'),
    ('TFCU (temporal)', 90.10, 'base'), ('CFB', 89.94, 'mid'), ('CFB++ w/ SBI', 93.35, 'hi'),
], 75, 95, [75, 80, 85, 90, 95], W=640, label_w=170, row_h=50, axis_title='Average AUC บน 5 cross-dataset (%)')

ABL = dotplot([
    ('Dual-branch (sum)', 87.04, 'base'), ('+ L<tspan baseline-shift="sub" font-size="11">dec</tspan>', 88.14, 'mid'), ('+ UFM + L<tspan baseline-shift="sub" font-size="11">EDU</tspan>', 88.78, 'hi'),
], 86, 89.5, [86, 87, 88, 89], W=420, label_w=150, row_h=74, axis_title='Avg. AUC (%)')

# Table IX ของบทความ: CFB + L_EDU, คอลัมน์ Avg. (เดิมใช้ 79.42 ซึ่งเป็น ACC ของ variant "+L_dec" ใน Table VI ไม่ใช่ τ = 0.5)
UTOM = dotplot([
    ('τ = 0.5', 80.13, 'base'), ('τ<tspan baseline-shift="sub" font-size="11">ot</tspan>', 80.63, 'mid'), ('τ จาก UTOM', 82.29, 'hi'),
], 78, 84, [78, 80, 82, 84], W=420, label_w=120, row_h=64, axis_title='Avg. ACC (%) · Table IX')

FFPP = stackbar([('Real', 1000, 'cr'), ('Deepfakes', 1000, 'cf1'), ('Face2Face', 1000, 'cf2'),
                 ('FaceSwap', 1000, 'cf3'), ('NeuralTextures', 1000, 'cf4')], W=640)
CDF = stackbar([('Real', 178, 'cr'), ('Fake', 340, 'cf1')], W=640)


def splitbars(rows, maxv, W=700, label_w=120, bar_h=40, gap=26, right=150):
    """rows: (name, sub, [(lab,count,cls)]) ; length proportional to count on one shared scale"""
    x0=label_w; span=W-label_w-right
    H=len(rows)*(bar_h+gap)
    o=[f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    for k,(name,sub,segs) in enumerate(rows):
        y=k*(bar_h+gap)+6
        tot=sum(c for _,c,_ in segs)
        o.append(f'<text class="lb" x="0" y="{y+bar_h/2:.1f}" font-weight="600">{name}</text>')
        o.append(f'<text class="tk" x="0" y="{y+bar_h/2+17:.1f}">{sub}</text>')
        x=x0
        for lab,c,cls in segs:
            w=c/maxv*span
            o.append(f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{max(w-1.5,0.5):.1f}" height="{bar_h}"/>')
            if w>46: o.append(f'<text class="sb" x="{x+w/2:.1f}" y="{y+bar_h/2+5:.1f}" text-anchor="middle">{c:,}</text>')
            x+=w
        o.append(f'<text class="vl" x="{x+10:.1f}" y="{y+bar_h/2+5:.1f}" font-weight="600">{tot:,} ({tot/5000*100:.0f}%)</text>')
    o.append('</svg>')
    return ''.join(o)

def segs(r):
    return [('Real',r,'cr'),('DF',r,'cf1'),('F2F',r,'cf2'),('FS',r,'cf3'),('NT',r,'cf4')]
SPLIT_BEFORE = splitbars([('ทั้งหมด','1,000 identity',segs(1000))], 5000, bar_h=44)
SPLIT_AFTER = splitbars([('Train','360 pairs',segs(720)),('Validation','70 pairs',segs(140)),('Test','70 pairs',segs(140))], 5000, bar_h=40)

import json
CROPS = [img(f'crop{k}.png', 260) for k in range(5)]

def linechart(xs, ys, lo, hi, ticks, W=400, H=290, ylab='', fmt='{:.2f}', cls='m2', last_label=True, mark_best=False):
    L, R, T, B = 52, 18, 14, 40
    sx = lambda v: L + (v - xs[0]) / (xs[-1] - xs[0]) * (W - L - R)
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    for t in ticks:
        o.append(f'<line class="g" x1="{L}" y1="{sy(t):.1f}" x2="{W-R}" y2="{sy(t):.1f}"/>')
        o.append(f'<text class="tk" x="{L-8}" y="{sy(t)+4:.1f}" text-anchor="end">{t:g}</text>')
    for x in xs:
        o.append(f'<text class="tk" x="{sx(x):.1f}" y="{H-B+18}" text-anchor="middle">{x}</text>')
    o.append(f'<text class="tk" x="{(L+W-R)/2:.1f}" y="{H-4}" text-anchor="middle">Epoch</text>')
    pts = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(xs, ys))
    o.append(f'<polyline points="{pts}" fill="none" class="{cls}" stroke-width="2.5"/>')
    for x, y in zip(xs, ys):
        o.append(f'<circle class="{cls}f" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5"/>')
    if last_label:
        o.append(f'<text class="vl" x="{sx(xs[-1])-6:.1f}" y="{sy(ys[-1])-10:.1f}" text-anchor="end" font-weight="600">{fmt.format(ys[-1])}</text>')
        o.append(f'<text class="vl" x="{sx(xs[0])+6:.1f}" y="{sy(ys[0])-10:.1f}">{fmt.format(ys[0])}</text>')
    o.append('</svg>')
    return ''.join(o)

EP = list(range(1, 11))
VAL_AUC = [78.48, 86.01, 91.13, 93.19, 94.36, 95.26, 95.75, 95.96, 96.23, 96.45]
TR_LOSS = [2.3923, 1.2096, 0.8682, 0.7849, 0.7266, 0.7213, 0.7345, 0.7169, 0.6852, 0.6093]
PEAR = [0.6794, -0.2832, -0.5202, -0.4956, -0.4543, -0.4173, -0.3915, -0.3695, -0.3572, -0.3430]
CURVE_AUC = linechart(EP, VAL_AUC, 75, 100, [75, 80, 85, 90, 95, 100], ylab='Val AUC (%)')
CURVE_LOSS = linechart(EP, TR_LOSS, 0, 2.5, [0, 0.5, 1, 1.5, 2, 2.5], cls='m1', fmt='{:.2f}')
CURVE_PEAR = linechart(EP, PEAR, -0.75, 0.75, [-0.5, 0, 0.5], cls='m0', fmt='{:+.2f}')

def rocchart(pts, W=340, H=300, auc=''):
    L, R, T, B = 48, 14, 12, 44
    s = W - L - R; h = H - T - B
    sx = lambda v: L + v * s; sy = lambda v: T + (1 - v) * h
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    for t in [0, .25, .5, .75, 1]:
        o.append(f'<line class="g" x1="{sx(t):.1f}" y1="{T}" x2="{sx(t):.1f}" y2="{T+h}"/><line class="g" x1="{L}" y1="{sy(t):.1f}" x2="{L+s}" y2="{sy(t):.1f}"/>')
        o.append(f'<text class="tk" x="{sx(t):.1f}" y="{T+h+17}" text-anchor="middle">{t:g}</text><text class="tk" x="{L-7}" y="{sy(t)+4:.1f}" text-anchor="end">{t:g}</text>')
    o.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" class="m0" stroke-dasharray="4 4"/>')
    poly = ' '.join(f'{sx(a):.1f},{sy(b):.1f}' for a, b in pts)
    o.append(f'<polygon points="{sx(0)},{sy(0)} {poly} {sx(1)},{sy(0)}" style="fill:var(--accent-soft)"/>')
    o.append(f'<polyline points="{poly}" fill="none" class="m2" stroke-width="2.5"/>')
    o.append(f'<text class="vl" x="{sx(.55):.1f}" y="{sy(.25):.1f}" font-weight="600">AUC = {auc}</text>')
    o.append(f'<text class="tk" x="{L+s/2:.1f}" y="{H-6}" text-anchor="middle">False positive rate</text>')
    o.append(f'<text class="tk" transform="translate(12,{T+h/2:.1f}) rotate(-90)" text-anchor="middle">True positive rate</text>')
    o.append('</svg>')
    return ''.join(o)
ROC_FF = rocchart(json.load(open(os.path.join(D, 'roc.json'))), auc='96.72')

PERMETHOD = dotplot([
    ('Deepfakes', 99.66, 'mid'), ('FaceSwap', 99.19, 'mid'), ('Face2Face', 98.03, 'mid'),
    ('NeuralTextures', 89.99, 'hi'), ('รวมทุก method', 96.72, 'base'),
], 85, 100, [85, 90, 95, 100], W=440, label_w=150, row_h=44, axis_title='AUC real vs method (%)')

CDF_AUC = dotplot([
    ('ฝึกเอง · main', 84.97, 'hi'), ('ฝึกเอง · auxiliary', 85.65, 'mid'), ('ฝึกเอง · fused', 86.10, 'mid'),
    ('FB ไม่มี CFB (บทความ)', 89.18, 'base'), ('CFB (บทความ)', 90.46, 'base'),
], 80, 92, [80, 84, 88, 92], W=440, label_w=200, row_h=54, axis_title='Celeb-DF video-AUC (%)')
CDF_CLS = dotplot([
    ('Real (TNR)', 87.64, 'mid'), ('Fake (TPR)', 62.35, 'hi'),
], 50, 100, [50, 60, 70, 80, 90, 100], W=380, label_w=110, row_h=80, axis_title='Accuracy ที่ τ = 0.5 (%)')
CDF_THR = dotplot([
    ('τ = 0.5', 71.04, 'base'), ('τ = 0.293', 78.96, 'hi'),
], 65, 85, [65, 70, 75, 80, 85], W=380, label_w=110, row_h=80, axis_title='Celeb-DF ACC (%)')

def pairplot(rows, lo, hi, ticks, W=440, label_w=150, row_h=50, top=14, axis_title='', fmt='{:.2f}'):
    """rows: (label, a, b) a = reference (grey), b = ours (accent)"""
    x0, x1 = label_w, W - 56
    sx = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (x1 - x0)
    gb = top + row_h * len(rows); H = gb + 46
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    for t in ticks:
        x = sx(t)
        o.append(f'<line class="g" x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{gb}"/><text class="tk" x="{x:.1f}" y="{gb+20}" text-anchor="middle">{t:g}</text>')
    if axis_title:
        o.append(f'<text class="tk" x="{(x0+x1)/2:.1f}" y="{gb+40}" text-anchor="middle">{axis_title}</text>')
    for k, (lab, a, b) in enumerate(rows):
        y = top + row_h * k + row_h / 2
        xa, xb = sx(a), sx(b)
        o.append(f'<text class="lb" x="0" y="{y+5:.1f}">{lab}</text>')
        o.append(f'<line class="m0" x1="{min(xa,xb):.1f}" y1="{y:.1f}" x2="{max(xa,xb):.1f}" y2="{y:.1f}" stroke-width="2" stroke-dasharray="3 3"/>')
        o.append(f'<circle class="m0f" cx="{xa:.1f}" cy="{y:.1f}" r="6.5"/><circle class="m2f" cx="{xb:.1f}" cy="{y:.1f}" r="7.5"/>')
        hiv, lov = (a, b) if a >= b else (b, a)
        xr, xl = max(xa, xb), min(xa, xb)
        o.append(f'<text class="vl" x="{xr+12:.1f}" y="{y+5:.1f}">{fmt.format(hiv)}</text>')
        o.append(f'<text class="vl" x="{xl-10:.1f}" y="{y+5:.1f}" text-anchor="end" font-weight="600">{fmt.format(lov)}</text>' if abs(xa-xb) > 26 else '')
    o.append('</svg>')
    return ''.join(o)

XDS = pairplot([('FF++ c23 (in-domain)', 99.24, 96.57), ('Celeb-DF v2', 90.05, 84.97), ('WildDeepfake *', 89.78, 84.44)],
    80, 100, [80, 85, 90, 95, 100], W=440, label_w=165, row_h=58, axis_title='Video-AUC (%)')
PERM2 = pairplot([('Deepfakes', 99.99, 99.68), ('FaceSwap', 99.91, 99.19), ('Face2Face', 98.97, 98.04), ('NeuralTextures', 98.10, 89.37)],
    85, 100, [85, 90, 95, 100], W=440, label_w=130, row_h=50, axis_title='AUC real vs method (%)')
ABL2 = pairplot([('FB · CE', 89.18, 87.92), ('FB · EDU', 89.34, 87.24), ('CFB base', 89.77, 87.04), ('+ L<tspan baseline-shift="sub" font-size="11">dec</tspan>', 90.31, 85.53),
    ('+ L<tspan baseline-shift="sub" font-size="11">dec</tspan> + UFM', 91.16, 86.39), ('CFB full', 90.46, 84.97)],
    82, 92, [82, 84, 86, 88, 90, 92], W=470, label_w=130, row_h=46, axis_title='Celeb-DF video-AUC (%)')
UTOM2 = pairplot([('τ = 0.5', 84.10, 77.25), ('τ<tspan baseline-shift="sub" font-size="11">ot</tspan> (source)', 74.84, 68.81), ('τ<tspan baseline-shift="sub" font-size="11">UTOM</tspan> (source)', 83.12, 74.15)],
    65, 90, [65, 70, 75, 80, 85, 90], W=420, label_w=130, row_h=62, axis_title='ACC เฉลี่ยบน target ที่ไม่เคยเห็น (%)')


def seedstrip(rows, lo, hi, ticks, W=520, label_w=150, row_h=54, top=14, axis_title=''):
    """rows: (label, [seed values], paper or None) · จุด = แต่ละ seed, ขีดตั้ง = ค่าเฉลี่ย, สี่เหลี่ยมเทา = บทความ"""
    x0, x1 = label_w, W - 24
    sx = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (x1 - x0)
    gb = top + row_h * len(rows); H = gb + 46
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    for t in ticks:
        x = sx(t)
        o.append(f'<line class="g" x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{gb}"/><text class="tk" x="{x:.1f}" y="{gb+20}" text-anchor="middle">{t:g}</text>')
    if axis_title:
        o.append(f'<text class="tk" x="{(x0+x1)/2:.1f}" y="{gb+40}" text-anchor="middle">{axis_title}</text>')
    for k, (lab, vals, paper) in enumerate(rows):
        y = top + row_h * k + row_h / 2
        o.append(f'<text class="lb" x="0" y="{y+5:.1f}">{lab}</text>')
        o.append(f'<line class="m0" x1="{sx(min(vals)):.1f}" y1="{y:.1f}" x2="{sx(max(vals)):.1f}" y2="{y:.1f}" stroke-width="2"/>')
        if paper is not None:
            xp = sx(paper)
            o.append(f'<rect class="m0f" x="{xp-6:.1f}" y="{y-6:.1f}" width="12" height="12"/>')
        for v in vals:
            o.append(f'<circle class="m2f" cx="{sx(v):.1f}" cy="{y:.1f}" r="6" fill-opacity="0.75"/>')
        m = sum(vals) / len(vals); xm = sx(m)
        o.append(f'<line class="m2" x1="{xm:.1f}" y1="{y-13:.1f}" x2="{xm:.1f}" y2="{y+13:.1f}" stroke-width="3"/>')
        o.append(f'<text class="vl" x="{xm:.1f}" y="{y-17:.1f}" text-anchor="middle" font-weight="600">{m:.2f}</text>')
    o.append('</svg>')
    return ''.join(o)

SEEDS = seedstrip([('FB · CE · CDF', [87.92, 86.23, 86.84], 89.18), ('CFB full · CDF', [84.97, 87.35, 87.69], 90.46)],
    84, 91, [84, 85, 86, 87, 88, 89, 90, 91], W=520, label_w=150, row_h=90, axis_title='Celeb-DF video-AUC main branch (%) · seed 1234, 2025, 3407')

# ---- cross-forgery (DeepFakeFace, checkpoint ผู้เขียน) และ few-shot (3g) ----
DFF = dotplot([
    ('Celeb-DF (อ้างอิง · วิดีโอ)', 90.05, 'base'),
    ('InsightFace · face swap', 89.75, 'mid'),
    ('SD Inpainting', 80.71, 'mid'),
    ('SD 1.5 · text-to-image', 73.62, 'hi'),
], 70, 95, [70, 75, 80, 85, 90, 95], W=560, label_w=230, row_h=62, axis_title='AUC (%) · checkpoint ผู้เขียน')

DFF2 = dotplot([
    ('Celeb-DF (วิดีโอ)', 90.05, 'base'),
    ('InsightFace swap', 89.75, 'mid'),
    ('SD Inpainting', 80.71, 'mid'),
    ('SD 1.5 text2img', 73.62, 'hi'),
], 70, 95, [70, 80, 90], W=360, label_w=140, row_h=52, axis_title='AUC (%)')

def deltaplot(rows, lo, hi, ticks, W=440, label_w=170, row_h=44, top=14, axis_title=''):
    """rows: (label, delta, kind) · lollipop เริ่มจาก 0"""
    x0, x1 = label_w, W - 50
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    gb = top + row_h * len(rows); H = gb + 46
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart">']
    for t in ticks:
        x = sx(t)
        o.append(f'<line class="g" x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{gb}"/><text class="tk" x="{x:.1f}" y="{gb+20}" text-anchor="middle">{t:+g}</text>')
    o.append(f'<line x1="{sx(0):.1f}" y1="{top-4}" x2="{sx(0):.1f}" y2="{gb}" stroke="var(--mdark)" stroke-width="1.5"/>')
    if axis_title:
        o.append(f'<text class="tk" x="{(x0+x1)/2:.1f}" y="{gb+40}" text-anchor="middle">{axis_title}</text>')
    for k, (lab, v, kind) in enumerate(rows):
        y = top + row_h * k + row_h / 2
        cls = {'base': 'm0', 'hi': 'm2'}[kind]
        o.append(f'<text class="lb" x="0" y="{y+5:.1f}" style="font-size:14.5px">{lab}</text>')
        o.append(f'<line class="{cls}" x1="{sx(0):.1f}" y1="{y:.1f}" x2="{sx(v):.1f}" y2="{y:.1f}" stroke-width="3"/>')
        o.append(f'<circle class="{cls}f" cx="{sx(v):.1f}" cy="{y:.1f}" r="6.5"/>')
        o.append(f'<text class="vl" x="{sx(v)+12:.1f}" y="{y+5:.1f}">{v:+.2f}</text>' if v >= 0 else
                 f'<text class="vl" x="{sx(v)-12:.1f}" y="{y+5:.1f}" text-anchor="end">{v:+.2f}</text>')
    o.append('</svg>')
    return ''.join(o)

# Table V (CFB): CDF 90.46→91.33/91.31, DFDC 84.90→87.72/88.07, FFIW 90.97→96.65 · เรา: fewshot_summary.csv (รอบ fp32, 20 ต.ค.)
FEWSHOT = deltaplot([
    ('บทความ · CDF 10-shot', 0.87, 'base'), ('บทความ · CDF 20-shot', 0.85, 'base'),
    ('บทความ · DFDC 10-shot', 2.82, 'base'), ('บทความ · DFDC 20-shot', 3.17, 'base'), ('บทความ · FFIW 20-shot', 5.68, 'base'),
    ('เรา · CDF 10-shot', -0.65, 'hi'), ('เรา · CDF 20-shot', -0.08, 'hi'),
], -2, 7, [0, 2, 4, 6], W=380, label_w=172, row_h=36, axis_title='ΔAUC (จุด) · CFB main branch')


# ---- Part I ใหม่ (22 ต.ค. 2569): crop Fig. 3 แยก 3 ขั้น, ภาพตัวอย่าง FF++, แผนภาพ patch/mask, แนวคิด UTOM ----
FIG3FB = img('fig3.png', 1000, (55, 273, 550, 437))
FIG3CFB = img('fig3.png', 1300, (553, 273, 1187, 436))
FIG3UTOM = img('fig3.png', 800, (1188, 273, 1558, 437))
FFEX = img('ffpp_examples.jpeg', 900)
import random

def patchvis(src, mask_ratio=0.0, W=300, seed=7):
    n = 14; c = W / n
    idx = set(random.Random(seed).sample(range(n * n), round(n * n * mask_ratio)))
    o = [f'<svg viewBox="0 0 {W} {W}" width="100%" role="img" class="chart" aria-label="ใบหน้าแบ่งเป็น 14 x 14 patch">',
         f'<image href="{src}" x="0" y="0" width="{W}" height="{W}" preserveAspectRatio="xMidYMid slice"/>']
    for k in sorted(idx):
        r, q = divmod(k, n)
        o.append(f'<rect x="{q*c:.2f}" y="{r*c:.2f}" width="{c:.2f}" height="{c:.2f}" fill="#8e9a9c"/>')
    for t in range(n + 1):
        o.append(f'<line x1="{t*c:.2f}" y1="0" x2="{t*c:.2f}" y2="{W}" stroke="#fff" stroke-opacity=".75" stroke-width="1"/>')
        o.append(f'<line x1="0" y1="{t*c:.2f}" x2="{W}" y2="{t*c:.2f}" stroke="#fff" stroke-opacity=".75" stroke-width="1"/>')
    o.append('</svg>')
    return ''.join(o)

GRIDVIS = patchvis(CROPS[0], 0.0)
GRIDVIS0 = GRIDVIS
MASKVIS = patchvis(CROPS[0], 0.40)

def utomvis(W=520, H=300):
    import math
    L, R, T, B = 20, 20, 56, 50
    lo, hi = 0.0, 0.75
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    g = lambda x, m, s: math.exp(-0.5 * ((x - m) / s) ** 2)
    xs = [lo + i * (hi - lo) / 200 for i in range(201)]
    base = H - B; amp = H - T - B - 20
    def path(m, s):
        pts = ' '.join(f'{sx(x):.1f},{base - amp * g(x, m, s):.1f}' for x in xs)
        return f'M{sx(lo):.1f},{base} L{pts} L{sx(hi):.1f},{base} Z'
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" class="chart" aria-label="แนวคิด threshold คงที่ 0.5 เทียบกับ threshold จาก UTOM">']
    o.append(f'<path d="{path(0.13, 0.05)}" style="fill:var(--real)" fill-opacity=".55"/>')
    o.append(f'<path d="{path(0.30, 0.065)}" style="fill:var(--f1)" fill-opacity=".55"/>')
    o.append(f'<line x1="{L}" y1="{base}" x2="{W-R}" y2="{base}" stroke="var(--mdark)" stroke-width="1.5"/>')
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        o.append(f'<text class="tk" x="{sx(t):.1f}" y="{base+18}" text-anchor="middle">{t:g}</text>')
    o.append(f'<text class="tk" x="{(L+W-R)/2:.1f}" y="{H-8}" text-anchor="middle">คะแนนหลังปรับสเกล p′</text>')
    o.append(f'<text class="lb" x="{sx(0.13):.1f}" y="{base-amp-6:.1f}" text-anchor="middle" font-weight="600">Real</text>')
    o.append(f'<text class="lb" x="{sx(0.30):.1f}" y="{base-amp-6:.1f}" text-anchor="middle" font-weight="600">Fake</text>')
    x5 = sx(0.5)
    o.append(f'<line x1="{x5:.1f}" y1="48" x2="{x5:.1f}" y2="{base}" stroke="var(--muted)" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text class="tk" x="{x5+8:.1f}" y="62" font-weight="600">τ = 0.5 (คงที่)</text>')
    o.append(f'<text class="tk" x="{x5+8:.1f}" y="82">→ ตัดสินเกือบทั้งหมด</text>')
    o.append(f'<text class="tk" x="{x5+8:.1f}" y="100">ว่าเป็น real</text>')
    xu = sx(0.215)
    o.append(f'<line x1="{xu:.1f}" y1="8" x2="{xu:.1f}" y2="{base}" stroke="var(--accent)" stroke-width="3"/>')
    o.append(f'<text class="vl" x="{xu-8:.1f}" y="18" text-anchor="end" font-weight="600" style="fill:var(--accent)">τ จาก UTOM</text>')
    o.append('</svg>')
    return ''.join(o)

UTOMVIS = utomvis()


# ---- DF40 (ระดับ 3i · results_level3i) ----
DF40_FF = stackbar([('Real', 140, 'cr'), ('FR', 278, 'cf2'), ('FS', 1396, 'cf4')], W=640)
DF40_CDF = stackbar([('Real', 178, 'cr'), ('FR', 1307, 'cf2'), ('FS', 2976, 'cf4')], W=640)
DF40_CMP = pairplot([('SadTalker', 87.94, 55.33), ('FOMM', 99.15, 87.65), ('FaceDancer', 95.23, 80.25), ('MobileSwap', 99.18, 89.86),
    ('SimSwap', 95.92, 88.85), ('InSwapper', 97.22, 91.82), ('UniFace', 98.93, 96.45)],
    50, 100, [50, 60, 70, 80, 90, 100], W=470, label_w=110, row_h=46, axis_title='video-AUC (%) · DF40 ff + cdf')

html = open(os.path.join(D, 'deck2.tpl.html'), encoding='utf-8').read()
for k, v in dict(FIG3=FIG3, FIG3TOP=FIG3TOP, FIG4=FIG4, FIG5=FIG5, PRE=PRE, SOTA=SOTA,
                 ABL=ABL, UTOM=UTOM, FFPP=FFPP, CDF=CDF, SPLIT_BEFORE=SPLIT_BEFORE, SPLIT_AFTER=SPLIT_AFTER, CURVE_AUC=CURVE_AUC, CURVE_LOSS=CURVE_LOSS, CURVE_PEAR=CURVE_PEAR, ROC_FF=ROC_FF, PERMETHOD=PERMETHOD, CROP0=CROPS[0], CROP1=CROPS[1], CROP2=CROPS[2], CROP3=CROPS[3], CROP4=CROPS[4], CDF_AUC=CDF_AUC, CDF_CLS=CDF_CLS, CDF_THR=CDF_THR, XDS=XDS, PERM2=PERM2, ABL2=ABL2, UTOM2=UTOM2, SEEDS=SEEDS, DFF=DFF, DFF2=DFF2, FEWSHOT=FEWSHOT, FIG3FB=FIG3FB, FIG3CFB=FIG3CFB, FIG3UTOM=FIG3UTOM, FFEX=FFEX, GRIDVIS=GRIDVIS, GRIDVIS0=GRIDVIS0, MASKVIS=MASKVIS, UTOMVIS=UTOMVIS, DF40_FF=DF40_FF, DF40_CDF=DF40_CDF, DF40_CMP=DF40_CMP, **{f'AUG{i}': a for i, a in enumerate(AUGS)}).items():
    html = html.replace('{{' + k + '}}', v)
assert '{{' not in html
open(OUT, 'w', encoding='utf-8').write(html)
print(len(html))
