import base64, io, os
from PIL import Image

D = os.path.dirname(os.path.abspath(__file__))
PG = os.path.join(D, 'pg')
OUT = '/mnt/user-data/outputs/ffdbackbone-crispdm-deck.html'

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

PRE = dotplot([
    ('ไม่ pre-train', 61.12, 'base'),
    ('SL · ImageNet-1K', 77.11, 'base'),
    ('SSL · ImageNet-1K (BEiT v2)', 82.70, 'mid'),
    ('SSL · Face (BEiT)', 83.77, 'mid'),
    ('SSL · 1K → Face (FB)', 86.56, 'hi'),
], 55, 90, [55, 60, 65, 70, 75, 80, 85, 90], W=700, label_w=250, row_h=56, axis_title='Average cross-dataset AUC (%)')

SOTA = dotplot([
    ('CADDM', 79.07, 'base'), ('FTCN', 81.34, 'base'), ('SBI', 84.22, 'base'),
    ('TFCU (temporal)', 90.10, 'base'), ('CFB', 89.94, 'mid'), ('CFB++ w/ SBI', 93.35, 'hi'),
], 75, 95, [75, 80, 85, 90, 95], W=640, label_w=170, row_h=50, axis_title='Average AUC บน 5 cross-dataset (%)')

ABL = dotplot([
    ('Dual-branch (sum)', 87.04, 'base'), ('+ L<tspan baseline-shift="sub" font-size="11">dec</tspan>', 88.14, 'mid'), ('+ UFM + L<tspan baseline-shift="sub" font-size="11">EDU</tspan>', 88.78, 'hi'),
], 86, 89.5, [86, 87, 88, 89], W=420, label_w=150, row_h=74, axis_title='Avg. AUC (%)')

UTOM = dotplot([
    ('τ = 0.5', 79.42, 'base'), ('τ จาก UTOM', 82.29, 'hi'),
], 76, 84, [76, 78, 80, 82, 84], W=420, label_w=120, row_h=90, axis_title='Avg. ACC (%)')

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

html = open(os.path.join(D, 'deck2.tpl.html'), encoding='utf-8').read()
for k, v in dict(FIG3=FIG3, FIG3TOP=FIG3TOP, FIG4=FIG4, FIG5=FIG5, PRE=PRE, SOTA=SOTA,
                 ABL=ABL, UTOM=UTOM, FFPP=FFPP, CDF=CDF, SPLIT_BEFORE=SPLIT_BEFORE, SPLIT_AFTER=SPLIT_AFTER, CURVE_AUC=CURVE_AUC, CURVE_LOSS=CURVE_LOSS, CURVE_PEAR=CURVE_PEAR, ROC_FF=ROC_FF, PERMETHOD=PERMETHOD, CROP0=CROPS[0], CROP1=CROPS[1], CROP2=CROPS[2], CROP3=CROPS[3], CROP4=CROPS[4], CDF_AUC=CDF_AUC, CDF_CLS=CDF_CLS, CDF_THR=CDF_THR, XDS=XDS, PERM2=PERM2, ABL2=ABL2, UTOM2=UTOM2).items():
    html = html.replace('{{' + k + '}}', v)
assert '{{' not in html
open(OUT, 'w', encoding='utf-8').write(html)
print(len(html))
