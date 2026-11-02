"""Build site/content.js from the Markdown notes so index.html can show them in full.

Run from the repo root after editing any lecture / quiz Markdown:

    python site/build.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = 'https://github.com/Kanvara001/MSC_DPML_2026/blob/main/'

MODULES = [
    ('overview', 'ภาพรวมรายวิชา', 'เกณฑ์คะแนนและแผนกิจกรรม'),
    ('basics', 'พื้นฐาน AI / ML', 'ข้อมูล โมเดล และ LLM'),
    ('dataprep', 'Data Preparation', 'คุณภาพข้อมูล leakage และ preprocessing'),
    ('image', 'Image Data', 'การเกิดภาพดิจิทัล และ preprocessing ภาพ'),
    ('classify', 'Classification & Neural Network', 'Decision tree, Naive Bayes, evaluation, perceptron'),
    ('quiz', 'Quiz', 'ไฟล์ที่ส่งและวิธีทำ'),
    ('research', 'Research Project', 'AI Detection: Deepfake (FFDBackbone)'),
]

# id, module, tag, markdown path, one-line summary, extra links [(label, repo path)], tools
PAGES = [
    ('grading', 'overview', '00', 'course/00_grading-and-schedule.md',
     'สัดส่วนคะแนน เกณฑ์เกรด และกิจกรรม Research Paper', [], []),
    ('lecture-01', 'basics', 'L1', 'course/lectures/lecture_01_intro-ml.md',
     'แนวคิด ML, Deep Learning, CPU/GPU, ข้อมูลและโมเดล', [], []),
    ('lecture-02', 'basics', 'L2', 'course/lectures/lecture_02_llm.md',
     'Discriminative vs Generative, 5 ขั้นตอนการพัฒนา LLM, RAG, Agentic AI', [], []),
    ('lecture-03', 'dataprep', 'L3', 'course/lectures/lecture_03_data-prep-session1.md',
     'Data quality, profiling, splitting, leakage, evaluation hygiene',
     [('เอกสาร Data Preparation (HTML)', 'course/materials/data-preparation/index.html')], []),
    ('lecture-04', 'dataprep', 'L4', 'course/lectures/lecture_04_data-preprocessing.md',
     'Data cleaning, integration, reduction, imbalanced data, PCA',
     [('สไลด์ (PDF)', 'course/slides/03_data-preprocessing_annotated.pdf'),
      ('SMOTE performance (HTML)', 'course/topics/smote/performance.html'),
      ('สรุป SMOTE25 (PDF)', 'course/topics/smote/smote25_data-section-summary.pdf')], []),
    ('lecture-05', 'image', 'L5', 'course/lectures/lecture_05_image-formation.md',
     'กล้อง เซนเซอร์ ภาพการแพทย์ sampling aliasing สี resize/normalize',
     [('สไลด์ (PDF 31 MB)', 'course/slides/image-data_class1_how-a-digital-image-is-formed.pdf')], []),
    ('lecture-06', 'image', 'L6', 'course/lectures/lecture_06_image-preprocessing-training.md',
     'Filtering, CLAHE, the rule, transfer learning, augmentation, pipelines',
     [('สไลด์ (PDF 19 MB)', 'course/slides/image-data_class2_preprocessing-before-and-during-training.pdf')],
     ['gauss']),
    ('lecture-07', 'classify', 'L7', 'course/lectures/lecture_07_classification-basic.md',
     'Decision tree (information gain), Naive Bayes, linear classifier, evaluation',
     [('สไลด์ (PDF)', 'course/slides/08_classification-basic_annotated.pdf'),
      ('Decision Tree interactive (HTML)', 'course/topics/decision-tree/Decision_tree.html'),
      ('Decision Tree slides (HTML)', 'course/topics/decision-tree/Decision_tree_slides.html'),
      ('Decision Tree บทบรรยาย (PDF)', 'course/topics/decision-tree/Decision_tree_lecture-notes.pdf')],
     ['roc']),
    ('lecture-08', 'classify', 'L8', 'course/lectures/lecture_08_neural-network.md',
     'Perceptron learning rule, AND vs XOR, delta rule, gradient descent',
     [('เอกสาร (PDF)', 'course/slides/neural-network_perceptron_annotated.pdf')], ['perceptron']),
    ('quiz-01', 'quiz', 'Q1', 'quizzes/quiz_01.md',
     'Gaussian kernel 5×5 (σ = 1) และการ normalize ด้วย 273',
     [('ไฟล์ที่ส่ง (PDF)', 'quizzes/quiz_01_gaussian-kernel.pdf')], ['gauss']),
    ('quiz-02', 'quiz', 'Q2', 'quizzes/quiz_02.md',
     'Perceptron learning ของ (X1 ∧ X2) ∨ X3 และ metric ของ test set',
     [('ไฟล์ที่ส่ง (PDF)', 'quizzes/quiz_02_perceptron.pdf')], ['perceptron']),
    ('research-overview', 'research', 'RP', 'research-project/README.md',
     'หัวข้อ AI Detection 3 กลุ่ม และ paper ที่เลือก', [], []),
    ('research-ffd', 'research', 'FFD', 'research-project/deepfake-ffdbackbone/README.md',
     'ทำซ้ำ FFDBackbone ตาม CRISP-DM: ขั้นการทดลองและผลหลัก',
     [('สไลด์นำเสนอ (HTML)', 'research-project/deepfake-ffdbackbone/slides/ffdbackbone-crispdm-deck.html'),
      ('Paper (PDF)', 'research-project/deepfake-ffdbackbone/paper/Revisiting_Face_Forgery_Detection_From_Facial_Representation_to_Forgery_Detection.pdf')],
     []),
]

LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)\s]+)\)')


def rewrite_links(text, md_path):
    """Point Markdown links at the right place when the page is shown from the repo root."""
    by_path = {p[3]: p[0] for p in PAGES}
    base = os.path.dirname(md_path)

    def fix(m):
        bang, label, target = m.groups()
        if re.match(r'^(https?:|mailto:|#)', target):
            return m.group(0)
        path, _, frag = target.partition('#')
        resolved = os.path.normpath(os.path.join(base, path)).replace('\\', '/')
        if resolved in by_path:
            new = '#/' + by_path[resolved]
        elif not bang and os.path.isdir(os.path.join(ROOT, resolved)):
            new = GH.replace('/blob/', '/tree/') + resolved
        elif resolved.endswith(('.md', '.csv', '.py', '.ipynb', '.json', '.txt')):
            new = GH + resolved  # GitHub renders these better than the raw file
        else:
            new = resolved + ('#' + frag if frag else '')
        return f'{bang}[{label}]({new})'

    return LINK.sub(fix, text)


def main():
    pages = []
    for pid, mod, tag, path, summary, links, tools in PAGES:
        with open(os.path.join(ROOT, path), encoding='utf-8') as f:
            text = f.read()
        title_match = re.match(r'\s*#\s+(.+)\n', text)
        title = title_match.group(1).strip() if title_match else pid
        body = text[title_match.end():] if title_match else text
        pages.append({
            'id': pid, 'module': mod, 'tag': tag, 'title': title, 'summary': summary,
            'source': GH + path, 'links': [{'label': l, 'href': h} for l, h in links],
            'tools': tools, 'md': rewrite_links(body, path),
        })
    data = {'modules': [{'id': i, 'name': n, 'desc': d} for i, n, d in MODULES], 'pages': pages}
    out = os.path.join(ROOT, 'site', 'content.js')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('// Generated by site/build.py from the Markdown notes. Do not edit by hand.\n')
        f.write('window.COURSE = ' + json.dumps(data, ensure_ascii=False, indent=1) + ';\n')
    print(f'wrote {out} ({len(pages)} pages)')


if __name__ == '__main__':
    main()
