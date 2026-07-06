import os

files_and_markers = [
    ('src/pipeline_1_ingestion.py',    'LOCAL_RAW_DIR'),
    ('src/pipeline_2_ontology.py',     'CONCURRENCY'),
    ('src/pipeline_3_dynamic_tagging.py', 'BOX_THRESHOLD'),
    ('src/pipeline_3c_l4_attribute_tagging.py', 'pil_img'),
    ('src/pipeline_4_routing_clustering.py', 'random.seed'),
    ('src/pipeline_5_interim_captions.py', 'tightly packed'),
    ('src/pipeline_6_eval_chair.py',   'Absence'),
    ('src/pipeline_7_llm_judge.py',    'CONCURRENCY'),
    ('src/pipeline_8_comparison_report.py', 'chair_metrics_v3.json'),
    ('src/utils/async_gemini.py',      'image.load()'),
]

ok = 0
for fname, marker in files_and_markers:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
        if marker in content:
            print(f'  [OK] {os.path.basename(fname)} -- contains "{marker}"')
            ok += 1
        else:
            print(f'  [FAIL] {os.path.basename(fname)} -- missing "{marker}"')
    except Exception as e:
        print(f'  [ERROR] {os.path.basename(fname)} -- {e}')

# Extra: verify hardcoded score is gone from pipeline_8
with open('src/pipeline_8_comparison_report.py', 'r', encoding='utf-8') as f:
    p8 = f.read()
if 'v3_chair_i_mop   = 41.14' in p8:
    print('  [FAIL] pipeline_8 -- hardcoded score 41.14 still present!')
else:
    print('  [OK] pipeline_8 -- hardcoded scores removed')

print(f'\nResult: {ok}/10 files verified')
