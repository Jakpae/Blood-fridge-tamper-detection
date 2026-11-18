"""ui.html에 core.js와 cfg.json을 넣어 ../demo/index.html 한 파일로 만든다."""
from pathlib import Path
here = Path(__file__).parent
ui = (here / 'ui.html').read_text(encoding='utf-8')
html = ui.replace('/*CORE*/', (here / 'core.js').read_text(encoding='utf-8')).replace('/*CFG*/', (here / 'cfg.json').read_text(encoding='utf-8'))
(here.parent / 'demo' / 'index.html').write_text(html, encoding='utf-8')
print('demo/index.html 생성')
