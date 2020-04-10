#coding: utf-8

"""
Name   : *.spec に oneline で必要なフォルダを追記
Auther : Humi@bass_clef_
"""

file_name = 'smabrog.spec'
target = 'exe = EXE(pyz,\n'
insert_text = """          Tree('data',prefix='data'),
          Tree('tesseract',prefix='tesseract'),
          Tree('resource',prefix='resource'),
"""

with open(file_name, encoding='utf-8') as f:
	lines = f.readlines()

for n, line in enumerate(lines):
	if (target == line):
		print(f'add line {n}')
		lines.insert(n+1, insert_text)
		break

with open(file_name, mode='w', encoding='utf-8') as f:
	f.writelines(lines)
