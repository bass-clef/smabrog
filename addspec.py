#coding: utf-8

"""
Name   : *.spec に oneline で必要なフォルダを追記
Auther : Humi@bass_clef_
"""

file_name = 'smabrog.spec'

insert_target = [
	'# -*- mode: python ; coding: utf-8 -*-\n',
	'exe = EXE(pyz,\n',
]
insert_text = [
	'from kivy_deps import sdl2, glew\n',
	"""          Tree('data',prefix='data'),
          Tree('tesseract',prefix='tesseract'),
          Tree('resource',prefix='resource'),
          *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],\n""",
]

change_target = [
	'             hiddenimports=[],\n'
]
change_text = [
	"             hiddenimports=['win32file', 'win32timezone'],\n"
]

with open(file_name, encoding='utf-8') as f:
	lines = f.readlines()

insert_count = 0
change_count = 0
for n, line in enumerate(lines):
	# 追記
	if ( insert_count < len(insert_target) ):
		if (insert_target[insert_count] == line):
			print(f'add line {n}')
			lines.insert(n+1, insert_text[insert_count])
			insert_count += 1

	# 変更
	if ( change_count < len(change_target) ):
		if (change_target[change_count] == line):
			print(f'change line {n}')
			lines[n] = change_text[change_count]
			change_count += 1

	if ( len(insert_target) <= insert_count and len(change_target) <= change_count ):
		break

with open(file_name, mode='w', encoding='utf-8') as f:
	f.writelines(lines)
