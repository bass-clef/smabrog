pyinstaller --onefile --clean --icon=icon.ico -n smabrog smabrog.py

rem add *.spec
rem Tree('data',prefix='data'),
rem Tree('tesseract',prefix='tesseract'),
rem Tree('resource',prefix='resource'),
rem later> pyinstaller smabrog.spec
