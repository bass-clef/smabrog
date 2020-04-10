pyinstaller --onefile --clean --icon=icon.ico -n smabrog smabrog.py
python addspec.py
pyinstaller smabrog.spec
