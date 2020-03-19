#coding: utf-8

"""
Name   : smabrog (smash bro. log)
Auther : Humi@bass_clef_

"""

import cv2
from ctypes import *
from ctypes.wintypes import *
import datetime
from collections import defaultdict
import difflib
from enum import IntEnum
from forbiddenfruit import curse
from PIL import Image, ImageGrab
import json
import logging
import matplotlib
import numpy
from collections import OrderedDict
import os
from pathlib import Path
import pyocr
import pyocr.builders
import re
import sys
import time
import traceback
import unicodedata
import urllib
import urllib.request



class Utils:
	@staticmethod
	# PIL型 -> OpenCV型
	def pil2cv(image):
		new_image = numpy.array(image, dtype=numpy.uint8)
		if new_image.ndim == 2:  # モノクロ
			pass
		elif new_image.shape[2] == 3:  # カラー
			new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
		elif new_image.shape[2] == 4:  # 透過
			new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
		return new_image

	# OpenCV型 -> PIL型
	def cv2pil(image):
		new_image = image.copy()
		if new_image.ndim == 2:  # モノクロ
			pass
		elif new_image.shape[2] == 3:  # カラー
			new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
		elif new_image.shape[2] == 4:  # 透過
			new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
		new_image = Image.fromarray(new_image)
		return new_image

	# 無効な文字列の削除
	def replace_invalid_char(word):
		return re.sub(r'[\W|,|.]', '', word)

	# 任意の四角形の中にある何かの輪郭にそって image を加工して返す
	# @param bool filled		True にすると minsize 未満の面積の点を (255,255,255) で塗りつぶす
	def trimming_any_rect(image, gray_image, margin=0, minsize=1e1, maxsize=1e5, filled=False, fill_color=[255,255,255]):
		width = int(image.shape[1])
		height = int(image.shape[0])
		contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		word_x = width
		word_y = height
		word_width = 0
		word_height = 0
		for i in range(0, len(contours)):
			if 0 < len(contours[i]):
				area = cv2.contourArea(contours[i])
				# ノイズの除去
				if (area < minsize):
					if (filled):
						cv2.drawContours(image, contours, i, fill_color, -1)
					continue
				if (maxsize < area):
					if (filled and 1e5 != maxsize):
						cv2.drawContours(image, contours, i, fill_color, -1)
					continue

				rect = contours[i]
				x, y, w, h = cv2.boundingRect(rect)
				word_x = min(word_x, x)
				word_y = min(word_y, y)
				word_width = max(word_width, x+w)
				word_height = max(word_height, y+h)
		result_image = image[max(word_y-margin, 0):min(word_height+margin, height), max(word_x-margin, 0):min(word_width+margin, width)]
		if ( any([shape == 0 for shape in result_image.shape]) ):
			return image
		return result_image

	# image と color_image をマッチングテンプレートで比較して一番確率が高い 確率,座標 を返す
	# @param Image mask_image		指定した場合は image を事前にマスク処理する
	#								is_trans=True の場合は透過色用のマスクとして使用する
	def match_masked_color_image(image, color_image, mask_image=None, is_trans=False, method=cv2.TM_CCOEFF_NORMED):
		convert_image = image.copy()
		if (is_trans):
			if (mask_image is None):
				channels = cv2.split(color_image)
				zero_channel = numpy.zeros_like(channels[0])
				mask = numpy.array(channels[3])
				mask[channels[3] == 0] = 1
				mask[channels[3] == 100] = 0
				transparent_mask = cv2.merge([zero_channel, zero_channel, zero_channel, mask])
			else:
				transparent_mask = mask_image.copy()
				if (transparent_mask.shape[2] == 3):
					transparent_mask = cv2.cvtColor(transparent_mask, cv2.COLOR_RGB2RGBA)
			trans_color_image = color_image.copy()

			if (convert_image.shape[2] == 3):
				convert_image = cv2.cvtColor(convert_image, cv2.COLOR_RGB2RGBA)
			if (trans_color_image.shape[2] == 3):
				trans_color_image = cv2.cvtColor(trans_color_image, cv2.COLOR_RGB2RGBA)

			result = cv2.matchTemplate(convert_image, trans_color_image, cv2.TM_CCORR_NORMED, mask=transparent_mask)
		else:
			if (not mask_image is None):
				convert_image = cv2.bitwise_and(convert_image, mask_image)
			result = cv2.matchTemplate(convert_image, color_image, method)

		_, ratio, _, pos = cv2.minMaxLoc(result)
		return ratio, pos

	# consoleの幅いっぱいに空白をつめて表示 (デフォルトCRすると前の文字が残ってて読みづらいため)
	def width_full_print(message, carriage_return=True, logger_func=None):
		columns, _ = os.get_terminal_size()
		m_len = len(message) + len([(c) for c in message if (unicodedata.east_asian_width(c) in 'FWA')])
		print('\r' if carriage_return else '', message, ' ' * (columns - m_len - 1), end='')

		if (not logger_func is None):
			logger_func(message)

	# putTextさんへ。なんで改行コード反映してくれないの???
	def draw_text_line(image, texts, pos=(0,0), thickness=2, color=(0,0,0), font_scale=1.0, line_margin=3):
		texts = texts.split('\n') if type(texts) == str else texts
		for lines, text in enumerate(texts):
			size, ymin = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
			cv2.putText(image, text, (pos[0], pos[1] + (lines+1)*(size[1]+ymin+line_margin)), cv2.FONT_HERSHEY_DUPLEX,
				font_scale, color, thickness, cv2.LINE_AA)

class FrameState(IntEnum):
	FS_UNKNOWN = 0
	FS_READY_TO_FIGHT = 1
	FS_READY_OK = 2
	FS_WHAT_CHARACTER = 3
	FS_BATTLE = 4
	FS_BATTLE_END = 5
	FS_RESULT = 6
	FS_LOADING = 7

class SmaBroEngine:
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(logging.DEBUG)
		now = datetime.datetime.now().strftime("%Y_%m_%d_%H")
		log_handler = logging.FileHandler(filename=f'log/{now}.log')
		log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)8s %(message)s'))
		self.logger.addHandler(log_handler)

	# コンフィグの読み込み
	def _load_config(self):
		self.config_path = 'config.json'

		with open(self.config_path, encoding='utf-8') as file:
			config_string = file.read()
			self.config = json.loads(config_string, object_pairs_hook=OrderedDict)[0]

	# リソースの読み込み/作成
	def _load_resources(self):
		# 外部リソース
		tools = pyocr.get_available_tools()
		if len(tools) == 0:
			self.logger.error('No OCR tool found')
			sys.exit(1)
		self.ocr_tool = tools[0]

		# 変数リソース
		self.chara = self.config['resource']['character']
		self.team_color_list = ['red', 'blue', 'yellow', 'green']
		self.resource_size = { 'width':640, 'height':360 }
		self.battle_rate = defaultdict()
		self.back_power = 0
		self.battle_streak_rate = 0	# 正数で連勝数,負数で連敗数
		self.battle_history = []
		#self.time_zero = datetime.datetime(1900, 1, 1, 0, 0)

		# 画像リソース
		self.ready_to_fight_mask = cv2.imread(+'resource/ready_to_fight_mask.png', cv2.IMREAD_UNCHANGED)
		self.ready_to_fight_color = list(range(2))
		for i in range(2):
			self.ready_to_fight_color[i] = cv2.imread(+f'resource/ready_to_fight_color_{i}.png', cv2.IMREAD_UNCHANGED)
		self.ready_to_fight_name_power_mask = cv2.imread(+'resource/ready_to_fight_name_power_mask.png')

		self.ready_ok_mask = cv2.imread(+'resource/ready_ok_mask.png')
		self.ready_ok_color = cv2.imread(+'resource/ready_ok_color.png')
		self.ready_ok_name_mask = cv2.imread(+'resource/ready_ok_name_mask.png', cv2.IMREAD_GRAYSCALE)
		self.entry_name_unknown = cv2.imread(+'resource/entry_name_unknown.png', cv2.IMREAD_GRAYSCALE)
		self.with_4_battle_mask = cv2.imread(+'resource/with_4_battle_mask.png')
		self.with_4_battle_color = cv2.imread(+'resource/with_4_battle_color.png')

		self.vs_mask = cv2.imread(+'resource/vs_mask.png')
		self.vs_color = cv2.imread(+'resource/vs_color.png')
		self.group_team_color = dict(zip( self.team_color_list, list(range(4)) ))
		for color in self.team_color_list:
			self.group_team_color[color] = cv2.imread(+f'resource/group_team_{color}_color.png')
		self.rule_smash_color = cv2.imread(+'resource/rule_smash_color.png')
		self.rule_smash_or_team_mask = cv2.imread(+'resource/rule_smash_or_team_mask.png')

		self.battle_time_zero_mask = cv2.imread(+'resource/battle_time_zero_mask.png')
		self.battle_time_zero_color = cv2.imread(+'resource/battle_time_zero_color.png')
		self.battle_time_mask = cv2.imread(+'resource/battle_time_mask.png', cv2.IMREAD_GRAYSCALE)

		player_zeros = cv2.imread(+'resource/player_zeros.png')
		size = [34, 25]
		self.player_zeros_mask = player_zeros[0:size[1], 0:size[0]]
		self.player_zeros_color = player_zeros[size[1]:size[1]*2, 0:size[0]]
		plus_minus_ones = cv2.imread(+'resource/plus_minus_ones.png')
		size = [19, 14]
		self.minus_one_mask = plus_minus_ones[0:size[1], 0:size[0]]
		self.plus_one_mask = plus_minus_ones[0:size[1], size[0]:size[0]*2]
		self.minus_one_color = plus_minus_ones[size[1]:size[1]*2, 0:size[0]]
		self.plus_one_color = plus_minus_ones[size[1]:size[1]*2, size[0]:size[0]*2]

		self.stock_hyphen_mask = cv2.imread(+'resource/stock_hyphen_mask.png')
		self.stock_hyphen_color_black = cv2.imread(+'resource/stock_hyphen_color_black.png')
		self.stock_hyphen_color_white = cv2.imread(+'resource/stock_hyphen_color_white.png')
		self.stock_number_mask = cv2.imread(+'resource/stock_number_mask.png', cv2.IMREAD_GRAYSCALE)

		self.game_set_mask = cv2.imread(+'resource/game_set_mask.png')
		self.game_set_color = cv2.imread(+'resource/game_set_color.png')
		self.time_up_mask = cv2.imread(+'resource/time_up_mask.png')
		self.time_up_color = cv2.imread(+'resource/time_up_color.png')

		self.result_power_mask = cv2.imread(+'resource/result_power_mask.png')
		self.result_player_order_color = list(range(4 + 2))
		self.result_player_order_mask = list(range(4 + 2))
		for order in range(4):
			self.result_player_order_color[order] = cv2.imread(+f'resource/result_player_order_{order+1}_color.png')
			self.result_player_order_mask[order] = cv2.imread(+f'resource/result_player_order_{order+1}_mask.png')
		for order in range(2):
			self.result_player_order_color[4 + order] = cv2.imread(+f'resource/result_player_order_big_{order+1}_color.png')
			self.result_player_order_mask[4 + order] = cv2.imread(+f'resource/result_player_order_big_{order+1}_mask.png')

		self.battle_retry_mask = cv2.imread(+'resource/battle_retry_mask.png')
		self.battle_retry_color = cv2.imread(+'resource/battle_retry_color.png')

	# 初期化後起動のみ
	def _load_first_only(self):
		if ( 0 == self.config['capture']['width'] or 0 == self.config['capture']['height'] ):
			base_resolution = [16, 9]
			# 40, 80, 100, 120
			for magnification in [40, 53, 80, 100, 120] + list(range(41, 53)) + list(range(54, 80)) + list(range(81, 100)) + list(range(101, 120)):
				self.config['option']['find_capture_area'] = True
				self.config['option']['exit_not_found_capture'] = False
				resolution = [ base_resolution[0] * magnification, base_resolution[1] * magnification ]
				self.config['capture']['width'] = resolution[0]
				self.config['capture']['height'] = resolution[1]
				self._init_capture_window(in_loop=True)
				Utils.width_full_print(f'\rsearch... {resolution[0]}x{resolution[1]} - {self.ratio}%')
				if (0.99 <= self.ratio):
					break

			if (121 != base_resolution):
				Utils.width_full_print(
					f"resolution is {self.config['capture']['width']}x{self.config['capture']['height']}",
					logger_func=self.logger.info)

	# キャプチャに関する初期化
	def _init_capture_window(self, in_loop=False):
		# このためだけに透過色付きで読み込んでいたので,今後のために RGBA -> RGB 変換をする
		ready_to_fight_trans_mask = self.ready_to_fight_mask.copy()
		ready_to_fight_trans_color = self.ready_to_fight_color.copy()
		for key, color_image in enumerate(self.ready_to_fight_color):
			self.ready_to_fight_color[key] = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
		self.ready_to_fight_mask = cv2.cvtColor(self.ready_to_fight_mask, cv2.COLOR_RGBA2RGB)

		# [READY to FIGHT]によって自動判別
		self.ratio = 0.0
		if (self.config['option']['find_capture_area']):
			desktop_width = ctypes.windll.user32.GetSystemMetrics(0)
			desktop_height = ctypes.windll.user32.GetSystemMetrics(1)
			capture_erea = (0, 0, desktop_width, desktop_height)
			capture_image = Utils.pil2cv(ImageGrab.grab(bbox=capture_erea))
			convert_image = capture_image
			width = int(self.config['capture']['width'])
			height = int(self.config['capture']['height'])

			width_magnification = 1.0
			height_magnification = 1.0
			if ( (self.resource_size['width'], self.resource_size['height']) != (width, height) ):
				width_magnification = self.resource_size['width'] / width
				height_magnification = self.resource_size['height'] / height
				new_width = int(width_magnification * desktop_width)
				new_height = int(height_magnification * desktop_height)
				convert_image = cv2.resize( convert_image, dsize=(new_width, new_height) )
				# 下の座標をもとに戻すための倍率の計算
				width_magnification = width / self.resource_size['width']
				height_magnification = height / self.resource_size['height']

			# 解像度を見つけてから誤差を調整して探索
			for key, color_image in enumerate(ready_to_fight_trans_color):
				_, pos = Utils.match_masked_color_image(convert_image, color_image,
					ready_to_fight_trans_mask, is_trans=True)
				capture_area_image = convert_image[ pos[1]:int(pos[1]+self.resource_size['height']), pos[0]:int(pos[0]+self.resource_size['width']) ]
				is_ready_frame = self._is_ready_frame(capture_area_image)
				cv2.rectangle(convert_image,
					(pos[0], pos[1], self.resource_size['width'], self.resource_size['height']),
					255 if is_ready_frame else (0, 0, 255), 1)
				if ( not is_ready_frame ):
					continue

				p_ratio = {}
				for add_y in [-1, 0, 1]:
					for add_x in [-1, 0, 1]:
						x = int((pos[0]) * width_magnification) + add_x
						y = int((pos[1]) * height_magnification) + add_y
						if (x < 0 or y < 0 or desktop_height < y+height or desktop_width < x+width):
							continue

						capture_area_image = cv2.resize(capture_image[ y:y+height, x:x+width ], dsize=(self.resource_size['width'], self.resource_size['height']))
						if ( self._is_ready_frame(capture_area_image) ):
							p_ratio[tuple([add_x, add_y])] = self.ratio

				p, ratio = max( p_ratio.items(), key=lambda x:[1] )
				x = int(pos[0] * width_magnification) + p[0]
				y = int(pos[1] * height_magnification) + p[1]
				self.ratio = ratio
				self.config['capture']['x'] = x
				self.config['capture']['y'] = y
				if (self.config['option']['found_capture_area_fixed']):
					self.config['option']['find_capture_area'] = False
					cv2.imwrite('found_capture_area_fixed.png', convert_image)
				self.logger.info('found capture area {0}x{1}'.format(self.config['capture']['x'], self.config['capture']['y']))
				break

			if ( not is_ready_frame and not in_loop ):
				Utils.width_full_print('not found capture area.')
				if ( not self.config['option']['find_capture_area'] ):
					Utils.width_full_print('change config.option.find_capture_area to true.')
					self.config['option']['find_capture_area'] = True
					self.config['option']['found_capture_area_fixed'] = True
				if (self.config['option']['exit_not_found_capture']):
					sys.exit(0)

	# タイトルからウィンドウを捕捉 (これより, _init_capture_window ぶん回したほうが早いｗｗｗ おまけ程度)
	def _capture_window_title(self, title):
		handle = ctypes.windll.user32.FindWindowW(None, title)
		if (0 == handle):
			if (self.config['option']['exit_not_found_capture']):
				self.logger.error(f'not found window title "{title}"')
				sys.exit(0)
			return
		rect = RECT()
		ctypes.windll.user32.GetWindowRect(handle, pointer(rect))
		self.capture_erea = (rect.left, rect.top, rect.right, rect.bottom)

	# ウィンドウの捕捉
	def _capture_window(self):
		x = int(self.config['capture']['x'])
		y = int(self.config['capture']['y'])
		width = int(self.config['capture']['width'])
		height = int(self.config['capture']['height'])
		self.capture_erea = (x, y, x + width, y + height)
		if ('' != self.config['capture']['title']):
			self._capture_window_title(self.config['capture']['title'])
		area = self.capture_erea[2] * self.capture_erea[3]

		capture_image = Utils.pil2cv(ImageGrab.grab(bbox=self.capture_erea))
		capture_image = cv2.resize(capture_image, dsize=(self.resource_size['width'], self.resource_size['height']))
		return capture_image

	# スマブラの戦歴に関わる変数をデフォルトする
	def _default_battle_params(self, player_count=2):
		self.ratio = 0.0
		self.player_order_ratio = [0.99] * player_count

		self.frame_state		= FrameState.FS_UNKNOWN
		self.entry_name			= [''] * player_count
		self.rule_name			= 'stock'	# stock or time
		self.group_name			= 'smash'	# smash or team
		self.group_color		= [''] * player_count
		self.chara_came			= [''] * player_count
		self.battle_time		= [datetime.datetime(1900, 1, 1, 0, 0) for _ in range(2)]	# MM:SS.MS の書式
		self.player_damage		= [-1.0] * player_count
		self.defeated_player	= [-1] * player_count
		self.result_stock		= {'max':[-1] * player_count, 'count':[-1] * player_count}
		"""
			if rule_name == 'stock': max is max stock, count is stock
			if rule_name == 'time': max is add count, count is sub count
		"""
		self.power			= [0] * player_count
		self.player_order	= [-1] * player_count
		self.player_count	= player_count

		# ('', '', '', '') の作成だけしておく (読み取り先が無いと怒られるので)
		self._store_battle_rate(self.chara_came)

	""" capture系 hoge frame を検出した時にする処理 検出時間(昇順) """
	# [READY to FIGHT]画面
	def _capture_ready_to_fight_frame(self, capture_image):
		if (self.frame_state == FrameState.FS_READY_TO_FIGHT):
			return
		if (self.power[0] < 0):
			# 初期値 or エラーは計算しない
			return
		self.logger.debug(f'_capture_ready_to_fight{self.ratio}')

		width = int(capture_image.shape[1])
		height = int(capture_image.shape[0])
		convert_image = cv2.bitwise_and(capture_image, self.ready_to_fight_name_power_mask)
		capture_gray_image = cv2.cvtColor(convert_image, cv2.COLOR_RGB2GRAY)

		# 自分の戦闘力の増減値
		power_area_image = convert_image[int(height/2):height, int(width/2):width]
		gray_power_area_image = capture_gray_image[int(height/2):height, int(width/2):width]
		power_area_image = Utils.trimming_any_rect(power_area_image, gray_power_area_image)

		gray_power_area_image = cv2.cvtColor(power_area_image, cv2.COLOR_RGB2GRAY)
		_, gray_power_area_image = cv2.threshold(gray_power_area_image, 150, 255, cv2.THRESH_BINARY_INV)
		power_area_image = Utils.trimming_any_rect(power_area_image, gray_power_area_image, 1, 0)
		#cv2.imshow('capture_ready_to_fight_power', power_area_image)	# debug only
		power = self.ocr_tool.image_to_string(
			Utils.cv2pil(power_area_image), lang='eng',
			builder=pyocr.builders.DigitBuilder(tesseract_layout=6) )
		if ('' == power):
			return

		power = int(Utils.replace_invalid_char(power))
		self.power[0] = power
		power_diff = self.power[0] - self.back_power
		self.back_power = self.power[0]

		diff_message = ' diff: {sign}{power_diff}'.format(sign='' if power_diff < 0 else '+', power_diff=power_diff)
		self.logger.info(diff_message)
		print('\n', diff_message)

	# [Entry -> 準備OK]画面
	def _capture_ready_ok_frame(self, capture_gray_image):
		self.logger.debug(f'_ready_ok_frame{self.ratio}')

		width = int(capture_gray_image.shape[1])
		height = int(capture_gray_image.shape[0])
		convert_image = cv2.bitwise_and(capture_gray_image, self.ready_ok_name_mask)
		convert_image = cv2.bitwise_not(convert_image)

		name_area_image = list(range(2))
		name_area_image[0] = convert_image[int(height/2):height, 0:int(width/4)]
		name_area_image[1] = convert_image[int(height/2):height, int(width/4):int(width/2)]
		gray_convert_image = cv2.bitwise_not(convert_image)
		gray_name_area_image = list(range(2))
		gray_name_area_image[0] = gray_convert_image[int(height/2):height, 0:int(width/4)]
		gray_name_area_image[1] = gray_convert_image[int(height/2):height, int(width/4):int(width/2)]

		entry_name = ['', '']
		for key, image in enumerate(name_area_image):
			image = Utils.trimming_any_rect(image, gray_name_area_image[key])
			_, convert_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
			convert_image = cv2.bitwise_not(convert_image)
			image = Utils.trimming_any_rect(image, convert_image, margin=2)

			entry_name[key] = '' + self.ocr_tool.image_to_string(
				Utils.cv2pil(image), lang='jpn',
				builder=pyocr.builders.TextBuilder(tesseract_layout=6) )
			entry_name[key] = Utils.replace_invalid_char(entry_name[key])

			if (image.shape[0] < self.entry_name_unknown.shape[0] or image.shape[1] < self.entry_name_unknown.shape[1]):
				continue
			# 名前「????」と一致するかどうか (? を 4回検出)
			res = cv2.matchTemplate(image, self.entry_name_unknown, cv2.TM_CCOEFF_NORMED)
			loc = numpy.where(0.9 <= res)
			if (4 == len(loc[::-1][0])):
				# 名前検出したらなぜか「つののの」とか「みつププ」になるのでｗ、まぁ今後連戦監視とかに使うかもしれないし一応「????」にしておく
				entry_name[key] = '????'

		self.entry_name = entry_name
		self.logger.debug('entry_name=%s', entry_name)

	# [キノコ vs タケノコ]のようにキャラクター名が取得できる画面
	def _capture_character_name(self, capture_gray_image):
		self.logger.debug(f'_capture_character_name{self.ratio}')
		width = int(capture_gray_image.shape[1])
		height = int(capture_gray_image.shape[0])
		
		# 近似白黒処理して,輪郭捕捉して,キャラ名取得
		_, convert_image = cv2.threshold(capture_gray_image, 250, 255, cv2.THRESH_BINARY)
		gray_convert_image = convert_image
		convert_image = cv2.bitwise_not(convert_image)
		name_area_image = list(range(self.player_count))
		gray_name_area_image = list(range(self.player_count))
		one_width = width / self.player_count
		margin_width = one_width / 6
		for key, _ in enumerate(name_area_image):
			name_area_image[key] = convert_image[0:int(height/7), int(one_width*key+margin_width):int(one_width*(key+1))]
			gray_name_area_image[key] = gray_convert_image[0:int(height/7), int(one_width*key+margin_width):int(one_width*(key+1))]

		chara_came = [''] * self.player_count
		for key, image in enumerate(name_area_image):
			image = Utils.trimming_any_rect(image, gray_name_area_image[key], 5)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
			#cv2.imshow('name_area_image'+ str(key), name_area_image[key])		# debug only
			chara_came[key] = '' + self.ocr_tool.image_to_string(
				Utils.cv2pil(image), lang='eng',
				builder=pyocr.builders.TextBuilder() )
			chara_came[key] = Utils.replace_invalid_char(chara_came[key])
		if ( all(['' == name for name in chara_came]) ):
			return

		# 画像をごちゃごちゃしてもまだ誤字/脱字/不一致があるので一致率が 50% 以上で一番高いものにする
		name_ratio = [0.5] * self.player_count
		for key, name in enumerate(chara_came):
			name = name.upper()
			for chara, jpname in self.chara.items():
				ratio = difflib.SequenceMatcher(None, name, chara).ratio()
				if (name_ratio[key] <= ratio):
					self.chara_came[key] = jpname
					name_ratio[key] = ratio
			# 誤検知しまくったキャラ or 新キャラの為に残しておく
			if (self.chara_came[key] == ''):
				self.chara_came[key] = chara_came[key] +'?'

		# なかったら作成だけしておく
		self._store_battle_rate(self.chara_came)

		self.logger.info('with%s=%s=>%s,%s', self.player_count, chara_came, self.chara_came, name_ratio)

	# [GO!]画面の右上の [.00] の数字
	def _capture_go_frame(self, capture_gray_image):
		# 開始時間が不定なので,最初の数字を取得してそれを開始時間とする
		convert_image = cv2.bitwise_and(capture_gray_image, self.battle_time_mask)
		_, mask_image = cv2.threshold(convert_image, 200, 255, cv2.THRESH_BINARY)

		gray_convert_image = cv2.bitwise_and(capture_gray_image, mask_image)
		battle_time_area_image = Utils.trimming_any_rect(
			capture_gray_image, gray_convert_image, 0, 1e2, filled=True)

		_, convert_image = cv2.threshold(battle_time_area_image, 200, 255, cv2.THRESH_BINARY_INV)

		gray_convert_image = cv2.bitwise_not(battle_time_area_image)
		battle_time_area_image = cv2.bitwise_and(gray_convert_image, convert_image)
		#		battle_time_area_image = cv2.bitwise_not(battle_time_area_image)

		#		battle_time_area_image = Utils.trimming_any_rect(
		#			battle_time_area_image, convert_image, 2, 1e3, filled=True, fill_color=[0,0,0])

		cv2.imshow('battle_time_area_image', gray_convert_image)

		time = self.ocr_tool.image_to_string(
			Utils.cv2pil(battle_time_area_image), lang='eng',
			builder=pyocr.builders.DigitBuilder(tesseract_layout=6) )
		Utils.width_full_print(f'\rtime={time}')

		try:
			if (0 == int(time)):
				return
		except ValueError:
			return

		self.battle_time[0] = datetime.datetime(1900, 1, 1, 0, int(time[0]))

	# [N - N]画面
	def _capture_stock_number_frame(self, capture_gray_image):
		self.logger.debug(f'_capture_stock_number_frame{self.ratio}')
		width = int(capture_gray_image.shape[1])
		height = int(capture_gray_image.shape[0])
		# ストック数の上限が不定なので、マスク自体を作成してから摘出する
		convert_image = cv2.bitwise_and(capture_gray_image, self.stock_number_mask)
		_, mask_image = cv2.threshold(convert_image, 250, 255, cv2.THRESH_BINARY)

		convert_image = cv2.bitwise_and(capture_gray_image, mask_image)
		convert_image = cv2.bitwise_not(convert_image)

		stock_area_image = list(range(2))
		stock_area_image[0] = convert_image[0:height, 0:int(width/2)]
		stock_area_image[1] = convert_image[0:height, int(width/2):width]
		gray_convert_image = cv2.bitwise_not(convert_image)
		gray_stock_area_image = list(range(2))
		gray_stock_area_image[0] = gray_convert_image[0:height, 0:int(width/2)]
		gray_stock_area_image[1] = gray_convert_image[0:height, int(width/2):width]
		result_stock = list(range(2))
		for key, val in enumerate(stock_area_image):
			stock_area_image[key] = Utils.trimming_any_rect(
				stock_area_image[key], gray_stock_area_image[key], 5, 1e3, filled=True)
			#cv2.imshow('stock_area_image'+ str(key), stock_area_image[key])		# debug only

			stock = self.ocr_tool.image_to_string(
				Utils.cv2pil(stock_area_image[key]), lang='eng',
				builder=pyocr.builders.DigitBuilder(tesseract_layout=10) )
			if ('' == stock):
				continue
			result_stock[key] = int( Utils.replace_invalid_char(stock) )
		if ( any([ -1 == stock for stock in result_stock ]) ):
			return

		if ( all([ -1 == stock for stock in self.result_stock['max'] ]) ):
			if ( 2 <= abs(result_stock[0] - result_stock[1]) ):
				return	# 差がありすぎるものは誤検出として扱う

			max_stock = max(result_stock)
			if ( self.config['option']['online_rule'] ):
				if ( 3 < max_stock ):
					# online rule : 4ストック以上はありえない
					return
					
			self.result_stock['max'] = [max_stock] * len(result_stock)
			self.result_stock['count'] = self.result_stock['max']
			self.logger.debug('stock max=%s', self.result_stock['max'])

		f = all([ result_stock[key] == stock for key, stock in enumerate(self.result_stock['count']) ])
		if ( f ):
			return	# 2回目以降の検出

		diff = 0
		for key, stock in enumerate(self.result_stock['count']):
			diff += abs(stock - result_stock[key])
		if (2 <= diff):
			self.logger.debug('stock diff/stock=%s,%s', diff, result_stock)
			return	# 差分が 2 以上のものは誤検出として扱う(一度に2ストック以上減ることは無いよね?)

		self.result_stock['count'] = result_stock
		self.logger.debug('change stock=%s', self.result_stock)

	# 結果画面(戦闘力が見える)
	def _capture_result_frame(self, capture_image, capture_gray_image, player):
		if (4 == self.player_count):
			# 戦闘力の数字の画面はおそらく捕捉しきれないのであきらめるんご！
			return
		
		# 戦闘力の検出
		width = int(capture_image.shape[1])
		height = int(capture_image.shape[0])
		power_area_image = list(range(2))
		gray_power_area_image = list(range(2))
		convert_image = cv2.bitwise_and(capture_image, self.result_power_mask)
		power_area_image[0] = convert_image[int(height/4):int(height/2), 0:int(width/20*9)]
		power_area_image[1] = convert_image[int(height/4):int(height/2), int(width/20*11):width]
		convert_image = cv2.bitwise_and(capture_gray_image, cv2.cvtColor(self.result_power_mask, cv2.COLOR_RGB2GRAY))
		gray_power_area_image[0] = convert_image[int(height/4):int(height/2), 0:int(width/20*9)]
		gray_power_area_image[1] = convert_image[int(height/4):int(height/2), int(width/20*11):width]
		# 爆速+連打ニキがいて検出できなくなるので逆順で評価
		for key, image in enumerate(power_area_image):
			image = Utils.trimming_any_rect(image, gray_power_area_image[key])

			convert_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			_, convert_image = cv2.threshold(convert_image, 127, 255, cv2.THRESH_BINARY_INV)
			image = Utils.trimming_any_rect(image, convert_image, 1, 1)

			power = self.ocr_tool.image_to_string(
				Utils.cv2pil(image), lang='eng',
				builder=pyocr.builders.DigitBuilder(tesseract_layout=6) )
			#cv2.imshow('power_area_image'+ str(key), image)		# debug only
			#cv2.imshow('gray_power_area_image'+ str(key), convert_image)		# debug only
			try:
				self.power[key] = int(Utils.replace_invalid_char(power))
			except ValueError:
				pass


	""" is系 (watch含む) 検出時間(昇順) """
	# [READY to FIGHT]画面の検出
	def _is_ready_frame(self, capture_image):
		self.ratio = 0.0
		for color_image in self.ready_to_fight_color:
			ratio, _ = Utils.match_masked_color_image(capture_image, color_image, self.ready_to_fight_mask)
			if (self.ratio < ratio):
				self.ratio = ratio
		self.ratio = round(float(self.ratio), 3)
		return 0.99 <= self.ratio	# READY to FIGHT 画面はカーソルが混じることがあるので低めに設定

	# [対戦相手募集中 -> もうすぐ開始]画面の検出
	def _is_ready_ok_frame(self, capture_image):
		self.ratio, _ = Utils.match_masked_color_image(capture_image, self.ready_ok_color, self.ready_ok_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.95 <= self.ratio

	# 4人版の[準備OK]画面の検出
	def _is_with_4_battle_frame(self, capture_image):
		self.ratio, _ = Utils.match_masked_color_image(capture_image, self.with_4_battle_color, self.with_4_battle_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.95 <= self.ratio

	# [spam vs ham]画面かどうか
	def _is_vs_frame(self, capture_image):
		self.ratio, _ = Utils.match_masked_color_image(capture_image, self.vs_color, self.vs_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.97 <= self.ratio

	# [紅蒼乱闘戦]画面かどうかをついでに返す
	def _watch_smash_or_team_frame(self, capture_image):
		convert_image = capture_image
		convert_image = cv2.bitwise_and(convert_image, self.rule_smash_or_team_mask)
		if ('smash' == self.group_name):
			self.ratio, _ = Utils.match_masked_color_image(convert_image, self.rule_smash_color)
			self.ratio = round(float(self.ratio), 3)
			if (0.98 <= self.ratio):
				# 乱闘形式なので特に変更点はない
				self.group_color = list(self.team_color_list)
				return True

		# チーム色を特定する
		color_pos = [27, 47]
		add_pos = [160, 0]
		size = [80, 4]
		for player in range(self.player_count):
			self.ratio = ratio = 0.0
			color_image = convert_image[ color_pos[1]:color_pos[1]+size[1], color_pos[0]:color_pos[0]+size[0] ]
			#cv2.imshow(f'self.group_team_color[{player}]', color_image)	# debug only
			for color in self.team_color_list:
				ratio, _ = Utils.match_masked_color_image(color_image, self.group_team_color[color])
				ratio = round(float(ratio), 3)
				if (self.ratio < ratio):
					self.ratio = ratio
					self.group_color[player] = color

			if (self.ratio < 0.95):
				# そもそも _watch_smash_or_team_frame 画面でない可能性があるので,その時はすぐ返す
				break
			color_pos[0] += add_pos[0]

		if ( all(['' != color for color in self.group_color]) ):
			self.group_name = 'team'
			return True

		# 誤検出,初期化して返す
		self.group_color = [''] * self.player_count
		return False

	# [GO!]画面かどうか
	def _is_go_frame(self, capture_image):
		self.ratio, _ = Utils.match_masked_color_image(capture_image, self.battle_time_zero_color, self.battle_time_zero_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.98 <= self.ratio

	# プレイヤー毎の [0.0%] を監視
	def _watch_player_zero_damage(self, capture_image):
		convert_image = capture_image
		player_pos = [0, 0]
		add_pos = [0, 0]
		size = [34, 25]

		if (2 == self.player_count):
			return
		elif (4 == self.player_count):
			player_pos = [126, 307]
			add_pos = [136, 0]

		for player in range(self.player_count):
			zero_area_image = convert_image[ player_pos[1]:player_pos[1]+size[1], player_pos[0]:player_pos[0]+size[0] ]
			player_pos[0] += add_pos[0]
			ratio, _ = Utils.match_masked_color_image(zero_area_image, self.player_zeros_color, self.player_zeros_mask)
			if (0.98 <= ratio and 0.0 != self.player_damage[player]):
				self.player_damage[player] = 0.0
				if (-1 != self.defeated_player[player]):
					self.defeated_player[player] = -1
			elif (0.0 == self.player_damage[player]):
				self.player_damage[player] = 1.0

	# ストックが変動した[-1/+1]を探して反映させる
	def _watch_stock_changed_frame(self, capture_image):
		convert_image = capture_image
		player_pos = list(range(self.player_count))
		ones_size = [19, 14]
		size = [25, 14]

		if (2 == self.player_count):
			return
		elif (4 == self.player_count):
			player_pos = [ [116,296], [251,296], [388,296], [523,296] ]

		minus_player = plus_player = -1
		for player, pos in enumerate(player_pos):
			if ( self.player_damage[player] < 0.0 ):
				# [-1]受けた最中のプレイヤーは[0.0%]に戻るまで探さない
				continue

			ones_area_image = convert_image[ pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0] ]
			minus_ratio, p = Utils.match_masked_color_image(ones_area_image, self.minus_one_color, self.minus_one_mask, is_trans=True)
			if (0.4 <= minus_ratio):
				minus_image = ones_area_image[0:ones_size[1], p[0]:p[0]+ones_size[0]]
				#cv2.imshow(f'minus_image{player}', minus_image)
				if (minus_image.shape[1] == ones_size[0] and minus_image.shape[0] == ones_size[1]):
					minus_ratio, _ = Utils.match_masked_color_image(minus_image, self.minus_one_color, self.minus_one_mask)
					if (0.98 <= minus_ratio):
						minus_player = player

			plus_ratio, p = Utils.match_masked_color_image(ones_area_image, self.plus_one_color, self.plus_one_mask, is_trans=True)
			if (0.4 <= plus_ratio):
				plus_image = ones_area_image[0:ones_size[1], p[0]:p[0]+ones_size[0]]
				#cv2.imshow(f'plus_image{player}', plus_image)
				if (plus_image.shape[1] == ones_size[0] and plus_image.shape[0] == ones_size[1]):
					plus_ratio, _ = Utils.match_masked_color_image(plus_image, self.plus_one_color, self.plus_one_mask)
					if (0.98 <= plus_ratio):
						plus_player = player

		if (-1 != minus_player and -1 != plus_player):
			# 同じ相手同士のストックの変動は 前回ストックが減った相手のダメージが一度 0.0% を観測しないと計算しない
			if ( self.defeated_player[minus_player] != plus_player ):
				if ('time' != self.rule_name):
					self.rule_name = 'time'
				if ( -1 == self.result_stock['count'][minus_player] ):
					self.result_stock['count'][minus_player] = 0
				if ( -1 == self.result_stock['max'][minus_player] ):
					self.result_stock['max'][minus_player] = 0

				self.defeated_player[minus_player] = plus_player
				self.player_damage[minus_player] = -1.0
				self.result_stock['count'][minus_player] += 1
				self.result_stock['max'][plus_player] += 1

	# ストックが表示されてる画面か
	def _is_stock_number_frame(self, capture_image):
		black_ratio, _ = Utils.match_masked_color_image(capture_image, self.stock_hyphen_color_black, self.stock_hyphen_mask)
		white_ratio, _ = Utils.match_masked_color_image(capture_image, self.stock_hyphen_color_white, self.stock_hyphen_mask)
		self.ratio = round(float(max(black_ratio, white_ratio)), 3)
		return 0.98 <= self.ratio

	# [GAME SET][TIME UP]画面かどうか
	def _is_game_end_frame(self, capture_image):
		game_set_ratio, _ = Utils.match_masked_color_image(capture_image, self.game_set_color, self.game_set_mask)
		time_up_ratio, _ = Utils.match_masked_color_image(capture_image, self.time_up_color, self.time_up_mask)
		self.ratio = round(float(max(game_set_ratio, time_up_ratio)), 3)
		return 0.97 <= self.ratio

	# 結果画面(戦闘力が見える)かどうかをついでに返す
	def _watch_reuslt_frame(self, capture_image, capture_gray_image):
		# この画面の検出率がすこぶる悪い

		if (2 == self.player_count):
			order_pos = [ [200, 0], [470, 0] ]
		elif (4 == self.player_count):
			order_pos = [ [90, 0], [250, 0], [420, 0], [580, 0] ]

		for reverse_player, pos in enumerate(reversed(order_pos)):
			player = self.player_count - reverse_player - 1

			# 順位の検出
			convert_image =  capture_image[pos[1]:(100+pos[1]), pos[0]:(100+pos[0])]
			order_color_len = len(self.result_player_order_color)
			order_ratio = [-1] * order_color_len
			pos = [0,0] * order_color_len
			for order in range(order_color_len):
				ratio, p = Utils.match_masked_color_image(convert_image, self.result_player_order_color[order],
					self.result_player_order_mask[order], is_trans=True)
				order_ratio[order] = round(float(ratio), 3)
				pos[order] = p

			order = int(numpy.argmax(order_ratio))
			self.ratio = order_ratio[order]
			order = order % 4	# 順位のテンプレートに大きさが異なるものがあるため

			# 戦闘力だけは順位が決まっても取得し続ける
			self._capture_result_frame(capture_image, capture_gray_image, player)

			if (self.player_order_ratio[player] <= self.ratio and
				self.player_order[player] != int( 1 + order ) and
				order in range(0, self.player_count) ):
				# 同時に結果が捕捉できるタイミングが違うのでプレイヤーごとに処理 (かつ片方が取れるかもしれない状況ということはもう片方が取れる確率も高いので power 判定はもう片方のも行う)
				# しかもより確率が高い order が検出された場合,更新する
				#Utils.draw_text_line(convert_image, f'{ratio}\n{int(1 + order )}', thickness=1,color=(255,255,127), font_scale=0.4)		# debug only
				#cv2.imshow('_watch_reuslt_frame_masked{0}'.format(player), convert_image)	# debug only

				self.player_order_ratio[player] = self.ratio
				self.player_order[player] = int( 1 + order )
				self.logger.info('player,order,power=%s,%s,%s', player, self.player_order[player], self.power[player])

		return False

	# 「同じ相手との再戦を希望しますか？」画面か
	def _is_battle_retry_frame(self, capture_image):
		if ( all([(-1 == order) for order in self.player_order]) ):
			# 順位が全く取れていない場合は何もしない(※これはイメージです の画像くらい違うものを検出してる可能性があるため)
			return False

		self.ratio, _ = Utils.match_masked_color_image(capture_image, self.battle_retry_color, self.battle_retry_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.97 <= self.ratio


	# frame がどこかを特定して応じて処理をする
	def _capture_any_frame(self, capture_image):

		gray_image = cv2.cvtColor(capture_image, cv2.COLOR_RGB2GRAY)

		#cv2.imshow('capture', capture_image)	# debug only

		zero = cv2.countNonZero(gray_image)
		if (zero/(self.capture_erea[2]*self.capture_erea[3]) < 0.05):
			# まっ黒に近い画面は処理しない
			return

		# [READY to FIGHT!]画面だけは検出する、どの状況でも。なぜなら正常に終わらなくても回ってくるので、回線落ちなどで (英文表記)
		if (self._is_ready_frame(capture_image)):
			# 前回の試合の結果をうまく検出できていなかった時用の修正用
			self._battle_end()
			# (戦闘力の差分)
			self._capture_ready_to_fight_frame(capture_image)
			self.frame_state = FrameState.FS_READY_TO_FIGHT
			return
		elif (self.frame_state in {FrameState.FS_UNKNOWN}):
			return

		if (self.frame_state in {FrameState.FS_RESULT, FrameState.FS_READY_TO_FIGHT}):
			# [まもなく開始]画面 (プレイヤー名)
			if (self._is_ready_ok_frame(capture_image)):
				self._default_battle_params()
				self.frame_state = FrameState.FS_READY_OK
				self._capture_ready_ok_frame(gray_image)
			return

		if (self.frame_state == FrameState.FS_READY_OK):
			if (2 == self.player_count):
				# battle with 4 (人数を 2 から 4 へ変更,もはやプレイヤー名とか正確に取れないので無視する)
				if (self._is_with_4_battle_frame(capture_image)):
					self.logger.debug(f'_is_with_4_battle_frame{self.ratio}')
					entry_name = self.entry_name
					self._default_battle_params(player_count=4)
					self.entry_name = entry_name
					self.frame_state = FrameState.FS_READY_OK
					return

				# [{foo} vs {bar}]画面 (1on1,キャラクター名)
				if (self._is_vs_frame(capture_image)):
					self.frame_state = FrameState.FS_WHAT_CHARACTER
					self._capture_character_name(gray_image)
					cv2.imwrite('last_foo_vs_bar.png', capture_image)

			elif (4 == self.player_count):
				# 大乱闘 or チーム乱闘 (smash or team,キャラクター名)
				is_smash_or_team_frame = self._watch_smash_or_team_frame(capture_image)
				if (is_smash_or_team_frame):
					self.frame_state = FrameState.FS_WHAT_CHARACTER
					self._capture_character_name(gray_image)
					cv2.imwrite('last_vs_character.png', capture_image)
			return

		if (self.frame_state == FrameState.FS_WHAT_CHARACTER):
			# [GO!]画面 (開始時間)
			if (self._is_go_frame(capture_image)):
				self.logger.debug(f'_is_go_frame{self.ratio}')
				self.frame_state = FrameState.FS_BATTLE
				#self._capture_go_frame(gray_image)		# TODO:開始時間取りたいけど数値の認識率が低すぎる、、、。
			return

		if (self.frame_state == FrameState.FS_BATTLE):
			# [GAME SET][TIME UP]画面
			if (self._is_game_end_frame(capture_image)):
				self.frame_state = FrameState.FS_BATTLE_END
				return

			# [0.0]% 監視 それ以外だと [1.0]% が代入される
			self._watch_player_zero_damage(capture_image)

			# [N - N]画面 (ストック数) (この画面は [1on1]でルール[stock] の時しか表示されない)
			if (self._is_stock_number_frame(capture_image)):
				self._capture_stock_number_frame(gray_image)

			# ストックの[-1 or +1]を監視,観測するとダメージに [-1.0]% が代入される (この画面はルール[time]の時しか表示されない)
			self._watch_stock_changed_frame(capture_image)

			return

		if (self.frame_state == FrameState.FS_BATTLE_END):
			# [結果]画面 (戦闘力,勝敗)
			is_result_frame = self._watch_reuslt_frame(capture_image, gray_image)
			if (self._is_battle_retry_frame(capture_image)):
				# 爆速ニキがどうしても捕捉できないので片方の戦歴から予測して調整する
				self.logger.debug(f'_is_battle_retry_frame{self.ratio}')
				is_result_frame = True

			if (is_result_frame):
				self._battle_end()
				self.frame_state = FrameState.FS_RESULT
				cv2.imwrite('last_foo_vs_bar_power.png', capture_image)

			return

	# 試合終了
	def _battle_end(self):
		if (self.frame_state != FrameState.FS_BATTLE_END):
			return
		if ( all([(-1 == order) for order in self.player_order]) ):
			# 順位が全く取れていない場合は何もしない(※これはイメージです の画像くらい違うものを検出してる可能性があるため)
			return False

		self.logger.debug(f'_battle_end {self.rule_name}')

		self._research_battle_result()

		if ('stock' == self.rule_name):
			self._stock_rate()
		elif ('time' == self.rule_name):
			self._time_rate()

		self._save_history()

	# 結果画面が正常に捕捉できなかった場合の救済措置
	def _research_battle_result(self):
		# 	順位を -1 から N に変更
		order_remain = [ (player+1) for player in range(self.player_count) if not (player+1) in self.player_order ]
		if (1 < len(order_remain)):
			# わからない順位が複数ある
			Utils.width_full_print(f"can't found player order's for {order_remain}", logger_func=self.logger.info)
			return
		if (0 == len(order_remain)):
			# 正常なのでなにもしない
			self.logger.debug(f'player order no anything changed. {self.player_order}')
			return

		min_order = min(self.player_order)
		if (all([ min_order == order for order in self.player_order ])):
			# 全プレイヤーが同じ順位 (正常なストックから推測)
			if ('stock' == self.rule_name):
				lowest = int(numpy.argmax(self.player_order))
				highest = int(numpy.argmin(self.player_order))
			elif ('time' == self.rule_name):
				pass
		else:
			# かけている順位が一つ (他の順位から推測)
			player = int(numpy.argmin(self.player_order))
			self.player_order[player] = order_remain[0]
			self.logger.debug(f'player order changed. {player} -> {order_remain[0]}th')

	# 1pプレイヤー目線のキャラ別勝敗数を蓄える
	def _store_battle_rate(self, chara_name, win=0, lose=0):
		win = win + 0
		lose = lose + 0
		key = tuple(chara_name)
		if ( not key in self.battle_rate ):
			self.battle_rate[key] = { 'win':0, 'lose':0 }

		self.battle_rate[key]['win'] += win
		self.battle_rate[key]['lose'] += lose
		
		plus = 1 if 0 < win else -1 if 0 < lose else 0
		if (0 == plus):
			return

		if ( abs(self.battle_streak_rate + plus) < abs(self.battle_streak_rate) ):
			# 計算後の値が現在の絶対値より小さくなる場合 連hoge が切れたことになるのでリセットする
			self.battle_streak_rate = 0
		self.battle_streak_rate += plus
		self.battle_history.append(plus)

	# 勝敗の保存 (stock戦)
	def _stock_rate(self):
		if ( all([-1 == player for player in self.result_stock['count']]) ):
			# ストック数が1
			self.result_stock = { 'count':[1]*self.player_count, 'max':[1]*self.player_count }

		lowest = int(numpy.argmax(self.player_order))
		highest = int(numpy.argmin(self.player_order))
		lowest_order = self.player_order[lowest]
		highest_order = self.player_order[highest]
		if ( self.result_stock['count'][lowest] == self.result_stock['count'][highest] ):
			# サドンデスなどでストック数が同じになる場合があるので、負けたほうをひとつ減算
			for player in numpy.where( self.player_order == lowest_order )[0]:
				# team サドンデス用に where match
				self.result_stock['count'][player] -= 1

		# 順位に基づいてストック数を再計算
		highest_count = lowest_count = 0
		for player in numpy.where( self.player_order == highest_order )[0]:
			highest_count += self.result_stock['count'][player]
		for player in numpy.where( self.player_order == lowest_order )[0]:
			lowest_count += self.result_stock['count'][player]
		if ( highest_count < lowest_count ):
			# ストックの合計において,勝っている方 < 負けている方 の場合,検出エラーとして扱い,順位に基づいてストック数の最大を 1 として算出する
			for player in numpy.where( self.player_order == lowest_order )[0]:
				self.result_stock['count'][player] = 0
				self.result_stock['max'][player] = 1
			for player in numpy.where( self.player_order == highest_order )[0]:
				self.result_stock['count'][player] = 1
				self.result_stock['max'][player] = 1
			self.logger.debug(f'stock recalc {self.result_stock}')

		self._store_battle_rate( self.chara_came,
			self.player_order[0] == highest_order,
			self.player_order[0] == lowest_order)

	# 勝敗の保存 (time戦)
	def _time_rate(self):
		# 順位に基づいてポイントを再計算
		lowest = int(numpy.argmax(self.player_order))
		highest = int(numpy.argmin(self.player_order))
		lowest_order = self.player_order[lowest]
		highest_order = self.player_order[highest]
		highest_point = lowest_point = 0
		for player in self.player_order:
			point = self.result_stock['max'][player] - self.result_stock['count'][player]
			if (lowest == player):
				lowest_point = point
			elif (highest == player):
				highest_point = point

		if ( highest_point < lowest_point ):
			# 撃墜数 - 落下数 において,勝っている方 < 負けている方 の場合,検出エラーとして扱い,順位に基づいてポイントの最大を player_count として算出する
			# time戦に関しては result_stock の内容が count=落下数 max=撃墜数 となっている
			max_point = self.player_count - len( list(numpy.where(self.player_order == highest)) ) + 1
			max_point = max_point if max_point == self.player_count else max_point - 1
			for player, order in enumerate(self.player_order):
				self.result_stock['count'][player] = order - 1
				self.result_stock['max'][player] = (max_point - order)
			"""
			o: count, max
			1: 1-1=0, 4-1=3
			2: 2-1=1, 4-2=2
			3: 3-1=2, 4-3=1
			4: 4-1=3, 4-4=0

			1: 1-1=0, 2-1=1
			1: 1-1=0, 2-1=1
			2: 2-1=1, 2-2=0
			2: 2-1=1, 2-2=0
			"""

		self._store_battle_rate( self.chara_came,
			self.player_order[-1] > self.player_order[0],
			self.player_order[-1] < self.player_order[0] )

	# 戦歴の読込
	def _load_history(self):
		po = Path('./log/')
		history_pattern = self.config['log']['name'].format(now=r'\d+', chara=r'\S+')
		dirlist = [file for file in po.glob('*.json') if re.search(history_pattern, str(file))]
		for history_file in dirlist:
			Utils.width_full_print(f'\r{history_file}')
			with open(history_file) as f:
				try:
					history_file = json.load(f)
					player_info = history_file['player']
				except json.decoder.JSONDecodeError:
					# 空 or json が正しくないものは無視する
					continue

			# キャラ別戦歴
			self._store_battle_rate( [player['name'] for player in player_info[0:] ],
				player_info[-1]['order'] > player_info[0]['order'],
				player_info[-1]['order'] < player_info[0]['order'] )

		self.battle_streak_rate = 0

	# 試合結果を hoge.foo に保存
	def _save_history(self):
		self.logger.info('saved log')
		# jsonに整形
		dt_now = datetime.datetime.now()
		now = dt_now.strftime('%Y%m%d%H%M%S')
		history_file_name = self.config['log']['name'].format(now=now, chara=self.chara_came)
		player = list(range(self.player_count))
		for key, chara in enumerate(self.chara_came):
			player[key] = {
				'name': chara,
				'order': self.player_order[key],
				'power': self.power[key],
				'stock': self.result_stock['count'][key],
				'group': self.group_color[key]
			}
		json_object = {
			'now': now,
			'rule': {
				'name': self.rule_name,
				'group': self.group_name,
				'stock': self.result_stock['max'],
				'time': 0
			},
			'player': player
		}

		# 保存先が [URL なら POST 叩く][ローカルなら open->write]
		path = self.config['log']['path']
		if (0 < len(urllib.parse.urlparse(path).scheme)):
			headers = self.config['log']['headers']
			request = urllib.request.Request(path, urllib.parse.urlencode(json_object).encode('ascii'), headers)

			try:
				with urllib.request.urlopen(request) as response:
					body = response.read()
				self.logger.info('url=[%s],responce=%s', path, body)
			except urllib.error.HTTPError as e:
				self.logger.error('HTTPError:%s', e)
			except urllib.error.URLError as e:
				self.logger.error('URLError:%s', e)
		else:
			with open(path.format(file_name=history_file_name), mode='w') as f:
				f.write( json.dumps(json_object, sort_keys=True) )

		print('\rsaved=', history_file_name, player)

	# バトル中に表示する情報
	def _battle_information(self, capture_image):
		# N戦の勝率
		battle_streak_ratio_max = self.config['option']['battle_streak_ratio_max']
		battle_streak_ratio = [{'win':0.0, 'lose':0.0, 'length':0} for _ in battle_streak_ratio_max]
		total_count = len(self.battle_history)
		if (0 < total_count):
			for key, battle_max in enumerate(battle_streak_ratio_max):
				count = min(total_count, len(self.battle_history[:battle_max]) )
				win_count = lose_count = 0
				for result in self.battle_history[-count:]:
					win_count += 1 if 0 < result else 0
					lose_count += 1 if result < 0 else 0
				battle_streak_ratio[key]['win'] = round(win_count / count * 100.0, 1)
				battle_streak_ratio[key]['lose'] = round(lose_count / count * 100.0, 1)
				battle_streak_ratio[key]['length'] = count

		# 表示する情報
		chara_came = list(self.chara_came)
		count = 65
		for key, chara in enumerate(self.chara_came):
			if ('' == chara):
				chara_came[key] = chr(count)
				count += 1
		player_info = dict(zip( tuple(chara_came), list(zip( self.result_stock['count'], self.group_color )) ))
		battle_information_content = {}
		for name in ['gui_text', 'cui_text']:
			battle_information_content[name] = self.config['option']['battle_information'][name].format(
					frame=FrameState(self.frame_state + 1).name, ratio=self.ratio,
					entry_name=self.entry_name, group_name=self.group_name,
					group_color=self.group_color,
					chara_came=self.chara_came, stock=self.result_stock['count'],
					player_info=f'{player_info}',
					battle_rate=self.battle_rate[tuple(self.chara_came)],
					streak=abs(self.battle_streak_rate),
					streak_name=('lose' if self.battle_streak_rate < 0 else 'win'),
					streak_ratio=battle_streak_ratio
				)

		if (self.config['option']['battle_informationGUI']):
			battle_information_image = numpy.zeros(capture_image.shape, dtype=numpy.uint8)
			battle_information_image[:] = tuple(self.config['option']['battle_information']['back'])
			Utils.draw_text_line(battle_information_image, battle_information_content['gui_text'],
				tuple(self.config['option']['battle_information']['pos']),
				self.config['option']['battle_information']['tickness'],
				self.config['option']['battle_information']['color'], 1.0)
			cv2.imshow(self.config['option']['battle_information']['caption'], battle_information_image)

		if (self.config['option']['battle_informationCUI']):
			Utils.width_full_print(battle_information_content['cui_text'])

	# 1 frame 中の処理
	def _capture_frame(self):
		# 1 frame capture
		capture_image = self._capture_window()
		self._capture_any_frame(capture_image)
		self._battle_information(capture_image)
		cv2.waitKey(1)

		# GUIを表示しているときだけ WM_CLOSE で終了する
		if ( self.config['option']['battle_informationGUI'] ):
			if ( cv2.getWindowProperty( self.config['option']['battle_information']['caption'], cv2.WND_PROP_VISIBLE ) < 1 ):
				return False
		return True

	# 初期化処理
	def _initialize(self):
		Utils.width_full_print('initialize...', logger_func=self.logger.debug)
		self._load_config()
		self._load_resources()
		self._load_first_only()
		self._load_history()
		self._init_capture_window()
		self._default_battle_params()

	# 終了処理
	def _finalize(self):
		self.logger.info('finalize')
		if (self.config['option']['every_time_find_capture_area']):
			self.config['capture']['width'] = 0
			self.config['capture']['height'] = 0

		with open(self.config_path, mode='w', encoding='utf-8') as file:
			file.write( json.dumps([self.config], indent=4, ensure_ascii=False) )

		Utils.width_full_print('exit.', logger_func=self.logger.info)

	# main loop
	def _main_loop(self):
		Utils.width_full_print('ready.', logger_func=self.logger.info)
		while( self._capture_frame() ):
			pass

	# exeception dump
	def _dump(self, err):
		Utils.width_full_print('\nhad exception. please send *.log to Humi@bass_clef_')
		_value, _type, _traceback = sys.exc_info()
		self.logger.error(traceback.format_exception(_value, _type, _traceback))

	""" 外部向け関数(runのみｗｗｗ) """
	# エンジン始動！
	def run(self):
		try:
			self._initialize()
			self._main_loop()
		except KeyboardInterrupt:
			pass
		except Exception as e:
			self._dump(e)
		finally:
			self._finalize()

if __name__ == '__main__':
	# exe と一緒に含める リソースフォルダのパス へのアクセスを可能にする関数
	def _resource(filename):
		if (not type(filename) is str):
			return filename

		if hasattr(sys, "_MEIPASS"):
			return os.path.join(sys._MEIPASS, filename)
		return os.path.join(filename)

	# strさん にお猿さんの呪いの実で +単項演算子 で パッケージ?コンパイル済みリソース? へのアクセスを処理してもらうようにする
	curse(str, '__pos__', _resource)

	# main engine
	engine = SmaBroEngine()
	engine.run()
