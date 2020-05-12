#coding: utf-8

"""
Name   : smabrog (smash bro. log)
Auther : Humi@bass_clef_

"""


import os
import warnings
# matplotlib から exe 起動時に警告が出るのを抑制する
warnings.filterwarnings('ignore', '(?s).*MATPLOTLIBDATA.*', category=UserWarning)
os.environ['KIVY_NO_CONSOLELOG'] = '1'

from collections import Counter, defaultdict, OrderedDict
from ctypes import *
from ctypes.wintypes import *
import cv2
import datetime
import concurrent.futures
import difflib
from enum import IntEnum
from forbiddenfruit import curse
import japanize_matplotlib
import json
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.lang import Builder
from kivy.logger import Logger as kivy_logger
from kivy.resources import resource_add_path
from kivy.uix.widget import Widget
import logging
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import multiprocessing
import numpy
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import pyocr
import pyocr.builders
import random
import re
import signal
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

	# 透過色付きでない image から黒のみを透過色とした mask を作成
	def make_trans_mask_noalpha_channel_(image):
		channels = cv2.split(image)
		zero_channel = numpy.zeros_like(channels[0])
		mask = numpy.array(channels[3])
		mask[channels[3] == 0] = 1
		mask[channels[3] == 100] = 0
		return cv2.merge([zero_channel, zero_channel, zero_channel, mask])

	# dest の上に透過色に応じて original_source を pos の位置に貼り付ける
	def paste_image_pos_(dest, original_source, pos):
		s_h, s_w = original_source.shape[0:2]
		d_h, d_w = dest.shape[0:2]
		x = pos[0]
		y = pos[1]

		if (original_source.shape[2] == 4):
			source = original_source
			mask = original_source[:,:,3]
		else:
			source = cv2.cvtColor(original_source, cv2.COLOR_RGB2RGBA)
			mask = cv2.cvtColor( Utils.make_trans_mask_noalpha_channel(source), cv2.COLOR_RGBA2GRAY)

		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		mask = mask / 255.0

		dest[ y:y+s_h, x:x+s_w ] = dest[ y:y+s_h, x:x+s_w ] * 1 - mask
		masked_image = cv2.bitwise_not(cv2.cvtColor( Utils.pil2cv(source[:,:,:3] * mask), cv2.COLOR_BGR2RGB ))

		dest_mask = cv2.cvtColor( Utils.pil2cv(dest[ y:y+s_h, x:x+s_w ]), cv2.COLOR_BGR2RGB )
		dest[ y:y+s_h, x:x+s_w ] = cv2.bitwise_and(masked_image, dest_mask)
		return dest

	# 透過色付きでない image から color_code を透過色とした mask を作成
	def make_trans_mask_noalpha_channel(image, color_code=(0,0,0,255), distance=0):
		lower_color = numpy.array(color_code)
		upper_color = numpy.array(color_code)
		upper_color[:-1] += distance
		mask = cv2.inRange(image, lower_color, upper_color)
		mask = cv2.bitwise_not(mask)

		b = image[:,:,0]
		g = image[:,:,1]
		r = image[:,:,2]
		return cv2.merge((b, g, r, mask))

	# dest の上に透過色に応じて original_source を pos の位置に貼り付ける
	def paste_image_pos(original_dest, original_source, pos):
		s_h, s_w = original_source.shape[0:2]
		d_h, d_w = original_dest.shape[0:2]
		x = pos[0]
		y = pos[1]

		if (original_source.shape[2] == 4):
			source = original_source
		else:
			source = cv2.cvtColor(original_source, cv2.COLOR_RGB2RGBA)

		mask = Utils.make_trans_mask_noalpha_channel(source)

		if (original_dest.shape[2] == 4):
			dest = original_dest
		else:
			dest = cv2.cvtColor(original_dest, cv2.COLOR_RGB2RGBA)

		dest[ y:y+s_h, x:x+s_w ] = mask
		return cv2.cvtColor(dest, cv2.COLOR_RGBA2RGB)


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
			return image, (word_x, word_y, word_width, word_height)
		return result_image, (word_x, word_y, word_width, word_height)

	# image と color_image をマッチングテンプレートで比較して一番確率が高い 確率,座標 を返す
	# @param Image mask_image		指定した場合は image を事前にマスク処理する
	#								is_trans=True の場合は透過色用のマスクとして使用する
	def match_masked_color_image(image, color_image, mask_image=None, is_trans=False, method=cv2.TM_CCOEFF_NORMED, raw_result=False):
		convert_image = image.copy()
		if (is_trans):
			if (mask_image is None):
				transparent_mask = Utils.make_trans_mask_noalpha_channel(color_image)
				"""
				channels = cv2.split(color_image)
				zero_channel = numpy.zeros_like(channels[0])
				mask = numpy.array(channels[3])
				mask[channels[3] == 0] = 1
				mask[channels[3] == 100] = 0
				transparent_mask = cv2.merge([zero_channel, zero_channel, zero_channel, mask])
				"""
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

		if (raw_result):
			return result

		_, ratio, _, pos = cv2.minMaxLoc(result)
		return ratio, pos

	# 無効な文字列の削除
	def replace_invalid_char(word):
		return re.sub(r'[\W|,|.]', '', word)

	# consoleの幅いっぱいに空白をつめて表示 (デフォルトCRすると前の文字が残ってて読みづらいため)
	def width_full_print(message, carriage_return=True, logger_func=None):
		columns, _ = os.get_terminal_size()
		m_len = 0
		for c in message:
			m_len += 2 if unicodedata.east_asian_width(c) in 'FWA' else 1
		print('\r' if carriage_return else '', message, ' ' * (columns - m_len - 1), end='')

		if (not logger_func is None):
			logger_func(message)

class KivyWidget(Widget):
	def __init__(self, **kwargs):
		super(KivyWidget, self).__init__(**kwargs)

class KivyApp(App):
	def __init__(self, **kwargs):
		super(KivyApp, self).__init__(**kwargs)
		self.var = None
		self.root = None
		self.file_path = None
		self.default_on_window = None

	def _enum_dispatch(self, root):
		root.dispatch('on_update')
		if ( hasattr(root, 'children') ):
			for child in root.children:
				self._enum_dispatch(child)

	def init(self, file_path, var):
		self.file_path = file_path
		self.var = var

	def update(self, deltatime):
		with self.root.canvas.before:
			if (self.var.animation_texture is None):
				self.var.animation_texture = Texture.create(size=self.var.black_image.shape[-2::-1])
			else:
				image = self.var.black_image 
				if (not self.var.gui_info['image'] is None):
					image = self.var.gui_info['image']

				image = cv2.flip(image, 0)
				self.var.animation_texture.blit_buffer( image.tostring() )
				Rectangle(texture=self.var.animation_texture, pos=(0, 0), size=image.shape[-2::-1] )

		with self.root.canvas:
			self._enum_dispatch(self.root)

	def build(self):
		with open(self.file_path, encoding='utf-8') as file:
			kivy_string = file.read()

		root = Builder.load_string( kivy_string )
		if (root is None):
			Utils.width_full_print('kivy root is None', self.var.logger.info)
			return

		self.root = KivyWidget()
		self.root.add_widget(root)
		Clock.schedule_interval(self.update, 1/self.var.config['resource']['kivy']['fps'])
		return self.root

	def close(self, **kwargs):
		#self.var.config['resource']['kivy']['pos'] = [0, 20]
		Utils.width_full_print('stoped')
		return super(KivyApp, self).close(**kwargs)


class SmaBroEngine:
	FrameState = IntEnum(
		'FrameState',
		['FS_UNKNOWN',
		'FS_READY_TO_FIGHT',
		'FS_READY_OK',
		'FS_WHAT_CHARACTER',
		'FS_BATTLE',
		'FS_BATTLE_END',
		'FS_RESULT',
		'FS_SAVED',
		'FS_LOADING',
		'FS_END_STATE'
		])

	AnimationState = IntEnum(
		'AnimationState',
		['AS_NONE',
		'AS_CHARACTER_RATE_GRAPH_BEGIN',
		'AS_CHARACTER_RATE_GRAPH',
		'AS_STREAK_RATE_BEGIN',
		'AS_STREAK_RATE',
		'AS_END_STATE'
		])

	def __init__(self):
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(logging.DEBUG)
		now = datetime.datetime.now().strftime("%Y_%m_%d_%H")
		log_handler = logging.FileHandler(filename=f'log/{now}.log')
		log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)8s %(message)s', datefmt='%M:%S'))
		self.logger.addHandler(log_handler)

		# kivy のログを同じファイルに書き込む
		kivy_logger.addHandler(log_handler)

	# コンフィグの読み込み
	def _load_config(self):
		self.config_path = 'config.json'

		with open(self.config_path, encoding='utf-8') as file:
			config_string = file.read()
			self.config = json.loads(config_string, object_pairs_hook=OrderedDict)[0]

	# リソースの読み込み/作成
	def _load_resources(self):
		# 変数リソース
		self.chara = self.config['resource']['character']
		self.team_color_list = ['red', 'blue', 'yellow', 'green']
		self.resource_size = { 'width':640, 'height':360 }
		self.capture_image = None
		self.sync = 0
		self.back_character = ''
		self.back_power = 0
		self.battle_rate = defaultdict()
		self.battle_streak_rate = 0	# 正数で連勝数,負数で連敗数
		self.battle_history = {}
		self.power_history = {}
		self.power_limit = { 'min':-1, 'max':-1 }
		self.time_zero = datetime.datetime(1900, 1, 1, 0, 0)
		self.gui_info = None
		self.battle_streak_ratio = [ {'win':0.0, 'lose':0.0, 'length':0} for _ in self.config['option']['battle_streak_ratio_max'] ]
		self.black_image = Utils.pil2cv( numpy.zeros((self.resource_size['height'], self.resource_size['width'], 3)) )

		self.animation_state = self.AnimationState.AS_NONE
		self.animation_count = 0
		self.animation_calc = 0
		self.animation_image = None
		self.animation_texture = None
		self._make_interface_canvas()

		# フォント
		self.font = ImageFont.truetype(self.config['resource']['font']['normal'][0], self.config['resource']['font']['normal'][1])
		self.small_font = ImageFont.truetype(self.config['resource']['font']['small'][0], self.config['resource']['font']['small'][1])

		# これとは別で管理したい並列処理は with で別途記述 (スレッドを取り合ってしまい処理が進まずデッドロックになるため)
		self.executor = concurrent.futures.ThreadPoolExecutor(self.config['option']['max_workers'])
		self.process_executor = concurrent.futures.ProcessPoolExecutor(self.config['option']['max_workers'])

		""" 外部リソース """
		# tesseract-OCR
		tools = pyocr.get_available_tools()
		if len(tools) == 0:
			self.logger.error('No tesseract-OCR tool found')
			sys.exit(1)
		self.ocr_tool = tools[0]
		# kivy
		Config.set('graphics', 'left', self.config['resource']['kivy']['pos'][0])
		Config.set('graphics', 'top', self.config['resource']['kivy']['pos'][1])
		for option in self.config['resource']['kivy']['option']:
			Config.set(option[0], option[1], option[2])
		LabelBase.register(DEFAULT_FONT, self.config['resource']['font']['normal'][0])
		self.kivy_app = KivyApp()
		self.kivy_app.init(self.config['resource']['kivy']['path'], self)
		self.kivy_app.title = self.config['resource']['kivy']['title']

		# 解析に必要な画像リソース
		self.loading_mask = cv2.imread(+'resource/loading_mask.png')
		self.loading_color = cv2.imread(+'resource/loading_color.png')

		self.ready_to_fight_trans_mask = cv2.imread(+'resource/ready_to_fight_mask.png', cv2.IMREAD_UNCHANGED)
		self.ready_to_fight_trans_color = list(range(2))
		for i in range(2):
			self.ready_to_fight_trans_color[i] = cv2.imread(+f'resource/ready_to_fight_color_{i}.png', cv2.IMREAD_UNCHANGED)
		self.ready_to_fight_mask = cv2.cvtColor(self.ready_to_fight_trans_mask, cv2.COLOR_RGBA2RGB)
		self.ready_to_fight_color = list(range(2))
		for key, color_image in enumerate(self.ready_to_fight_trans_color):
			self.ready_to_fight_color[key] = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)

		self.ready_to_fight_name_power_mask = cv2.imread(+'resource/ready_to_fight_name_power_mask.png')

		self.ready_ok_mask = cv2.imread(+'resource/ready_ok_mask.png', cv2.IMREAD_UNCHANGED)
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
		self.rule_time_stock_mask = cv2.imread(+'resource/rule_time_stock_mask.png')
		self.rule_time_stock_color = cv2.imread(+'resource/rule_time_stock_color.png')

		self.battle_time_zero_mask = cv2.imread(+'resource/battle_time_zero_mask.png', cv2.IMREAD_UNCHANGED)
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

		#その他の画像リソース
		self.smabro_icon = cv2.imread(+'resource/smabro_icon.png', cv2.IMREAD_UNCHANGED)

	# キャプチャに関する初期化
	def _init_capture_area(self):
		if (not self.config['option']['find_capture_area']):
			if ( 0 == self.config['capture']['width'] and 0 == self.config['capture']['height'] ):
				self.config['capture']['x'] = 0
				self.config['capture']['y'] = 0
				self.config['capture']['width'] = self.resource_size['width']
				self.config['capture']['height'] = self.resource_size['height']
			return
		if (self.config['option']['every_time_find_capture_area']):
			self.config['capture']['width'] = self.config['capture']['height'] = 0

		resolution_list = [ [self.config['capture']['width'],self.config['capture']['height']] ]
		if ( 0 == self.config['capture']['width'] or 0 == self.config['capture']['height'] ):
			base_resolution = [16, 9]
			magnification_list = [40, 53, 80, 100, 120] + list(range(41, 53)) + list(range(54, 80)) + list(range(81, 100)) + list(range(101, 120))
			resolution_list = list(range(len(magnification_list)))
			for key, m in enumerate(magnification_list):
				resolution_list[key] = [ int(base_resolution[0] * m), int(base_resolution[1] * m) ]

			self.config['option']['find_capture_area'] = True
			self.config['option']['exit_not_found_capture'] = False
			Utils.width_full_print(f'\rfinding capture area...')


		# このためだけに透過色付きで読み込んでいたので,今後のために RGBA -> RGB 変換をする
		self.sync = -1
		with concurrent.futures.ThreadPoolExecutor(self.config['option']['max_workers']) as executor:
			result = list(executor.map(
				self._find_resolution, range(len(resolution_list)), resolution_list
				))

		if (-1 != self.sync):
			ratio, x, y = result[self.sync]
			width, height = resolution_list[self.sync]

			self.logger.info(f'found capture area {x}x{y}')
			self.config['capture']['x'] = x
			self.config['capture']['y'] = y
			self.config['capture']['width'] = width
			self.config['capture']['height'] = height

			if (self.config['option']['every_time_find_capture_area']):
				Utils.width_full_print( f"resolution is {width}x{height}", logger_func=self.logger.info )
			if (self.config['option']['found_capture_area_fixed']):
				self.config['option']['find_capture_area'] = False

			self._capture_window()
			cv2.imwrite('found_capture_area.png', self.capture_image)

		if ( -1 == self.sync or 0 == self.config['capture']['width'] or 0 == self.config['capture']['height'] ):
			Utils.width_full_print('not found capture area.')
			if (self.config['option']['exit_not_found_capture']):
				Utils.width_full_print('exit_not_found_capture true.')
				raise KeyboardInterrupt()

			if (self.config['option']['every_time_find_capture_area']):
				raise KeyboardInterrupt()
			else:
				self.config['capture']['width'] = self.resource_size['width']
				self.config['capture']['height'] = self.resource_size['height']

	# キャプチャ対象の解像度の検出
	def _find_resolution(self, index, resolution):
		if (-1 != self.sync):
			time.sleep(0.0001)
			return 0.0, None, 0, 0, 0, 0

		ratio, convert_image, x, y = self._find_capture_area(resolution[0], resolution[1])
		if (ratio < 0.99):
			return 0.0, None, 0, 0, 0, 0

		self.sync = index
		self.capture_image = convert_image
		return ratio, x, y

	# キャプチャすべき場所の検出
	def _find_capture_area(self, width, height):
		# [READY to FIGHT]によって自動判別
		result_ratio = 0.0
		desktop_width = ctypes.windll.user32.GetSystemMetrics(0)
		desktop_height = ctypes.windll.user32.GetSystemMetrics(1)
		capture_erea = (0, 0, desktop_width, desktop_height)
		capture_image = Utils.pil2cv(ImageGrab.grab(bbox=capture_erea))
		convert_image = capture_image

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
		x = y = 0
		for key, color_image in enumerate(self.ready_to_fight_trans_color):
			# デバッグ途中に特殊な環境での意味のわからない、これは一致するのぉ？？？、という条件が一致して
			# 本来取得すべき座標が見逃されていたものがあったため、
			# 97%以上の[READY to FIGHT]の座標を25件まで列挙して、
			# それから _is_ready_frame かけて合格したやつをさらに下記で微調整する
			result = Utils.match_masked_color_image(convert_image, color_image,
				self.ready_to_fight_trans_mask, is_trans=True, raw_result=True)
			pos_list = numpy.where(0.96 <= result)
			pos_list = list(zip( *pos_list[::-1] ))
			ratio_pos_list = list(zip( [ result[pt[1]][pt[0]] for pt in pos_list ], pos_list ))
			ratio_pos_list = numpy.sort(ratio_pos_list, axis=0)[0:25]

			if (len(ratio_pos_list) < 1):
				break

			p_ratio = {}
			for result_ratio, pos in ratio_pos_list:
				capture_area_image = convert_image[ pos[1]:int(pos[1]+self.resource_size['height']), pos[0]:int(pos[0]+self.resource_size['width']) ]

				# より正確な位置を特定
				base_x = int(pos[0] * width_magnification)
				base_y = int(pos[1] * height_magnification)
				for add_y in range(-1, 1):
					for add_x in range(-1, 1):
						x = base_x + add_x
						y = base_y + add_y
						if (x < 0 or y < 0 or desktop_height < y+height or desktop_width < x+width):
							continue

						capture_area_image = cv2.resize(capture_image[ y:y+height, x:x+width ], dsize=(self.resource_size['width'], self.resource_size['height']))
						is_ready_frame, ratio = self._is_ready_frame(capture_area_image)
						if ( is_ready_frame ):
							p_ratio[tuple([x, y])] = ratio
							if (0.997 <= ratio):
								# 確実に検出できてるため以後全部スキップ
								break
					else:
						continue
					break

			if (len(p_ratio) < 1):
				break
			self.logger.debug(f'near detected resolution {width}x{height} : {p_ratio}')

			p, result_ratio = max( p_ratio.items(), key=lambda x:x[1] )
			x, y = p[0:2]
			break

		return result_ratio, convert_image, x, y

	# タイトルからウィンドウを捕捉 (これより, _init_capture_area ぶん回したほうが早いｗｗｗ おまけ程度)
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

		self.capture_image = Utils.pil2cv(ImageGrab.grab(bbox=self.capture_erea))
		self.capture_image = cv2.resize(self.capture_image, dsize=(self.resource_size['width'], self.resource_size['height']))

	# 結果画面が正常に捕捉できなかった場合の救済措置
	def _research_battle_result(self):
		# 異常な戦闘力
		# 直近10試合の平均の戦闘力から 100万以上離れていた場合,戦闘力を誤検出として記録 (数字が増えてしまう誤検出)
		# 相手の戦闘力も自身と近いはずなので,それも誤検出として扱う
		if (9 < len( self.power_history[self.chara_name[0]] )):
			power_history = self.power_history[self.chara_name[0]][::-1]
			power_history = power_history[0:10]
			power = numpy.array(power_history)[:,1].tolist()
			for i in range(self.player_count):
				if ( 1e6 < abs(numpy.mean(power) - self.power[i]) ):
					self.logger.debug(f'player {i} power is impossible value {self.power[i]} -> 0')
					self.power[i] = 0


		# 順位を -1 から N に変更
		order_remain = [ (player+1) for player in range(self.player_count) if not (player+1) in self.player_order ]
		if (1 < len(order_remain)):
			# わからない順位が複数ある
			Utils.width_full_print(f"can't found player order's for {order_remain}", logger_func=self.logger.info)
			return
		if (0 == len(order_remain)):
			# 正常なのでなにもしない
			self.logger.debug(f'player order no anything changed. {self.player_order}')
			return

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

		if (not chara_name[0] in self.battle_history or not chara_name[0] in self.power_history):
			self.battle_history[chara_name[0]] = []
			self.power_history[chara_name[0]] = []
		
		plus = 1 if 0 < win else -1 if 0 < lose else 0
		if (0 == plus):
			return

		if ( abs(self.battle_streak_rate + plus) < abs(self.battle_streak_rate) ):
			# 計算後の値が現在の絶対値より小さくなる場合 連hoge が切れたことになるのでリセットする
			self.battle_streak_rate = 0
		self.battle_streak_rate += plus
		self.battle_history[chara_name[0]].append(plus)

		if (10000 < self.power[0]):
			# ごめん 1万以下は人権ないかも… (誤検出を省く)
			self.power_history[chara_name[0]].append( [len(self.battle_history[chara_name[0]]), self.power[0]] )

		battle_streak_ratio_max = self.config['option']['battle_streak_ratio_max']
		self.battle_streak_ratio = [{'win':0.0, 'lose':0.0, 'length':0} for _ in battle_streak_ratio_max]
		if ( 2 < len(self.battle_history[chara_name[0]]) ):
			# N戦の勝率
			total_count = len(self.battle_history[chara_name[0]])
			if (0 < total_count):
				battle_history = self.battle_history[chara_name[0]][::-1]
				for key, battle_max in enumerate(battle_streak_ratio_max):
					count = min(total_count, len(battle_history[0:battle_max]) )
					win_count = lose_count = 0
					for result in battle_history[0:count]:
						win_count += 1 if 0 < result else 0
						lose_count += 1 if result < 0 else 0

					self.battle_streak_ratio[key]['win'] = round(win_count / count * 100.0, 1)
					self.battle_streak_ratio[key]['lose'] = round(lose_count / count * 100.0, 1)
					self.battle_streak_ratio[key]['length'] = count

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

		self._store_battle_rate( self.chara_name,
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

		self._store_battle_rate( self.chara_name,
			self.player_order[-1] > self.player_order[0],
			self.player_order[-1] < self.player_order[0] )

	# 戦歴の読込
	def _load_history(self, count, history_path):
		with open(history_path) as f:
			try:
				history_file = json.load(f)
				player_info = history_file['player']
			except json.decoder.JSONDecodeError:
				player_info = None

		# _store_battle_rate がスタック形式で保存してるので,同期
		while count != self.sync:
			time.sleep(0.0001)

		if (not player_info is None):
			Utils.width_full_print(f'\rloading... {history_path}')
			self.power = [ player_info[0]['power'], player_info[1]['power'] ]
			self.chara_name = [player['name'] for player in player_info[0:] ]
			# キャラ別戦歴
			self._store_battle_rate( self.chara_name,
				player_info[-1]['order'] > player_info[0]['order'],
				player_info[-1]['order'] < player_info[0]['order'] )

		self.sync = count+1
	def _load_historys(self):
		po = Path('./log/')
		history_pattern = self.config['log']['name'].format(now=r'\d+', chara=r'\S+')
		dirlist = [file for file in po.glob('*.json') if re.search(history_pattern, str(file))]

		self.sync = 0
		with concurrent.futures.ThreadPoolExecutor() as executor:
			_ = list(executor.map(
				self._load_history,
				range(len(dirlist)),
				dirlist
				))

		self.battle_streak_rate = 0

	# 試合結果を hoge.foo に保存
	def _save_history(self):
		self.logger.info('saved log')
		# jsonに整形
		dt_now = datetime.datetime.now()
		now = dt_now.strftime('%Y%m%d%H%M%S')
		history_file_name = self.config['log']['name'].format(now=now, chara=self.chara_name)
		player = list(range(self.player_count))
		for key, chara in enumerate(self.chara_name):
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
				'time': [f'{t.minute:02}:{t.second:02}.{t.microsecond:02}' for t in self.battle_time]
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

	# スマブラの戦歴に関わる変数をデフォルトする
	def _default_battle_params(self, player_count=2):
		self.ratio = 0.0
		self.player_order_ratio = [0.99] * player_count
		self.time_stock_ratio	= []

		self.frame_state		= self.FrameState.FS_UNKNOWN
		self.entry_name			= [''] * player_count
		self.rule_name			= 'stock'	# stock or time
		self.group_name			= 'smash'	# smash or team
		self.group_color		= [''] * player_count
		self.chara_name			= [''] * player_count
		self.battle_time		= [datetime.datetime(1900, 1, 1, 0, 0) for _ in range(2)]
		self.player_damage		= [-1.0] * player_count
		self.defeated_player	= [-1] * player_count
		self.result_stock		= {'max':[-1] * player_count, 'count':[-1] * player_count}
		"""
			if rule_name == 'stock': max is max stock, count is stock
			if rule_name == 'time': max is add count, count is sub count
		"""
		self.power_diff		= 0
		self.power			= [0] * player_count
		self.player_order	= [-1] * player_count
		self.player_count	= player_count

		# ('', '', '', '') の作成だけしておく (読み取り先が無いと怒られるので)
		self._store_battle_rate(self.chara_name)


	""" capture系 hoge frame を検出した時にする処理 検出時間(昇順) """
	# [READY to FIGHT]画面
	def _capture_ready_to_fight_frame(self):
		if (self.frame_state == self.FrameState.FS_READY_TO_FIGHT):
			return
		if (self.power[0] < 0):
			# 初期値 or エラーは計算しない
			return
		self.logger.debug(f'_capture_ready_to_fight{self.ratio}')

		width = int(self.capture_image.shape[1])
		height = int(self.capture_image.shape[0])
		convert_image = cv2.bitwise_and(self.capture_image, self.ready_to_fight_name_power_mask)
		capture_gray_image = cv2.cvtColor(convert_image, cv2.COLOR_RGB2GRAY)

		# 自分の戦闘力の増減値
		power_area_image = convert_image[int(height/2):height, int(width/2):width]
		gray_power_area_image = capture_gray_image[int(height/2):height, int(width/2):width]
		power_area_image, p = Utils.trimming_any_rect(power_area_image, gray_power_area_image)

		gray_power_area_image = cv2.cvtColor(power_area_image, cv2.COLOR_RGB2GRAY)
		_, gray_power_area_image = cv2.threshold(gray_power_area_image, 150, 255, cv2.THRESH_BINARY_INV)
		power_area_image, p = Utils.trimming_any_rect(power_area_image, gray_power_area_image, 1, 0)

		#cv2.imshow('capture_ready_to_fight_power', power_area_image)	# debug only
		power = self.ocr_tool.image_to_string(
			Utils.cv2pil(power_area_image), lang='eng',
			builder=pyocr.builders.DigitBuilder(tesseract_layout=6) )
		if ('' == power):
			return

		power = int(Utils.replace_invalid_char(power))
		self.power[0] = power
		self.power_diff = self.power[0] - self.back_power
		self.back_power = self.power[0]

		diff_message = ' diff: {sign}{power_diff}'.format(sign='' if self.power_diff < 0 else '+', power_diff=self.power_diff)
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
			image, _ = Utils.trimming_any_rect(image, gray_name_area_image[key])
			_, convert_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
			convert_image = cv2.bitwise_not(convert_image)
			image, _ = Utils.trimming_any_rect(image, convert_image, margin=2)

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

		chara_name = [''] * self.player_count
		for key, image in enumerate(name_area_image):
			image, _ = Utils.trimming_any_rect(image, gray_name_area_image[key], 5)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
			#cv2.imshow('name_area_image'+ str(key), name_area_image[key])		# debug only
			chara_name[key] = '' + self.ocr_tool.image_to_string(
				Utils.cv2pil(image), lang='eng',
				builder=pyocr.builders.TextBuilder() )
			chara_name[key] = Utils.replace_invalid_char(chara_name[key])
		if ( all(['' == name for name in chara_name]) ):
			return

		# 画像をごちゃごちゃしてもまだ誤字/脱字/不一致があるので一致率が 50% 以上で一番高いものにする
		name_ratio = [0.5] * self.player_count
		for key, name in enumerate(chara_name):
			name = name.upper()
			for chara, jpname in self.chara.items():
				ratio = difflib.SequenceMatcher(None, name, chara).ratio()
				if (name_ratio[key] <= ratio):
					self.chara_name[key] = jpname
					name_ratio[key] = ratio
			# 誤検知しまくったキャラ or 新キャラの為に残しておく
			if (self.chara_name[key] == ''):
				self.chara_name[key] = chara_name[key] +'?'

		# なかったら作成だけしておく
		self._store_battle_rate(self.chara_name)

		self.logger.info('with%s=%s=>%s,%s', self.player_count, chara_name, self.chara_name, name_ratio)

	# [時計マーク] {N}:00 [人マーク] {P} 画面
	def _capture_time_stock_frame(self, capture_gray_image):
		#self.logger.debug(f'_capture_time_stock{self.ratio}')

		time_stock_pos = { 'time':[275, 333], 'stock':[358, 333] }
		num = {'time':-1, 'stock':-1}
		for key, pos in time_stock_pos.items():
			area_image = capture_gray_image[ pos[1]:pos[1]+15, pos[0]:pos[0]+10 ]
			_, mask_image = cv2.threshold(area_image, 100, 255, cv2.THRESH_BINARY)
			mask_image = cv2.bitwise_and(area_image, mask_image)

			area_image = cv2.bitwise_not(area_image)
			area_image, _ = Utils.trimming_any_rect(
				area_image, mask_image, 2, 5, filled=True)

			#cv2.imshow(f'_capture_time_stock_frame_{key}', area_image)
			n = self.ocr_tool.image_to_string(
				Utils.cv2pil(area_image), lang='eng',
				builder=pyocr.builders.DigitBuilder(tesseract_layout=10) )

			try:
				num[key] = int(n)
			except ValueError:
				pass

		time = stock = -1
		if (any([ -1 != n for n in num.values() ])):
			for k, v in enumerate(num.items()):
				if (-1 == v):
					num = num.pop(k, None)
			num = { k:v for k, v in num.items() if -1 != v }
			self.time_stock_ratio.append(num)

			# 一番検出された回数が多いものをそれとする
			time_list = list(Counter(n['time'] for n in self.time_stock_ratio if 'time' in n).keys())
			if (0 < len(time_list)):
				time = time_list[0]
			stock_list = list(Counter(n['stock'] for n in self.time_stock_ratio if 'stock' in n).keys())
			if (0 < len(stock_list)):
				stock = stock_list[0]

			if ( time != self.battle_time[0].minute and time in [3,5,7] ):
				self.battle_time[0] = datetime.datetime(1900, 1, 1, 0, time)
				self.logger.debug(f' better {time}:{stock} <= {num}:{self.time_stock_ratio}')

			if ( stock != self.result_stock['max'][0] and stock in [1,2,3] ):
				self.result_stock['max'] = [stock] * self.player_count
				self.result_stock['count'] = self.result_stock['max']
				self.logger.debug('stock max=%s', self.result_stock['max'])

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
			stock_area_image[key], _ = Utils.trimming_any_rect(
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
	def _capture_result_frame(self, capture_gray_image, player):
		if (4 == self.player_count):
			# 戦闘力の数字の画面はおそらく捕捉しきれないのであきらめるんご！
			return
		
		# 戦闘力の検出
		width = int(self.capture_image.shape[1])
		height = int(self.capture_image.shape[0])
		power_area_image = list(range(2))
		gray_power_area_image = list(range(2))
		convert_image = cv2.bitwise_and(self.capture_image, self.result_power_mask)
		power_area_image[0] = convert_image[int(height/4):int(height/2), 0:int(width/20*9)]
		power_area_image[1] = convert_image[int(height/4):int(height/2), int(width/20*11):width]
		convert_image = cv2.bitwise_and(capture_gray_image, cv2.cvtColor(self.result_power_mask, cv2.COLOR_RGB2GRAY))
		gray_power_area_image[0] = convert_image[int(height/4):int(height/2), 0:int(width/20*9)]
		gray_power_area_image[1] = convert_image[int(height/4):int(height/2), int(width/20*11):width]
		# 爆速+連打ニキがいて検出できなくなるので逆順で評価
		for key, image in enumerate(power_area_image):
			image, _ = Utils.trimming_any_rect(image, gray_power_area_image[key])

			convert_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			_, convert_image = cv2.threshold(convert_image, 127, 255, cv2.THRESH_BINARY_INV)
			image, _ = Utils.trimming_any_rect(image, convert_image, 1, 1)

			power = self.ocr_tool.image_to_string(
				Utils.cv2pil(image), lang='eng',
				builder=pyocr.builders.DigitBuilder(tesseract_layout=6) )
			#cv2.imshow('power_area_image'+ str(key), image)		# debug only
			#cv2.imshow('gray_power_area_image'+ str(key), convert_image)		# debug only
			try:
				self.power[key] = int(Utils.replace_invalid_char(power))
			except ValueError:
				pass

	# 試合終了
	def _battle_end(self):
		if (self.frame_state != self.FrameState.FS_BATTLE_END):
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


	""" is系 (watch含む) 検出時間(昇順) """
	# Loading
	def _is_loading_frame(self):
		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.loading_color, self.loading_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.99 <= self.ratio

	# [READY to FIGHT]画面の検出
	def _is_ready_frame(self, capture_image=None):
		if (capture_image is None):
			capture_image = self.capture_image

		with concurrent.futures.ThreadPoolExecutor() as executor:
			result = list(executor.map(
				Utils.match_masked_color_image,
				[capture_image, capture_image],
				self.ready_to_fight_color,
				[self.ready_to_fight_mask, self.ready_to_fight_mask]
				))
		self.ratio = ratio = round(float(max([ _[0] for _ in result ])), 4)
		return 0.99 <= ratio, ratio

	# [対戦相手募集中 -> もうすぐ開始]画面の検出
	def _is_ready_ok_frame(self):
		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.ready_ok_color, self.ready_ok_mask, is_trans=True)
		self.ratio = round(float(self.ratio), 3)
		return 0.97 <= self.ratio

	# 4人版の[準備OK]画面の検出
	def _is_with_4_battle_frame(self):
		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.with_4_battle_color, self.with_4_battle_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.95 <= self.ratio

	# [spam vs ham]画面かどうか
	def _is_vs_frame(self):
		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.vs_color, self.vs_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.97 <= self.ratio

	# [紅蒼乱闘戦]画面かどうかをついでに返す
	def _watch_smash_or_team_frame(self):
		convert_image = self.capture_image
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
	def _is_go_frame(self):
		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.battle_time_zero_color, self.battle_time_zero_mask, is_trans=True)
		self.ratio = round(float(self.ratio), 3)
		return 0.98 <= self.ratio

	# [時計マーク] {N}:00 [人マーク] {P} 画面
	def _is_time_stock_frame(self):
		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.rule_time_stock_color, self.rule_time_stock_mask, is_trans=True)
		self.ratio = round(float(self.ratio), 3)
		return 0.99 <= self.ratio

	# プレイヤー毎の [0.0%] を監視
	def _watch_player_zero_damage(self):
		convert_image = self.capture_image
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
	def _watch_stock_changed_frame(self):
		convert_image = self.capture_image
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
	def _is_stock_number_frame(self):
		with concurrent.futures.ThreadPoolExecutor() as executor:
			result = list(executor.map(
				Utils.match_masked_color_image,
				[self.capture_image, self.capture_image],
				[self.stock_hyphen_color_black, self.stock_hyphen_color_white],
				[self.stock_hyphen_mask, self.stock_hyphen_mask]
				))
		self.ratio = round(float(max([ _[0] for _ in result ])), 3)
		return 0.98 <= self.ratio

	# [GAME SET][TIME UP]画面かどうか
	def _is_game_end_frame(self):
		with concurrent.futures.ThreadPoolExecutor() as executor:
			result = list(executor.map(
				Utils.match_masked_color_image,
				[self.capture_image, self.capture_image],
				[self.game_set_color, self.time_up_color],
				[self.game_set_mask, self.time_up_mask],
				[True, True]
				))
		self.ratio = round(float(max([ _[0] for _ in result ])), 3)
		return 0.98 <= self.ratio

	# 結果画面(戦闘力が見える)かどうかをついでに返す
	def _watch_reuslt_frame(self, capture_gray_image):
		if (2 == self.player_count):
			order_pos = [ [200, 0], [470, 0] ]
		elif (4 == self.player_count):
			order_pos = [ [90, 0], [250, 0], [420, 0], [580, 0] ]

		for reverse_player, pos in enumerate(reversed(order_pos)):
			player = self.player_count - reverse_player - 1

			# 順位の検出
			convert_image =  self.capture_image[pos[1]:(100+pos[1]), pos[0]:(100+pos[0])]
			order_color_len = len(self.result_player_order_color)
			order_ratio = [-1] * order_color_len
			pos = [0,0] * order_color_len

			with concurrent.futures.ProcessPoolExecutor() as process_executor:
				result = list(process_executor.map(
					Utils.match_masked_color_image,
					[convert_image] * order_color_len,
					self.result_player_order_color,
					self.result_player_order_mask,
					[True] * order_color_len
					))
			order_ratio = [ round(float(ratio[0]), 3) for ratio in result ]
			order = numpy.argmax(order_ratio)

			self.ratio = order_ratio[order]
			order = order % 4	# 順位のテンプレートに大きさが異なるものがあるため

			# 戦闘力だけは順位が決まっても取得し続ける
			self._capture_result_frame(capture_gray_image, player)

			if (self.player_order_ratio[player] <= self.ratio and
				self.player_order[player] != int( 1 + order ) and
				order in range(0, self.player_count) ):
				# 同時に結果が捕捉できるタイミングが違うのでプレイヤーごとに処理 (かつ片方が取れるかもしれない状況ということはもう片方が取れる確率も高いので power 判定はもう片方のも行う)
				# しかもより確率が高い order が検出された場合,更新する
				#cv2.imshow('_watch_reuslt_frame_masked{0}'.format(player), convert_image)	# debug only

				self.player_order_ratio[player] = self.ratio
				self.player_order[player] = int( 1 + order )
				self.logger.info('player,order,power=%s,%s,%s', player, self.player_order[player], self.power[player])

		if ( all([-1 != order for order in self.player_order]) and any([ 10000 < power for power in self.power ]) ):
			return True

		return False

	# 「同じ相手との再戦を希望しますか？」画面か
	def _is_battle_retry_frame(self):
		if ( all([(-1 == order) for order in self.player_order]) ):
			# 順位が全く取れていない場合は何もしない(※これはイメージです の画像くらい違うものを検出してる可能性があるため)
			return False

		self.ratio, _ = Utils.match_masked_color_image(self.capture_image, self.battle_retry_color, self.battle_retry_mask)
		self.ratio = round(float(self.ratio), 3)
		return 0.98 <= self.ratio

	# frame がどこかを特定して応じて処理をする
	def _capture_any_frame(self):

		gray_image = cv2.cvtColor(self.capture_image, cv2.COLOR_RGB2GRAY)

		#cv2.imshow('capture', self.capture_image)	# debug only

		if (self._is_loading_frame()):
			# まっ黒に近い画面は処理しない
			return

		# [READY to FIGHT!]画面だけは検出する、どの状況でも。なぜなら正常に終わらなくても回ってくるので、回線落ちなどで (英文表記)
		is_ready_frame, _ = self._is_ready_frame()
		if (is_ready_frame):
			# 前回の試合の結果をうまく検出できていなかった時用の修正用
			self._battle_end()
			# (戦闘力の差分)
			self._capture_ready_to_fight_frame()
			self.frame_state = self.FrameState.FS_READY_TO_FIGHT
			return
		elif (self.frame_state in {self.FrameState.FS_UNKNOWN}):
			return

		if (self.frame_state in {self.FrameState.FS_READY_TO_FIGHT, self.FrameState.FS_RESULT, self.FrameState.FS_SAVED}):
			# [まもなく開始]画面 (プレイヤー名)
			if (self._is_ready_ok_frame()):
				self._battle_end()
				self._default_battle_params()
				self.frame_state = self.FrameState.FS_READY_OK
				self._capture_ready_ok_frame(gray_image)

			if (self._is_battle_retry_frame()):
				if (self.frame_state == self.FrameState.FS_RESULT):
					self.frame_state = self.FrameState.FS_SAVED

			return

		if (self.frame_state == self.FrameState.FS_READY_OK):
			if (2 == self.player_count):
				# battle with 4 (人数を 2 から 4 へ変更,もはやプレイヤー名とか正確に取れないので無視する)
				if (self._is_with_4_battle_frame()):
					self.logger.debug(f'_is_with_4_battle_frame{self.ratio}')
					entry_name = self.entry_name
					self._default_battle_params(player_count=4)
					self.entry_name = entry_name
					self.frame_state = self.FrameState.FS_READY_OK
					return

				# [{foo} vs {bar}]画面 (1on1,キャラクター名)
				# TODO:開始時間をここでついでにとりたい
				if (self._is_vs_frame()):
					self.frame_state = self.FrameState.FS_WHAT_CHARACTER
					self._capture_character_name(gray_image)

					_ = list(self.process_executor.map(cv2.imwrite, ['last_vs_character.png'], [self.capture_image]))

			elif (4 == self.player_count):
				# 大乱闘 or チーム乱闘 (smash or team,キャラクター名)
				is_smash_or_team_frame = self._watch_smash_or_team_frame()
				if (is_smash_or_team_frame):
					self.frame_state = self.FrameState.FS_WHAT_CHARACTER
					self._capture_character_name(gray_image)
					_ = list(self.process_executor.map(cv2.imwrite, ['last_vs_character.png'], [self.capture_image]))
			return

		if (self.frame_state == self.FrameState.FS_WHAT_CHARACTER):
			# [.00]画面 (開始時間)
			if (self._is_go_frame()):
				self.logger.debug(f'_is_go_frame{self.ratio}')
				self.frame_state = self.FrameState.FS_BATTLE

			# [時計マーク] {N}:00 [人マーク] {P} 画面
			if (self._is_time_stock_frame()):
				self._capture_time_stock_frame(gray_image)

			return

		if (self.frame_state == self.FrameState.FS_BATTLE):
			# [GAME SET][TIME UP]画面
			if (self._is_game_end_frame()):
				self.logger.debug(f'_is_game_end_frame{self.ratio}')
				self.frame_state = self.FrameState.FS_BATTLE_END
				return

			# [0.0]% 監視 それ以外だと [1.0]% が代入される
			self._watch_player_zero_damage()

			# [N - N]画面 (ストック数) (この画面は [1on1]でルール[stock] の時しか表示されない)
			if (self._is_stock_number_frame()):
				self._capture_stock_number_frame(gray_image)

			# ストックの[-1 or +1]を監視,観測するとダメージに [-1.0]% が代入される (この画面はルール[time]の時しか表示されない)
			self._watch_stock_changed_frame()

			return

		if (self.frame_state == self.FrameState.FS_BATTLE_END):
			# [結果]画面 (戦闘力,勝敗)
			is_result_frame = self._watch_reuslt_frame(gray_image)
			if (self._is_battle_retry_frame()):
				# 爆速ニキがどうしても捕捉できないので片方の戦歴から予測して調整する
				self.logger.debug(f'_is_battle_retry_frame{self.ratio}')
				is_result_frame = True

			if (is_result_frame):
				self._battle_end()
				self.frame_state = self.FrameState.FS_RESULT
				_ = list(self.process_executor.map(cv2.imwrite, ['last_foo_vs_bar_power.png'], [self.capture_image]))

			return


	# 文字 が主な情報
	def _user_interface_text(self):
		# 表示する情報
		chara_name = list(self.chara_name)
		base_char = 65
		for key, chara in enumerate(self.chara_name):
			if ('' == chara):
				chara_name[key] = chr(base_char)
				base_char += 1
		player_info = dict(zip( tuple(chara_name), list(zip( self.result_stock['count'], self.group_color )) ))
		battle_information_text = {}
		streak_name = (self.config['option']['battle_information']['streak_loses'] if self.battle_streak_rate < 0 else self.config['option']['battle_information']['streak_wins'])
		for name in ['gui_text', 'cui_text']:
			battle_information_text[name] = self.config['option']['battle_information'][name].format(
					frame=self.FrameState(self.frame_state + 1).name, ratio=self.ratio,
					entry_name=self.entry_name, group_name=self.group_name,
					group_color=self.group_color,
					chara_name=self.chara_name, stock=self.result_stock['count'],
					player_info=f'{player_info}',
					battle_rate=self.battle_rate[tuple(self.chara_name)],
					streak=abs(self.battle_streak_rate),
					streak_name=streak_name,
					streak_ratio=self.battle_streak_ratio
				)

		if (self.config['option']['battle_informationGUI']):
			self.gui_info['image'] = Image.fromarray(self.gui_info['image'])

			"""
			draw = ImageDraw.Draw(self.gui_info['image'])
			pos = tuple(self.config['option']['battle_information']['pos'])
			if ( self.config['option']['battle_information']['bold'] != self.config['option']['battle_information']['back'] ):
				# bold に背景色でない色を指定すると bold もどきにする
				for y in [-1, 1]:
					for x in [-1, 1]:
						txt = draw.text(
							(pos[0] + x, pos[1] + y),
							battle_information_text['gui_text'], font=self.font,
							fill=tuple(self.config['option']['battle_information']['bold'])
							)
			txt = draw.text(
				pos, battle_information_text['gui_text'], font=self.font,
				fill=tuple(self.config['option']['battle_information']['color'])
				)
			"""

			self.gui_info['image'] = numpy.array(self.gui_info['image'])
			self.gui_info['image'] = Utils.pil2cv(self.gui_info['image'])

		if (self.config['option']['battle_informationCUI']):
			Utils.width_full_print(battle_information_text['cui_text'])

		return self.gui_info

	# FS_WHAT_CHARACTER 中のアニメーション
	def _character_rate_graph(self):
		win = self.battle_rate[tuple(self.chara_name)]['win']
		lose = self.battle_rate[tuple(self.chara_name)]['lose']
		if (4 == self.animation_count):
			# グラフの作成,および試合数 0 からの遷移
			self._make_interface_canvas()
			self._make_character_rate_graph(win, lose)

		self.animation_count -= 1 if -10 < self.animation_count else 0
		# 本体と合成
		h, w = self.animation_image.shape[0:2]
		to_h, to_w = self.gui_info['image'].shape[0:2]
		self.gui_info['image'][ to_h-h:to_h,
			int((to_w-w)/2):int((to_w-w)/2+w) ] = self.animation_image

		# 消滅アニメ
		alpha = 1.0 / max(10-abs(self.animation_count), 1)
		black_image = Utils.pil2cv( numpy.zeros((h, w, 3)) )
		self.animation_image = cv2.addWeighted(black_image, (1-alpha), self.animation_image, (1-alpha), 0)
		# %表示
		self.gui_info['image'] = Image.fromarray(self.gui_info['image'])
		draw = ImageDraw.Draw(self.gui_info['image'])
		alpha = 1.0 - alpha
		color = list(self.config['option']['battle_information']['color'])
		color = tuple( map(lambda c: max(int(c * alpha), 0), color) )
		left_text = f'{win}勝' # 左寄せは 単位のせいで python デフォルトの機能で出来ないのでほげほげする
		left_text += ' ' * (8-len(left_text))
		x_margin = len(left_text)*10	# 10 = fontsize / 2
		draw.text( (to_w/2-x_margin, to_h-h/2), left_text, font=self.font, fill=color )
		draw.text( (to_w/2, to_h-h/2), f'{lose: >6}勝', font=self.font, fill=color )
		
		self.gui_info['image'] = numpy.array(self.gui_info['image'])
		self.gui_info['image'] = Utils.pil2cv(self.gui_info['image'])

	# FS_SAVED 後のアニメーション
	def _streak_rate(self):
		if (len(self.power_history[self.chara_name[0]]) < 2):
			return

		if (50 == self.animation_count):
			self.animation_image = numpy.zeros(self.capture_image.shape, dtype=numpy.uint8)

		battle_streak_ratio_max = self.config['option']['battle_streak_ratio_max']
		battle_streak_ratio_max_len = len(battle_streak_ratio_max)
		change_count = int(50 / battle_streak_ratio_max_len)
		if (0 == self.animation_count % change_count or 50 == self.animation_count):
			self.logger.debug(f'make animation_retry {self.animation_count} / {change_count}')
			streak_max = battle_streak_ratio_max[ int((50-self.animation_count) / change_count) ]
			power_history = self.power_history[self.chara_name[0]][::-1]
			power_history = power_history[0:streak_max]

			num = numpy.array(power_history)[:,0]	# 試合番号
			power = numpy.array(power_history)[:,1]	# 戦闘力

			self._make_interface_canvas()
			self.gui_info['ax'].plot(num, power, marker='.', color='pink', label='世界戦闘力の推移')

			self.gui_info['ax'].set_xlabel(f'試合数')
			self.gui_info['ax'].set_ylabel('世界戦闘力')
			self.gui_info['ax'].set_title(f'{self.chara_name[0]}の勝率 直近({len(power_history)}戦)', color='white')
			self.gui_info['ax'].xaxis.label.set_color('gray')
			self.gui_info['ax'].yaxis.label.set_color('gray')
			self.gui_info['ax'].tick_params(axis='x', colors='gray')
			self.gui_info['ax'].tick_params(axis='y', colors='gray')
			axlegend = self.gui_info['ax'].legend()
			axlegend.get_frame().set_edgecolor('#CFCFCF')
			for axtext in axlegend.get_texts():
				axtext.set_color('gray')

			self.gui_info['fig'].canvas.draw()
			image = numpy.array(self.gui_info['fig'].canvas.renderer.buffer_rgba())
			self.animation_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

		self.gui_info['image'] = Utils.paste_image_pos(self.gui_info['image'], self.animation_image, (0, 0))

		self.animation_count -= 1 if 1 < self.animation_count else 0


	# self.animation_state == begin_animation_state で do_condition を満たしていれば,アニメーションする
	# end_condition を満たしていると,デフォルトにする
	# TODO:増えてきたらfor化
	def _do_animation(self, begin_animation_state=AnimationState.AS_NONE, do_condition=False, end_condition=False):
		if (end_condition):
			if (self.animation_state == self.AnimationState(begin_animation_state+1) ):
				# 自動終了処理
				self.animation_state = self.AnimationState.AS_NONE
			return

		if (not do_condition):
			if (end_condition):
				# 主に事前準備でここにくる
				self.animation_state = self.AnimationState.AS_NONE
			if (self.animation_state == self.AnimationState.AS_NONE):
				return

		# キャラ別勝率のグラフ
		if (self.animation_state == self.AnimationState.AS_CHARACTER_RATE_GRAPH):
			self._character_rate_graph()
		# キャラ別戦歴のグラフ
		if (self.animation_state == self.AnimationState.AS_STREAK_RATE):
			self._streak_rate()

		if (self.animation_state == self.AnimationState.AS_NONE):
			if (begin_animation_state in {self.AnimationState.AS_CHARACTER_RATE_GRAPH_BEGIN, self.AnimationState.AS_CHARACTER_RATE_GRAPH}):
				# _character_rate_graph に必要な初期化
				self.animation_count = 4
				self.animation_state = self.AnimationState.AS_CHARACTER_RATE_GRAPH

		if (self.animation_state == self.AnimationState.AS_NONE):
			if (begin_animation_state == self.AnimationState.AS_STREAK_RATE_BEGIN):
				# _streak_rate に必要な初期化
				self.animation_count = 50
				self.animation_state = self.AnimationState.AS_STREAK_RATE

		if (not do_condition):
			if (begin_animation_state == self.AnimationState.AS_STREAK_RATE):
				self.animation_count = 0
				self.animation_state = self.AnimationState.AS_NONE

	# アニメーション が主な情報
	def _auto_begin_animation(self):
		if (not self.config['option']['battle_information_animation']):
			return
		if (2 != self.player_count):
			# 非対応
			return

		# [VS]画面中に キャラ別勝率を表示する
		self._do_animation(
			self.AnimationState.AS_CHARACTER_RATE_GRAPH_BEGIN,
			self.FrameState.FS_WHAT_CHARACTER == self.frame_state and -10 < self.animation_count,
			self.frame_state in {self.FrameState.FS_READY_OK, self.FrameState.FS_BATTLE}
		)

		# 結果が終わったあとのローディング画面,キャプチャ画面が検出されていない初期状態 でキャラ別戦歴のグラフを表示する
		self._do_animation(
			self.AnimationState.AS_STREAK_RATE_BEGIN,
			(self._is_loading_frame() and self.frame_state in {self.FrameState.FS_RESULT, self.FrameState.FS_SAVED}) or 
				(self.frame_state == self.FrameState.FS_UNKNOWN),
			self.frame_state in {self.FrameState.FS_RESULT, self.FrameState.FS_READY_TO_FIGHT}
		)

	# guiやanimationのための基盤の作成
	def _make_interface_canvas(self):
		if (self.gui_info is None):
			gui_image = numpy.zeros( (self.resource_size['height'], self.resource_size['width'], 3), dtype=numpy.uint8 )
			gui_image[:] = tuple(self.config['option']['battle_information']['back'])
			gui_image = numpy.array(gui_image)
			gui_image = Utils.pil2cv(gui_image)
		else:
			gui_image = self.gui_info['image']

		# figure,axesの削除、塗りつぶし
		plt.gcf().clear()
		plt.gca().clear()
		gui_image[:] = tuple(self.config['option']['battle_information']['back'])
		fig, ax = plt.subplots(
			figsize=(self.resource_size['width']/100, self.resource_size['height']/100), dpi=100,
			facecolor=tuple(self.config['option']['battle_information']['back'])
			)
		ax.patch.set_facecolor( tuple(self.config['option']['battle_information']['back']) )

		self.gui_info = { 'image':gui_image, 'fig':fig, 'ax':ax }

	# キャラクター戦歴の円グラフの作成
	def _make_character_rate_graph(self, win, lose):
		self.gui_info['ax'].pie(
			numpy.array([lose, win]),
			labels=['']*2,
			counterclock=False, startangle=90, radius=0.2, labeldistance=1.0,
			wedgeprops={'linewidth': 2, 'edgecolor':"black"},
			colors=['lightblue', 'pink'], textprops={'color':'white'})
		self.gui_info['fig'].canvas.draw()
		image = numpy.array(self.gui_info['fig'].canvas.renderer.buffer_rgba())
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
		# グラフの部分だけ取り出して,
		h, w = image.shape[0:2]
		image = image[ int(h/3+2):int(h/3*2+2), int(w/5*2+8):int(w/5*3+8) ]	# matplotlibのグラフなんで中央に描画してくれないん？？？
		h, w = image.shape[0:2]
		si_h, si_w = self.smabro_icon.shape[0:2]
		x = int((w-si_w)/2)
		y = int((h-si_h)/2)
		# iconの透過色付きの貼り付け
		self.animation_image = Utils.paste_image_pos(image, self.smabro_icon, (x, y))


	# 1 frame 中の処理
	def _main_frame(self):
		self._capture_window()

		self._capture_any_frame()

	# バトル中に表示する情報
	def _main_ui_frame(self):
		if (self.config['option']['battle_informationGUI'] or self.config['option']['battle_information_animation']):
			if (self.gui_info is None):
				self._make_interface_canvas()
			else:
				self.gui_info['image'][:] = tuple(self.config['option']['battle_information']['back'])

		self._do_animation()
		self._auto_begin_animation()

		self._user_interface_text()

	# 初期化処理
	def _initialize(self):
		Utils.width_full_print('initialize...', logger_func=self.logger.debug)
		self._load_config()
		self._load_resources()
		self._default_battle_params()
		self._init_capture_area()
		self._default_battle_params()
		self._load_historys()

	# 終了処理
	def _finalize(self):
		self.logger.info('finalize')

		with open(self.config_path, mode='w', encoding='utf-8') as file:
			file.write( json.dumps([self.config], indent=4, ensure_ascii=False) )

		Utils.width_full_print('exit.', logger_func=self.logger.info)

	# main loop
	def _main_loop(self):
		time.sleep(0.1)
		Utils.width_full_print('main ready.', logger_func=self.logger.info)
		while( True ):
			self._main_frame()
			self._main_ui_frame()

			cv2.waitKey(0)

			# kivy の終了を監視
			if (self.kivy_app is None):
				break

	def _kivy_loop(self):
		Utils.width_full_print('kivy ready.', logger_func=self.logger.info)
		self.kivy_app.run()
		self.kivy_app = None

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

			# kivy が run から処理を離してくれない
			# Clock.schedule_interval は Unity などと多分一緒で、処理が重くなるとフレームがスキップされる可能性があるため
			# GILの仕組み上マルチスレッドにしてもコルーチンと変わらないと思うけど run() が yeild形式なのでやむなし
			with concurrent.futures.ThreadPoolExecutor() as executor:
				_ = list(executor.map(
					lambda i: (self._main_loop if i else self._kivy_loop)(),
					list(range(2))
					))
		except KeyboardInterrupt:
			App.get_running_app().stop()
			self.kivy_app = None
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

	# kivy への独自イベントの on_update を処理してもらうモンキーパッチ
	def add_on_update_all():
		class_list = dict(Factory.classes)
		loaded_module = set(sys.modules)
		for class_name, item in class_list.items():
			cls = item['cls']
			# 下記のものを import/unimport したら (Factory から warning がでる) or (window の再ロードが起きる)
			if (cls is None and not (
					'Action' in class_name or 'Svg' in class_name or 'DropDown' in class_name or
					'Settings' in class_name or 'Spinner' in class_name or 'RstDocument' in class_name
				)):
				if (item['module']):
					module = __import__(
						name=item['module'],
						fromlist='*',
						level=0  # force absolute
					)
					if ( hasattr(module, class_name) ):
						cls = getattr(module, class_name)

					if (not cls is None):
						if ( hasattr(cls, '__events__') or re.search(r'[\w]+Layout$', class_name) ):
							cls = Factory.get(class_name)
							curse(cls, 'on_update', lambda *args, **kwargs: True)

							cls.__events__ += ('on_update',)

					# 適宜 unimport していかないと、メモリかなんかで python がデッドロックになる
					for module_name in list( set(sys.modules) - loaded_module ):
						sys.modules.pop(module_name)
	add_on_update_all()

	# strさん にお猿さんの呪いの実で +単項演算子 で パッケージ?コンパイル済みリソース? へのアクセスを処理してもらうようにする
	curse(str, '__pos__', _resource)

	# exe にした時に multi process だと exe 自体を新しく起動しようとするバグ回避
	multiprocessing.freeze_support()

	# 重い処理の途中で ctrl+c できるように
	signal.signal(signal.SIGINT, signal.SIG_DFL)

	# main engine
	engine = SmaBroEngine()
	engine.run()
