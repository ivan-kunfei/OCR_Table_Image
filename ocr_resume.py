from skimage import io, morphology
import numpy as np
import matplotlib.pyplot as plt
import os
from cnocr import CnOcr
from pandas import DataFrame


class Cell:
	def __init__(self, img, location):
		self.img = img
		self.location = location
		self.row_index = None
		self.col_index = None


# 得到二值化图像  传入0,1黑白图像
def get_binary_img(img, bi_th=0.95):
	img_binary = np.array(img)
	img_binary[img_binary <= bi_th] = 0
	img_binary[img_binary >= bi_th] = 1
	return img_binary


# 补黑边
def add_black_line(img, width=3):
	img = np.array(img)
	img[:, width - 1] = 0
	img[:, 1 - width] = 0
	img[width - 1, :] = 0
	img[1 - width, :] = 0
	return img


# 周边变白
def clean(img, width=3):
	# 去黑线
	img_binary = get_binary_img(img, bi_th=0.8)
	for i in range(img_binary.shape[0]):
		if np.sum(img_binary[i, :]) == 0:
			img[i, :] = 1
	for j in range(img_binary.shape[1]):
		if np.sum(img_binary[:, j]) == 0:
			img[:, j] = 1

	# 周边变白
	img = np.array(img)
	img[:, 0:width - 1] = 1
	img[:, 1 - width:-1] = 1
	img[0:width - 1, :] = 1
	img[1 - width:-1, :] = 1

	return img


def delete_black_line(img):
	# 去黑线
	img_binary = get_binary_img(img, bi_th=0.7)

	for i in range(img_binary.shape[0]):
		if np.mean(img_binary[i, :]) <= 0.03:
			img_binary[i, :] = 1
			img[i, :] = 1
	for j in range(img_binary.shape[1]):
		if np.mean(img_binary[:, j]) <= 0.03:
			img_binary[:, j] = 1
			img[:, j] = 1

	width = 7
	# 周边变白
	img_binary[:, 0:width - 1] = 1
	img_binary[:, 1 - width:-1] = 1
	img_binary[0:width - 1, :] = 1
	img_binary[1 - width:-1, :] = 1

	# 竖直方向去白
	temp_img = img_binary
	height, width = temp_img.shape
	idx_start, idx_end = None, None
	mean_row = np.mean(temp_img, 1)

	for i in range(height):
		if idx_start is None and mean_row[i] < 1:
			if i - 5 >= 0:
				idx_start = i - 5
			else:
				idx_start = 0
			break

	if idx_start is not None:
		for i in range(height - (idx_start + 3)):
			idx = - i - 1
			if mean_row[idx] < 1:
				idx_end = idx
				if i - 5 > 0:
					idx_end += 5
				else:
					idx_end = -1
				break

	if idx_start is not None and idx_end is not None:
		img = img[idx_start:idx_end, :]

	img_2 = np.array(img)

	img_binary = get_binary_img(img_2, 0.7)

	width = 7
	# 两边变白
	img_binary[:, 0:width - 1] = 1
	img_binary[:, 1 - width:-1] = 1

	temp_img = img_binary
	# 水平方向去白
	height, width = temp_img.shape
	idx_start, idx_end = None, None
	mean_col = np.mean(temp_img, 0)

	for i in range(width):
		if idx_start is None and mean_col[i] < 1:
			if i - 6 >= 0:
				idx_start = i - 6
			else:
				idx_start = 0
			break
	if idx_start is not None:
		for i in range(width - (idx_start + 3)):
			idx = - i - 1
			if mean_col[idx] < 1:
				idx_end = idx
				if i - 6 > 0:
					idx_end += 6
				else:
					idx_end = -1
				break
	if idx_start is not None and idx_end is not None:
		img_2 = img_2[:, idx_start:idx_end]

	return img_2


# 传入0,1黑白图像
def clean_white_vertical(img):
	temp_img = clean(img, width=8)
	temp_img = get_binary_img(temp_img)
	height, width = temp_img.shape
	idx_start, idx_end = None, None
	mean_row = np.mean(temp_img, 1)

	for i in range(height):
		if idx_start is None and mean_row[i] < 1:
			if i - 3 >= 0:
				idx_start = i - 3
			else:
				idx_start = 0
			break
	if idx_start is not None:
		for i in range(height - (idx_start + 1)):
			idx = - i - 1
			if mean_row[idx] < 1:
				idx_end = idx
				if i - 3 > 0:
					idx_end += 3
				else:
					idx_end = -1
				break
	if idx_start is not None and idx_end is not None:
		img = img[idx_start:idx_end, :]
	return img


# 传入0,1黑白图像
def clean_white_horizontal(img):
	temp_img = clean(img, width=8)
	temp_img = get_binary_img(temp_img)
	height, width = temp_img.shape
	idx_start, idx_end = None, None
	mean_col = np.mean(temp_img, 0)

	for i in range(width):
		if idx_start is None and mean_col[i] < 1:
			if i - 3 >= 0:
				idx_start = i - 3
			else:
				idx_start = 0
			break
	if idx_start is not None:
		for i in range(width - (idx_start + 1)):
			idx = - i - 1
			if mean_col[idx] < 1:
				idx_end = idx
				if i - 3 > 0:
					idx_end += 3
				else:
					idx_end = -1
				break
	if idx_start is not None and idx_end is not None:
		img = img[:, idx_start:idx_end]

	return img


def dil_ero(img, selem):
	img = morphology.dilation(img, selem)
	img = morphology.erosion(img, selem)
	return img


def cut_item(cellitem, idx_unirow):
	sub_items = []
	img = cellitem.img
	img = clean(img, width=8)  # 周边变白-----------

	height, width = img.shape
	row, col = cellitem.location
	bounds = []
	cut_indexes = [row]
	mean_line_pixels = np.mean(img, 1)
	flag = True
	a = None
	b = None

	for i in range(len(mean_line_pixels)):
		if flag and mean_line_pixels[i] < 0.97:  # if black
			a = i
			flag = False
		elif not flag and mean_line_pixels[i] >= 0.97:  # if white
			b = i
			flag = True
		if a is not None and b is not None:
			if b - a > 3:
				bounds.append((a, b))
			a = None
			b = None

	if len(bounds) <= 1:
		return None

	for i in range(len(bounds)):
		cut = None
		if i != len(bounds) - 1:
			down = bounds[i][1] + row
			up = bounds[i + 1][0] + row
			# print('down ',down,' up ',up)
			for each_row in idx_unirow:
				qualified_rows = []
				if down <= each_row <= up:
					qualified_rows.append(each_row)
				if qualified_rows:
					cut = max(qualified_rows)

			if not cut:  # 如果找不到 就用中间值
				cut = (down + up) // 2
				idx_unirow.append(cut)
		else:
			cut = height + row
		cut_indexes.append(cut)

	for i in range(len(cut_indexes) - 1):
		start_index = cut_indexes[i]
		end_index = cut_indexes[i + 1]
		sub_img = img[start_index - row:end_index - row, :]
		sub_item = Cell(sub_img, (start_index, col))
		sub_items.append(sub_item)
	return sub_items


def predict(file, cut=True):
	img_gray = io.imread(file, True)
	img_gray = add_black_line(img_gray)
	img_binary = get_binary_img(img_gray)  # 二值化图像
	img_binary_2 = get_binary_img(img_gray, bi_th=0.8)

	rows, cols = img_binary.shape
	scale = 30
	col_selem = morphology.rectangle(cols // scale, 1)  # 用长cols//scale,宽1的滤波器进行腐蚀膨胀
	row_selem = morphology.rectangle(1, rows // scale)  # 用长1，快rows//scale的滤波器进行腐蚀膨胀
	img_cols = dil_ero(img_binary, col_selem)  # 竖线图
	img_rows = dil_ero(img_binary, row_selem)  # 横线图
	img_line = img_cols * img_rows  # 有0的都取0 得到线图
	img_dot = img_cols + img_rows  # 有白的都取白 得到点图

	# 画二值化图像
	plt.imshow(img_binary, plt.cm.gray)
	plt.show()
	# 画线图
	plt.imshow(img_line, plt.cm.gray)
	plt.show()
	# 画点图
	# plt.imshow(img_dot, plt.cm.gray)
	# plt.show()

	for i in range(rows):
		for j in range(cols):
			if img_dot[i, j] == 2:
				img_dot[i, j] = 1

	'''
	点团浓缩为单个像素
	'''
	width = 7
	idx = np.argwhere(img_dot == 0)

	for each_dot in idx:
		cur_row, cur_col = each_dot
		for x in range(0, width + 1):
			for y in range(0, width + 1):
				if (not (x == 0 and y == 0)) and cur_row + x < rows and cur_col + y < cols:
					img_dot[cur_row + x, cur_col + y] = 1

	'''
	统一横纵坐标标准
	'''
	idx = np.argwhere(img_dot == 0)  # 点的坐标
	idx_unirow = np.unique(idx[:, 0])
	idx_unicol = np.unique(idx[:, 1])

	row_standard = []
	for i in range(len(idx_unirow)):
		each_unique_row = idx_unirow[i]
		if i == 0:
			row_standard.append(each_unique_row)
			continue
		if each_unique_row not in row_standard:
			flag = True
			for each_standard in row_standard:
				difference = abs(each_unique_row - each_standard)
				if difference <= 3:
					flag = False
					break
			if flag:
				row_standard.append(each_unique_row)

	col_standard = []
	for i in range(len(idx_unicol)):
		each_unique_col = idx_unicol[i]
		if i == 0:
			col_standard.append(each_unique_col)
			continue

		if each_unique_col not in col_standard:
			flag = True
			for each_standard in col_standard:
				difference = abs(each_unique_col - each_standard)
				if difference <= 3:
					flag = False
					break
			if flag:
				col_standard.append(each_unique_col)

	'''
	对点的横纵坐标进行统一
	'''

	for i in range(len(idx)):
		each_dot = idx[i]
		for each_row_standard in row_standard:
			row = each_dot[0]
			row_difference = abs(each_row_standard - row)
			if row_difference <= 3:
				each_dot[0] = each_row_standard
				break

		for each_col_standard in col_standard:
			col = each_dot[1]
			col_difference = abs(each_col_standard - col)
			if col_difference <= 3:
				each_dot[1] = each_col_standard
				break

	idx = np.unique(idx, axis=0)
	img_dot = np.ones((rows, cols))
	for each_dot in idx:
		row = each_dot[0]
		col = each_dot[1]
		img_dot[row, col] = 0

	idx = np.argwhere(img_dot == 0)  # 点的坐标


	# 点图
	plt.imshow(img_dot, plt.cm.gray)
	plt.show()

	'''对线的横纵坐标进行补线'''
	# 统一横线
	row_idx = np.argwhere(img_rows == 0)
	for i in range(len(row_idx)):
		each_row_dot = row_idx[i]
		row = each_row_dot[0]
		for each_row_standard in row_standard:
			difference = abs(row - each_row_standard)
			if difference <= 3:
				col = each_row_dot[1]
				img_rows[each_row_standard, col] = 0
				break
	# 统一竖线
	col_idx = np.argwhere(img_cols == 0)
	for i in range(len(col_idx)):
		each_col_dot = col_idx[i]
		col = each_col_dot[1]
		for each_col_standard in col_standard:
			difference = abs(col - each_col_standard)
			if difference <= 3:
				row = each_col_dot[0]
				img_cols[row, each_col_standard] = 0
				break

	# 画线图
	img_line = img_cols * img_line
	plt.imshow(img_line, plt.cm.gray)
	plt.show()
	# 画点图
	plt.imshow(img_dot, plt.cm.gray)
	plt.show()

	'''
	按黑线切分单元格
	'''
	idx = np.argwhere(img_dot == 0)  # 点的坐标
	idx_unirow = list(np.unique(idx[:, 0]))

	cell_items = []
	for i in range(len(idx_unirow) - 1):

		row_cur = idx_unirow[i]  # 第一行横坐标

		idx_row_cur = idx[idx[:, 0] == row_cur]  # 当前行的所有点坐标
		for j in range(len(idx_row_cur) - 1):  # 遍历当前行的所有点坐标作为起点
			# 检测左上角点是否可以作为起点
			# 检测左边是否连续
			col_cur = idx_row_cur[j][1]
			row_next_test_1 = idx_unirow[i + 1]
			black_line_test_1 = img_cols[row_cur:row_next_test_1, col_cur]
			if np.sum(black_line_test_1) != 0:
				continue

			# 检测上边是否连续
			col_next_text_2 = idx_row_cur[j + 1][1]
			black_line_test_2 = img_rows[row_cur, col_cur:col_next_text_2]
			if np.sum(black_line_test_2) != 0:
				continue

			for a in range(j + 1, len(idx_row_cur)):  # 遍历当前行的当前点右边的所有点的坐标

				col_next = idx_row_cur[a][1]  # 当前行下一个点（右上角点）的纵坐标
				idx_col_next = idx[idx[:, 1] == col_next]  # 下一列的所有点的坐标

				index = None
				length = len(idx_col_next)

				# 找到右上角的点在当前列的索引
				for b in range(length):
					if idx_col_next[b][0] == row_cur and idx_col_next[b][1] == col_next:
						index = b
						break

				# 检测右上角点是否可以作为起点
				if index + 1 < length:  # 如果这个起点不是这列的最下面的点
					row_next_test_3 = idx_col_next[index + 1][0]  # 右上角点下一行测试点的横坐标
					black_line_test_3 = img_cols[row_cur:row_next_test_3, col_next]
				else:
					continue
				if np.sum(black_line_test_3) != 0:
					continue
				for c in range(index + 1, length):
					row_next = idx_col_next[c][0]  # 第三个点的横坐标

					left_line = img_cols[row_cur:row_next, col_cur]
					right_line = img_cols[row_cur:row_next, col_next]

					if np.sum(left_line) != 0 or np.sum(right_line) != 0:
						break
					down_line = img_rows[row_next, col_cur:col_next]

					if np.sum(down_line) == 0:
						sub_img = img_binary_2[row_cur:row_next, col_cur:col_next]
						sub_item = Cell(sub_img, (row_cur, col_cur))
						cell_items.append(sub_item)
						break
				break

	cell_items_new = []
	if cut:
		# 更新切分
		for each_cell in cell_items:
			result = cut_item(each_cell, idx_unirow)
			if result:
				for each_sub_item in result:
					cell_items_new.append(each_sub_item)
			else:
				cell_items_new.append(each_cell)
	else:
		cell_items_new = cell_items

	x = []
	y = []

	for each in cell_items_new:
		x.append(each.location[0])
		y.append(each.location[1])

	unique_x = np.unique(x)
	unique_y = np.unique(y)
	unique_x.sort()
	unique_y.sort()

	df = DataFrame(np.full((len(unique_x), len(unique_y)), np.nan))
	# ocr = CnOcr(model_name='densenet-lite-fc', model_epoch=40, root='local_models')
	ocr = CnOcr()
	root = 'sub_cut_resume'

	if cut:
		t = 0
		for each_cell in cell_items_new:

			a, b = each_cell.location
			c = a + each_cell.img.shape[0]
			d = b + each_cell.img.shape[1]

			t += 1
			img = img_gray[a:c, b:d]

			img = delete_black_line(img)
			img *= 255

			row_index = np.argwhere(unique_x == each_cell.location[0])[0][0]
			col_index = np.argwhere(unique_y == each_cell.location[1])[0][0]

			dir = os.path.join(root,
							   str(t) + '_(' + str(row_index) + ', ' + str(col_index) + ')' + '[' + str(a) + ', ' + str(
								   b) + ']' + '.jpg')

			io.imsave(dir, img)

			try:
				value = ocr.ocr_for_single_line(img)
				value = ''.join(value)
			except:
				value = np.NAN

			print(value)

			df.iloc[row_index, col_index] = value

	else:
		t = 0

		for each_cell in cell_items_new:

			t += 1

			row_index = np.argwhere(unique_x == each_cell.location[0])[0][0]
			col_index = np.argwhere(unique_y == each_cell.location[1])[0][0]

			result = cut_item(each_cell, idx_unirow)
			values = []
			if result:
				for each_sub_cell in result:
					a, b = each_sub_cell.location
					c = a + each_sub_cell.img.shape[0]
					d = b + each_sub_cell.img.shape[1]
					sub_img = img_gray[a:c, b:d]

					sub_img = delete_black_line(sub_img)

					sub_img *= 255

					dir = os.path.join(root,
									   str(t) + '_(' + str(row_index) + ', ' + str(col_index) + ')' + '[' + str(
										   a) + ', ' + str(
										   b) + ']' + '.jpg')

					io.imsave(dir, sub_img)

					try:
						value = ocr.ocr_for_single_line(sub_img)
						value = ''.join(value)
					except:
						value = ''
					values.append(value)
				values = '\n'.join(values)
			else:
				a, b = each_cell.location
				c = a + each_cell.img.shape[0]
				d = b + each_cell.img.shape[1]
				img = img_gray[a:c, b:d]
				img = delete_black_line(img)
				img *= 255
				dir = os.path.join(root,
								   str(t) + '_(' + str(row_index) + ', ' + str(col_index) + ')' + '[' + str(
									   a) + ', ' + str(
									   b) + ']' + '.jpg')

				io.imsave(dir, img)

				try:
					values = ocr.ocr_for_single_line(img)
					values = ''.join(values)
				except:
					values = ''

			print(values)

			df.iloc[row_index, col_index] = values

	df.to_excel('result_resume.xlsx', index=False)


predict('resume.jpg', cut=False)
