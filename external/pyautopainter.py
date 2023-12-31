#!/usr/bin/env python

import os, math, sys, imageio, palettable, numpy as np, math, random, glob, threading, io, time, colorsys
import PIL.Image, PIL.ImageDraw, PIL.ImageOps
from flask import Flask, escape, request, send_file, render_template


class Palette:
	palette = None
	
	def __init__(self, palette=None):
		self.palette = palette
		
	def distance(self, c1, c2):
		(r1,g1,b1) = c1
		(r2,g2,b2) = c2
		return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) **2)

	def nearest_color(self, color, palette=None):
		if not palette:
			palette = self.palette
		colors_dict = {}
		for i in range(len(palette)):
			colors_dict[i] = palette[i]
		closest_colors = sorted(colors_dict, key=lambda point: self.distance(color, colors_dict[point]))
		code = tuple(colors_dict[closest_colors[0]])
		return code

	def retinted_color(self, color, palette=None):
		nearest = self.nearest_color(color)
		nearest_hsv = colorsys.rgb_to_hsv(nearest[0]/1.0, nearest[1]/1.0, nearest[2]/1.0)
		color_hsv = colorsys.rgb_to_hsv(color[0]/1.0, color[1]/1.0, color[2]/1.0)
		
		hue = (1.0 + lerp(color_hsv[0], nearest_hsv[0], 0.75)) % 1.0
		sat = max(0.0, min(1.0, lerp(color_hsv[1], nearest_hsv[1], 0.5)))
		value = max(0.0, min(255.0,lerp(color_hsv[2], nearest_hsv[2], 0.3)))
		
		new = colorsys.hsv_to_rgb(hue, sat, int(value))
		
		return (int(new[0]), int(new[1]), int(new[2]))
		
	def load_from_image(self, filename):
		image = PIL.Image.open(filename).convert('RGB')
		colors = image.getcolors(image.width*image.height)
		self.palette = []
		print('Loaded', len(colors), 'colors')
		for color in colors:
			self.palette.append((color[1][0], color[1][1], color[1][2]))

def lerp(v1, v2, d):
	return v1 * (1 - d) + v2 * d

def image_average_color(image):
	colour_tuple = [None, None, None]
	for channel in range(3):

		# Get data for one channel at a time
		pixels = image.getdata(band=channel)

		values = []
		for pixel in pixels:
			values.append(pixel)

		colour_tuple[channel] = sum(values) / len(values)
	return tuple([int(colour_tuple[0]), int(colour_tuple[1]), int(colour_tuple[2])])

def rect_average_color(image, box):
	rect = image.crop(box=box)
	return image_average_color(rect)

def create_brush_from_color(brush, color):
	new_brush = brush.copy()
	pixels = new_brush.load()
	for x in range(0,new_brush.width):
		for y in range(0,new_brush.height):
			pixel = pixels[x,y]
			pixels[x,y] = (color[0],color[1],color[2],int(color[3]*pixel[3]/255.0))
	return new_brush

def draw_brush(canvas, position, brush, brush_size):
	if brush.width != brush_size[0] or brush.height != brush_size[1]:
		brush = brush.resize(brush_size, PIL.Image.NEAREST)
	canvas.paste(brush, (int(position[0]-brush_size[0]*0.5),int(position[1]-brush_size[1]*0.5)), mask=brush)

class AutoPainter:
	canvas = None
	height_canvas = None
	generate_heightmap = True
	save_incremental = True
	save_gif = True
	gif_frames = []
	gif_size = 540
	color_distance_threshold = 20
	total_saved_index = 0
	configuration = []
	default_palette_names = ['(None)', 'greens/yellows', 'blues/purples/greens', 'inferno', 'mycarta cube1', 'purple/gray']
	configuration_names = ['default', 'detail', 'quick', 'Quick (darker subject)', 'Quick (lighter subject)', 'dark subject', 'light subject', 'dark subject (huge image)', 'light subject (huge image)']
	progress_image = io.BytesIO()
	running = False
	finished = True
	message = 'Idle'
	brush_size_multiplier = 1.25
	palette_strict = False
	def __init__(self):
		self.load_image('bobross_trees.jpg')
		self.configuration = self.get_configuration('quick')
		self.base_brushes = []
		brush_filenames = glob.glob('brushes/*.png')
		for fn in brush_filenames:
			brush = PIL.Image.open(fn).convert('RGBA')
			self.base_brushes.append(brush)
		self.canvas = PIL.Image.new('RGB', (512,512), (255,255,255))
		self.canvas.save(self.progress_image, format='JPEG')

	def load_image(self, filename):
		self.reference_image = PIL.Image.open('input/'+filename).convert('RGBA')
		self.reference_image = self.reference_image.convert('RGB')
	
	def setup_palette(self, palette_name):
		if palette_name == 'greens/yellows':
			return Palette(palettable.cartocolors.sequential.TealGrn_7.colors+palettable.cartocolors.sequential.agGrnYl_7.colors+palettable.cmocean.sequential.Algae_20.colors+palettable.cmocean.sequential.Turbid_20.colors)
		elif palette_name == 'blues/purples/greens':
			return Palette(palettable.cmocean.sequential.Dense_20.colors+palettable.cmocean.sequential.Deep_20.colors)
		elif palette_name == 'inferno':
			return Palette(palettable.matplotlib.Inferno_20.colors)
		elif palette_name == 'mycarta cube1':
			return Palette(palettable.mycarta.Cube1_15.colors)
		elif palette_name == 'purple/gray':
			return Palette(palettable.tableau.PurpleGray_12.colors)
		else:
			fn = os.path.join('palette', palette_name+'.png')
			if os.path.exists(fn):
				palette = Palette()
				palette.load_from_image(fn)
				return palette
			return None
	
	def stop(self):
		self.running = False
	
	def run(self):
		if self.running:
			print('Already running')
			return
		self.running = True
		self.finished = False
		self.message = 'Starting'
		self.canvas = PIL.Image.new('RGB', self.reference_image.size, (255,255,255))
		self.progress_image = io.BytesIO()
		self.canvas.save(self.progress_image, format='JPEG')
		self.height_canvas = PIL.Image.new('RGB', self.reference_image.size, (0,0,0))
		if self.autocontrast_cutoff:
			painter.reference_image = PIL.ImageOps.autocontrast(painter.reference_image, cutoff=self.autocontrast_cutoff, ignore=None)
		#if self.palette:
		#	self.reference_image = self.recolor_image(self.reference_image, self.palette)
		self.gif_frames = []
		
		for iteration in range(0,len(self.configuration)):
			if self.running:
				self.do_iteration(iteration, self.configuration, self.palette)
					
		self.canvas.save('output/out.png')
		if self.save_gif and len(self.gif_frames) > 1:
			self.gif_frames[0].save('output/out.gif', save_all=True, append_images=self.gif_frames[1:], optimize=False, duration=80, loop=0)
		if self.generate_heightmap:
			self.height_canvas.save('output/out_height.png')
		self.finished = True
		if self.running:
			self.message = 'Done'
		else:
			self.message = 'Idle'
		self.running = False
		print(self.message)

	def get_configuration(self, configuration_name):
		# [ (canvas_percent, alpha, ignore_lighter_than, ignore_darker_than) ]
		if configuration_name == 'default':
			return [
				(10, 1.0, 1, 0),
				(10, 0.75, 1, 0),
				(4, 0.95, 1, 0),
				(2, 0.9, 1, 0),
				(1, 0.8, 1, 0),
				(0.5, 0.5, 0.6, 0)
			]
		elif configuration_name == 'detail':
			return [
				(15, 1.0, 1, 0),
				(12, 0.75, 1, 0),
				(10, 0.75, 1, 0),
				(4, 0.95, 1, 0),
				(2, 0.9, 1, 0),
				(1, 0.8, 1, 0),
				(0.5, 0.7, 0.8, 0),
				(0.5, 0.5, 0.6, 0)
			]
		elif configuration_name == 'quick':
			return [
				(20, 1.0, 1, 0),
				(10, 0.95, 1, 0),
				(4, 0.9, 1, 0),
				(2, 0.85, 1, 0),
				(0.75, 0.9, 0.6, 0)
			]
		# [ (canvas_percent, alpha, ignore_lighter_than, ignore_darker_than) ]
		elif configuration_name == 'Quick (darker subject)':
			return [
				(20, 1.0, 0.9, 0.2),
				(10, 0.95, 0.8, 0.2),
				(4, 0.9, 0.7, 0.3),
				(2, 0.85, 0.6, 0.3),
				(2, 0.7, 0.65, 0.5),
				(0.75, 0.7, 0.25, 0)
			]
		elif configuration_name == 'Quick (lighter subject)':
			return [
				(20, 1.0, 1, 0.5),
				(10, 0.95, 0.6, 0.1),
				(4, 0.9, 0.7, 0.2),
				(2, 0.85, 0.9, 0.3),
				(2, 0.7, 0.5, 0.35),
				(0.75, 0.7, 0.75, 0)
			]
		elif configuration_name == 'dark subject':
			return [
				(50, 1.0, 1, 0),
				(50, 1.0, 1, 0),
				(10, 0.9, 1, 0),
				(10, 0.9, 1, 0),
				(10, 0.8, 1, 0),
				(8, 0.85, 1, 0),
				(6, 0.8, 0.9, 0),
				(4, 0.7, 0.9, 0),
				(3, 0.5, 0.9, 0),
				(2, 0.5, 1, 0),
				(2, 0.25, 0.8, 0),
				(2, 0.6, 0.8, 0),
				(2, 0.4, 1, 0),
				(1, 0.2, 0.8, 0),
				(0.5, 0.2, 0.7, 0),
				(0.25, 0.3, 0.6, 0)
			]
		elif configuration_name == 'light subject':
			return [
				(50, 1.0, 1, 0),
				(50, 1.0, 1, 0),
				(10, 0.9, 1, 0),
				(10, 0.9, 1, 0),
				(10, 0.8, 1, 0),
				(8, 0.85, 1, 0),
				(6, 0.8, 0.9, 0.1),
				(4, 0.7, 0.9, 0.1),
				(3, 0.5, 0.9, 0.1),
				(2, 0.5, 1, 0),
				(2, 0.25, 0.8, 0.2),
				(2, 0.6, 0.8, 0.2),
				(2, 0.4, 1, 0),
				(1, 0.2, 0.8, 0.2),
				(0.5, 0.2, 0.7, 0.3)
			]
		elif configuration_name == 'dark subject (huge image)':
			return [
				(25, 1.0, 1, 0),
				(25, 1.0, 1, 0),
				(10, 0.9, 1, 0),
				(10, 0.8, 1, 0),
				(8, 0.85, 1, 0),
				(6, 0.8, 0.9, 0),
				(4, 0.7, 0.9, 0),
				(3, 0.5, 0.9, 0),
				(2, 0.5, 1, 0),
				(2, 0.25, 0.8, 0),
				(2, 0.6, 0.8, 0),
				(2, 0.4, 1, 0),
				(1, 0.2, 0.8, 0),
			]
		elif configuration_name == 'light subject (huge image)':
			return [
				(25, 1.0, 1, 0),
				(25, 1.0, 1, 0),
				(10, 0.9, 1, 0),
				(10, 0.8, 1, 0),
				(8, 0.85, 1, 0),
				(6, 0.8, 1, 0.1),
				(4, 0.7, 1, 0.1),
				(3, 0.5, 1, 0.1),
				(2, 0.5, 1, 0),
				(2, 0.25, 1, 0.2),
				(2, 0.6, 1, 0.2),
				(2, 0.4, 1, 0),
				(1, 0.2, 1, 0.2),
			]
		else:
			return []

	def recolor_image(self, image, palette):
		print('Recoloring - Started')
		pixels = image.load()
		for x in range(0, image.width):
			for y in range(0, image.height):
				color = palette.nearest_color(pixels[x,y])
				pixels[x,y] = (color[0],color[1],color[2])
		print('Recoloring - Done')
		return image
	
	def do_iteration(self, iteration, percents, palette):
		size = max(self.reference_image.width, self.reference_image.height)
		null_palette = Palette()

		percent = percents[iteration]
		ignore_over_brightness = percent[2]
		ignore_under_brightness = percent[3]
		width_step = int(math.ceil(self.reference_image.width * percent[0] / 100))
		height_step = int(math.ceil(self.reference_image.height * percent[0] / 100))
		size = int(max(width_step*2*self.brush_size_multiplier, (height_step*2*self.brush_size_multiplier)))
		brush_size = (size, size)
		self.message = 'Iteration '+str(iteration+1)+' of '+str(len(percents))+' : brush size '+str(brush_size)
		print(self.message)
		color_areas = {}
		all_colors = []
		for x1 in range(0, self.reference_image.width, width_step):
			for y1 in range(0, self.reference_image.height, height_step):
				x2 = min(self.reference_image.width, x1 + width_step)
				y2 = min(self.reference_image.height, y1 + height_step)
				avg_color = rect_average_color(self.reference_image, (x1,y1, x2,y2))
				if palette:
					if self.palette_strict:
						color = palette.nearest_color(avg_color)
					else:
						color = palette.retinted_color(avg_color)
				else:
					color = avg_color
				
				key = str(color)
				center = (int(x1+width_step*0.5),int(y1+height_step*0.5))
				if key in color_areas:
					color_areas[key].append(center)
				else:
					color_areas[key] = [center]
					brightness = (color[0]+color[1]+color[2])/3
					if ignore_over_brightness >= brightness / 255.0 or ignore_under_brightness <= brightness / 255.0:
						all_colors.append((brightness, color))
		
		all_colors = sorted(all_colors, key=lambda x: x[0], reverse=True)
		resized_brushes = []
		for brush in self.base_brushes:
			resized_brushes.append(brush.resize(brush_size, PIL.Image.NEAREST))
		last_saved_index = 0
		current_height_canvas = None
		if self.generate_heightmap:
			current_height_canvas = PIL.Image.new('RGBA', self.reference_image.size, (0,0,0,0))
		last_time = time.time()
		for color_index in range(0, len(all_colors)):
			color = all_colors[color_index]
			if str(color[1]) in color_areas:
				brush_color = (color[1][0], color[1][1], color[1][2], int(percent[1]*255))
				
				for pixel in color_areas[str(color[1])]:
					if not self.running:
						return
					rnd = (random.uniform(-width_step*0.5*self.brush_size_multiplier, width_step*0.5*self.brush_size_multiplier),  random.uniform(-height_step*0.5*self.brush_size_multiplier, height_step*0.5*self.brush_size_multiplier))
					
					# compare brush color with average canvas color
					rect = ((pixel+rnd)[0]-brush_size[0]*0.5, (pixel+rnd)[1]-brush_size[1]*0.5, (pixel+rnd)[0]+brush_size[0]*0.5, (pixel+rnd)[1]+brush_size[1]*0.5)
					compare_color = rect_average_color(self.canvas, rect)
					color_distance = null_palette.distance(compare_color[0:3], brush_color[0:3])
					if color_distance < self.color_distance_threshold:
						continue
					brush_index = random.choice(resized_brushes)
					new_brush = create_brush_from_color(brush_index, brush_color)
					new_height_brush_under = None
					new_height_brush_over = None
					if self.generate_heightmap:
						new_height_brush_under = create_brush_from_color(brush_index, (0,0,0,int(percent[1]*128)))
						new_height_brush_over = create_brush_from_color(brush_index, (255,255,255,int(percent[1]*255)))
					
					angle = random.randint(0,360)
					draw_brush(self.canvas, pixel+rnd, new_brush.rotate(angle), brush_size)
					if self.generate_heightmap:
						draw_brush(current_height_canvas, pixel+rnd, new_height_brush_under.rotate(angle), brush_size)
						draw_brush(current_height_canvas, pixel+rnd, new_height_brush_over.rotate(angle), brush_size)
					del new_brush
					if self.generate_heightmap:
						del new_height_brush_under
						del new_height_brush_over
			if self.save_gif:
				if color_index >= last_saved_index + len(all_colors) * 0.33:
					if self.save_gif:
						last_saved_index = color_index
						new_image = self.canvas.copy()
						new_image.thumbnail([self.gif_size, self.gif_size], PIL.Image.LANCZOS)
						# self.gif_size
						self.gif_frames.append(new_image)
			if self.save_incremental:
				if color_index >= last_saved_index + len(all_colors) * 0.33:
					last_saved_index = color_index
					self.canvas.save('output/out_'+str(self.total_saved_index)+'.png')
					self.total_saved_index += 1
			if time.time() > last_time + 1:
				progress = io.BytesIO()
				self.canvas.save(progress, format='JPEG')
				self.progress_image = progress
				last_time = time.time()
		if self.generate_heightmap:
			height_canvas_mask = PIL.Image.new('RGBA', self.reference_image.size, (0,0,0,int(255/(len(percents)))))
			self.height_canvas.paste(height_canvas_mask, mask=height_canvas_mask)
			self.height_canvas.paste(current_height_canvas, mask=current_height_canvas)
			del height_canvas_mask
			del current_height_canvas

app = Flask(__name__)
painter = None
painter_process = None

def stop_painter():
	global painter
	if painter_process:
		painter.stop()

def start_painter():
	global painter_process
	global painter
	stop_painter()
	while painter.finished == False:
		time.sleep(0.25)
	if painter.finished == True:
		painter_process = threading.Thread(target=painter.run)
		painter_process.start()

@app.route('/')
def landing_page():
	name = request.args.get("name", "World")
	config_names = painter.configuration_names
	image_names = [os.path.basename(x) for x in glob.glob('input/*.*')]
	palette_names = painter.default_palette_names + [os.path.splitext(os.path.basename(x))[0] for x in glob.glob('palette/*.*')]
	return render_template('main.html', config_names=config_names, palette_names=palette_names, image_names=image_names)

@app.route('/status')
def route_status():
	global painter
	return painter.message

@app.route('/stop')
def route_stop():
	stop_painter()
	return ''

@app.route('/start')
def route_start():
	painter.load_image(request.values.get('image'))
	painter.palette = painter.setup_palette(request.values.get('palette'))
	painter.configuration = painter.get_configuration(request.values.get('configuration'))
	painter.color_distance_threshold = int(request.values.get('color_distance_threshold'))
	painter.palette_strict = request.values.get('palette_usage') == 'Exact'
	painter.autocontrast_cutoff = None
	if len(request.values.get('autocontrast','')) > 0:
		try:
			cutoff = int(request.values.get('autocontrast','0'))
			painter.autocontrast_cutoff = cutoff
		except Exception as e:
			print(e)
	start_painter()
	return ''

@app.route('/progress.jpg')
def route_progress():
	global painter
	file = painter.progress_image.getvalue()
	return send_file(io.BytesIO(file), download_name='progress.jpg', mimetype='image/jpeg')

@app.route('/progress.gif')
def route_progress_gif():
	global painter
	file = painter.progress_image.getvalue()
	return send_file('output/out.gif', download_name='progress.gif', mimetype='image/gif')

if __name__ == '__main__':
	painter = AutoPainter()
	app.run(port='8001')