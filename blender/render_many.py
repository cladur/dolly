import bpy
import os

C = bpy.context
scn = C.scene

list = bpy.data.collections['Collection'].objects
#list = bpy.context.selected_objects // uncomment this line if you want to target your selection instead

output_path = scn.render.filepath
# some render settings
bpy.data.scenes[0].render.resolution_x = 720
bpy.data.scenes[0].render.resolution_y = 720
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

# hide everything
for obj in list:
    obj.hide_render = True

# unhide and render one at a time
for obj in list:
    obj.hide_render = False
    bpy.data.scenes[0].render.filepath = os.path.join(output_path, obj.name + '.png')
    bpy.ops.render.render(write_still=True)
    obj.hide_render = True
