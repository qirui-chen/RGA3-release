import math
import numpy as np
import random

from pycocotools import mask as cocomask

from PIL import Image, ImageDraw
from skimage import measure
import matplotlib.pyplot as plt
import os

from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
from scipy.stats import multivariate_normal
from shapely.validation import explain_validity
from functools import reduce

color_pool = {
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'fuchsia': (255, 0, 255),
    'aqua': (0, 255, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'gold': (255, 215, 0),
}


words_shape ={
    "rectangle": ["within", "rectangle"], 
    "ellipse": ["within", "ellipse"],
    "triangle": ["with", "triangle"],
    "point": ["at", "point"], 
    "scribble" : ["with", "scribble"], 
    "mask contour": ["with", "mask contour"],
    "mask": ["with", "mask"],
    "arrow": ["pointed to by", "arrow"],
 }





def draw_arrow(draw, bbox_coord, outline_color, line_width, max_arrow_length=100, max_image_size=336, image_size_anchor=336):
    left, top, right, bottom = bbox_coord
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    
    # Arrow length related to the bounding box size
    bounding_box_side_length = min(right - left, bottom - top)
    arrow_length = random.uniform(0.8 * bounding_box_side_length, max_arrow_length)
    
    # Randomize the arrow angle
    angle = random.uniform(0, 2 * math.pi)
    
    # Optional: Slight randomness to the center position, but keep it within the bounding box
    center_x += random.uniform(-0.1, 0.1) * (right - left)
    center_y += random.uniform(-0.1, 0.1) * (bottom - top)
    
    # Arrowhead size related to arrow length (making the triangle smaller)
    arrow_head_size = max(random.uniform(0.1, 0.3) * arrow_length, int(4 * max_image_size / image_size_anchor))  # Smaller arrowhead
    
    # Recalculate the arrow start position to make sure the arrowhead points to the center of the mask
    arrow_start_x = center_x + (arrow_length) * math.cos(angle)  # Start further outside the bounding box
    arrow_start_y = center_y + (arrow_length) * math.sin(angle)
    
    # Recalculate the arrow end to ensure it connects properly with the arrowhead
    arrow_end_x = center_x
    arrow_end_y = center_y
    
    n_points = 20 
    
    control_point_1_x = arrow_start_x + 0.5 * random.uniform(-10, 10)
    control_point_1_y = arrow_start_y + 0.5 * random.uniform(-10, 10)
    
    control_point_2_x = arrow_end_x + 0.5 * random.uniform(-10, 10)
    control_point_2_y = arrow_end_y + 0.5 * random.uniform(-10, 10)
    
    bezier_path = []
    for t in np.linspace(0, 1, n_points):
        x = (1 - t)**3 * arrow_start_x + 3 * (1 - t)**2 * t * control_point_1_x + 3 * (1 - t) * t**2 * control_point_2_x + t**3 * arrow_end_x
        y = (1 - t)**3 * arrow_start_y + 3 * (1 - t)**2 * t * control_point_1_y + 3 * (1 - t) * t**2 * control_point_2_y + t**3 * arrow_end_y
        bezier_path.append((x, y))
    
    for i in range(1, len(bezier_path)):
        draw.line([bezier_path[i-1], bezier_path[i]], fill=outline_color, width=line_width)
    
    draw.polygon([
        (arrow_end_x + arrow_head_size * math.cos(angle + math.pi / 3),
         arrow_end_y + arrow_head_size * math.sin(angle + math.pi / 3)),
        (arrow_end_x, arrow_end_y),
        (arrow_end_x + arrow_head_size * math.cos(angle - math.pi / 3),
         arrow_end_y + arrow_head_size * math.sin(angle - math.pi / 3))
    ], fill=outline_color)





def draw_rectangle(draw, bbox_coord, outline_color, width):
    left, top, right, bottom = bbox_coord
    draw.rectangle([(left, top), (right, bottom)], outline=outline_color, width=width)


def draw_ellipse(draw, bbox_coord, mask_polygon, outline_color, width, size_ratio=1, aspect_ratio = 1.0):
    if mask_polygon!= None:
        minx, miny, maxx, maxy = mask_polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox_coord
    
    # Calculate the center of the bounding box
    center_x = (maxx + minx) / 2
    center_y = (maxy + miny) / 2
    
    # Calculate the dimensions of the new bounding box
    new_width = (maxx - minx) * size_ratio * aspect_ratio
    new_height = (maxy - miny) * size_ratio / aspect_ratio
    
    # Calculate the new minx, miny, maxx, maxy based on the new dimensions
    minx = center_x - new_width / 2
    miny = center_y - new_height / 2
    maxx = center_x + new_width / 2
    maxy = center_y + new_height / 2
    
    # Draw the ellipse
    bbox = [minx, miny, maxx, maxy]
    draw.ellipse(bbox, outline=outline_color, width=width)
    
    


def is_max_angle_less_than_150(points):
    for i in range(3):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % 3])
        p3 = np.array(points[(i + 2) % 3])
        
        a = np.linalg.norm(p3 - p2)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Calculate angle at p2 using cosine rule
        angle_at_p2 = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
        
        if angle_at_p2 > 150:
            return False
    return True



def get_random_point_within_bbox(bbox):
    left, top, right, bottom = bbox
    x = np.random.uniform(left, right)
    y = np.random.uniform(top, bottom)
    return x, y


def get_random_point_within_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    trial_num = 0
    while True:
        if  trial_num<50:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            if polygon.contains(point):
                return x, y
            trial_num += 1
        else:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            return x, y
        
        
      

def draw_rounded_triangle(draw, bbox_coord, mask_polygon, outline_color, width):
    while True:
        points = []
        for _ in range(3):
            if mask_polygon!= None:
                point = get_random_point_within_polygon(mask_polygon)
            else:
                point = get_random_point_within_bbox(bbox_coord)
            points.append(point)
        if is_max_angle_less_than_150(points):
            break
    draw.line([points[0], points[1], points[2], points[0]], fill=outline_color, width=width, joint='curve')



def draw_point(draw, bbox_coord, mask_polygon, outline_color=(255,0,0), radius=3, aspect_ratio=1.0):
    # Calculate the center and covariance matrix for multivariate normal distribution
    if mask_polygon is not None:
        minx, miny, maxx, maxy = mask_polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox_coord
    mean = [(maxx + minx) / 2, (maxy + miny) / 2]
    cov = [[(maxx - minx) / 8, 0], [0, (maxy - miny) / 8]]

    # Initialize counter for fail-safe mechanism
    counter = 0
    max_tries = 10

    while counter < max_tries:
        # Generate a random central point within the mask using a normal distribution
        cx, cy = multivariate_normal.rvs(mean=mean, cov=cov)
        center_point = Point(cx, cy)
        
        # If the generated point is within the polygon, break the loop
        if mask_polygon is not None and mask_polygon.contains(center_point):
            break
        
        counter += 1
    else:
        # If we failed to find a point within the polygon, fall back to a random point
        cx, cy = get_random_point_within_polygon(mask_polygon) if mask_polygon is not None else (cx, cy)
    
    # Draw the point as an ellipse
    x_radius = radius * aspect_ratio
    y_radius = radius / aspect_ratio
    bbox = [cx - x_radius, cy - y_radius, cx + x_radius, cy + y_radius]
    draw.ellipse(bbox, outline=outline_color, fill=outline_color)




def draw_scribble(draw, bbox_coord, mask_polygon, outline_color=(255, 0, 0), width=3,  max_image_size=336, image_size_anchor = 336):
            
    prev_point = None  # Initialize prev_point outside the loop
    if mask_polygon!= None:
        p0 = get_random_point_within_polygon(mask_polygon)
        p1 = get_random_point_within_polygon(mask_polygon)
        p2 = get_random_point_within_polygon(mask_polygon)
        p3 = get_random_point_within_polygon(mask_polygon)
    else:
        p0 = get_random_point_within_bbox(bbox_coord)
        p1 = get_random_point_within_bbox(bbox_coord)
        p2 = get_random_point_within_bbox(bbox_coord)
        p3 = get_random_point_within_bbox(bbox_coord)
    
    for t in np.linspace(0, 1, int(1000* max_image_size/image_size_anchor)):
        x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
        
        current_point = (x, y)
        if prev_point:
            draw.line([prev_point, current_point], fill=outline_color, width=width)
            
        prev_point = current_point  # Update prev_point to the current ending point



def draw_mask_contour(draw, bbox_coord, segmentation_coords, color="red", width=1, ):
    if segmentation_coords == None:
          segmentation_coords = [[bbox_coord[0], bbox_coord[1], bbox_coord[0], bbox_coord[3], 
                                bbox_coord[2], bbox_coord[3], bbox_coord[2], bbox_coord[1]]]
    for segment in segmentation_coords:
        coords = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                shifted_coords = [(x + dx, y + dy) for x, y in coords]
                draw.polygon(shifted_coords, outline=color)


def draw_mask(draw, bbox_coord,  segmentation_coords, color="red", width=1,   ):
    if segmentation_coords == None:
          segmentation_coords = [[bbox_coord[0], bbox_coord[1], bbox_coord[0], bbox_coord[3], 
                                bbox_coord[2], bbox_coord[3], bbox_coord[2], bbox_coord[1]]]
    for segment in segmentation_coords:
        coords = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
        draw.polygon(coords, outline= None, fill=color, width=width)


def safe_union(p1, p2):
    try:
        return unary_union([p1, p2])
    except Exception as e:
        # print(f"Error in union: {e}")
        return p1  

def image_blending(image, shape='rectangle', bbox_coord=None, segmentation=None, image_size_anchor=336, rgb_value=None, visual_prompt_style='', alpha=None, width=None, return_vip_img=False):
    img_width, img_height = image.size
    max_image_size = max(img_width, img_height)
    visual_prompt_img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    visual_prompt_img_canvas = ImageDraw.Draw(visual_prompt_img)
    
    if rgb_value is None:
        color_pool = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}  
        color, rgb_value = random.choice(list(color_pool.items()))
    
    if alpha is None:
        alpha = random.randint(188, 224) if shape != 'mask' else random.randint(72, 128)
    
    color_alpha = rgb_value + (alpha,)
    all_polygons_union = None
    
    if segmentation is not None:
        try:
            polygons = []
            for segmentation_coord in segmentation:
                mask_polygon = Polygon([(segmentation_coord[i], segmentation_coord[i+1]) for i in range(0, len(segmentation_coord), 2)])
                if not mask_polygon.is_valid:
                    mask_polygon = mask_polygon.buffer(0)
                if mask_polygon.is_valid and mask_polygon.area > 1e-6:
                    polygons.append(mask_polygon)
                # else:
                #     print(f"Invalid polygon removed: {explain_validity(mask_polygon)}")
            
            if polygons:
                polygons = sorted(polygons, key=lambda poly: poly.area, reverse=True)[:50]
                all_polygons_union = reduce(safe_union, polygons)
                mask_polygon = random.choice(polygons)
            else:
                all_polygons_union = None
                mask_polygon = None
        except Exception as e:
            # print(f"Error processing segmentation: {e}")
            mask_polygon = random.choice(polygons)
    else:
        all_polygons_union = mask_polygon = None
    
    # draw shapes
    if shape == 'rectangle':
        line_width =  max( int(3 *max_image_size/image_size_anchor), 1) if visual_prompt_style == 'constant' else max(random.randint( int(2 *max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_rectangle(visual_prompt_img_canvas, bbox_coord, color_alpha, line_width)
    elif shape == 'ellipse':
        line_width =  max(random.randint( int(2 *max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        size_ratio = 1.2 # random.uniform(1, 1.5) # 
        draw_ellipse(visual_prompt_img_canvas, bbox_coord, all_polygons_union, color_alpha, line_width, size_ratio = size_ratio)  
    elif shape == 'arrow':
        line_width = max(random.randint(int(1 * max_image_size / image_size_anchor), int(6 * max_image_size / image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        max_arrow_length= max( int(50 * max_image_size/image_size_anchor), 1)
        draw_arrow(visual_prompt_img_canvas, bbox_coord, color_alpha, line_width , max_image_size=max_image_size, max_arrow_length = max_arrow_length, image_size_anchor = image_size_anchor)
    elif shape == 'triangle':
        line_width =  max(random.randint(int(2 *  max_image_size/image_size_anchor), int(8 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_rounded_triangle(visual_prompt_img_canvas, bbox_coord, all_polygons_union, color_alpha, line_width)
    elif shape == 'point':
        radius =   max( int(8 * max_image_size/image_size_anchor), 1) if visual_prompt_style == 'constant' else  max(random.randint(int(10 * max_image_size/image_size_anchor),  int(15 *max_image_size/image_size_anchor)), 1)
        aspect_ratio =1 if random.random()<0.5 or  visual_prompt_style == 'constant' else random.uniform(0.5, 2.0)
        draw_point(visual_prompt_img_canvas, bbox_coord, mask_polygon, color_alpha, radius, aspect_ratio)
    elif shape == 'scribble':
        line_width =  max(random.randint(int(12 * max_image_size/image_size_anchor), int(15 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_scribble(visual_prompt_img_canvas, bbox_coord, mask_polygon, color_alpha, line_width, max_image_size=max_image_size, image_size_anchor = image_size_anchor)
    elif shape == 'mask':
        line_width = random.randint( int(0 *max_image_size/image_size_anchor), int(2 * max_image_size/image_size_anchor))
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_mask(visual_prompt_img_canvas, bbox_coord, segmentation, color_alpha, line_width)       
    elif shape == 'mask contour':
        line_width =  max(random.randint( int(1 *max_image_size/image_size_anchor), int(1.5 * max_image_size/image_size_anchor)), 1)
        line_width =  max( int(width *max_image_size/image_size_anchor), 1) if width != None else line_width
        draw_mask_contour(visual_prompt_img_canvas, bbox_coord, segmentation, color_alpha, line_width)

    image = image.convert("RGBA")
    image = Image.alpha_composite(image, visual_prompt_img)
    image = image.convert("RGB")

    if return_vip_img:
        return image, visual_prompt_img

    return image, None



def mask_to_segmentation_coords(mask):

    contours = measure.find_contours(mask, level=0.5)
    
    segmentation_coords = []
    for contour in contours:

        coords = np.round(contour).astype(int)


        coords = [(x, y) for y, x in coords]


        coords = list(dict.fromkeys(coords))


        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        if len(coords) < 4:
            continue

        # **确保多边形有效**
        polygon = Polygon(coords)
        if polygon.is_valid and polygon.area > 1e-6:
            flat_coords = [coord for point in coords for coord in point]
            segmentation_coords.append(flat_coords)

    return segmentation_coords


def get_bbox_from_mask(mask):

    rows = np.any(mask, axis=1)  
    cols = np.any(mask, axis=0)  
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    return (left, top, right + 1, bottom + 1) 


def blend_image(image, bbox_coord, segmentation_coords, mask=None):

    shape = random.choice(['rectangle'])  
    alpha = random.randint(168, 255) if shape != 'mask' else random.randint(48, 128)  
    color, rgb_value = random.choice(list(color_pool.items()))  

    if mask is not None:
        segmentation_coords = mask_to_segmentation_coords(mask)
        bbox_coord = get_bbox_from_mask(mask)

    blended_image = image_blending(
        image,
        shape=shape,
        bbox_coord=bbox_coord,
        segmentation=segmentation_coords,
        image_size_anchor=448,
        # alpha=alpha,
        rgb_value=rgb_value,
        # width=width,
    )
    
    return (color, shape), blended_image


def blend_image_from_mask(frame, mask, color, shape):

    if np.sum(mask) == 0:
        return frame

    else:
        segmentation_coords = mask_to_segmentation_coords(mask)
        bbox_coord = get_bbox_from_mask(mask)
        rgb_value = color_pool[color]

        blended_frame, vip_img = image_blending(
            frame,
            shape=shape,
            bbox_coord=bbox_coord,
            segmentation=segmentation_coords,
            rgb_value=rgb_value,
            image_size_anchor=448,
            visual_prompt_style='constant',
            return_vip_img=False,
        )

        return blended_frame


def video_blending_keyframes(frames, masks, is_key_frame, color, shape, return_vip_img=False):

    blended_frames = []
    vip_img = None
    rgb_value = color_pool[color]
    alpha = random.randint(168, 255) if shape != 'mask' else random.randint(72, 128)

    for frame, mask, flag in zip(frames, masks, is_key_frame):

        if np.sum(mask) == 0 or not flag:
            blended_frames.append(frame)
            continue

        segmentation_coords = mask_to_segmentation_coords(mask)
        bbox_coord = get_bbox_from_mask(mask)

        blended_frame, vip_img = image_blending(
            frame,
            shape=shape,
            bbox_coord=bbox_coord,
            segmentation=segmentation_coords,
            # alpha=alpha,
            rgb_value=rgb_value,
            # width=width,
            # visual_prompt_style='constant',
            image_size_anchor=448,
            return_vip_img=return_vip_img,
        )
        blended_frames.append(blended_frame)

    if return_vip_img:
        return blended_frames, vip_img
    
    return blended_frames