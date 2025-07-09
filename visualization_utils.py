import cv2
import numpy as np
import math

# Color mapping for each piece (BGR format for OpenCV)
PIECE_COLORS = {
    'pink_triangle': (203, 192, 255),
    'red_triangle': (0, 0, 255),
    'orange_triangle': (0, 165, 255),
    'blue_triangle': (255, 144, 30),
    'green_triangle': (0, 255, 0),
    'yellow_square': (0, 255, 255),
    'purple_parallelogram': (128, 0, 128)
}

# UI constants
BOX_THICKNESS = 1
TRANSPARENCY_ALPHA = 0.6

def get_piece_color(class_name):
    """Get the color for a piece based on its class name"""
    return PIECE_COLORS.get(class_name, (255, 255, 255))

def order_vertices_for_polygon(vertices):
    """Sorts 4 vertices to form a convex polygon"""
    center_x = sum(v[0] for v in vertices) / 4
    center_y = sum(v[1] for v in vertices) / 4
    sorted_vertices = sorted(
        vertices,
        key=lambda v: math.atan2(v[1] - center_y, v[0] - center_x)
    )
    return sorted_vertices

def draw_piece_on_frame(frame, piece_data, show_bbox=True, show_vertices=True):
    """
    Draw a piece on a frame with bounding box and vertices
    
    Args:
        frame: OpenCV frame to draw on
        piece_data: Dictionary with 'class_name', 'vertices', and 'bbox'
        show_bbox: Whether to show bounding box
        show_vertices: Whether to show vertex points
    """
    class_name = piece_data['class_name']
    vertices = piece_data.get('vertices', [])
    bbox = piece_data.get('bbox', [])
    
    color = get_piece_color(class_name)
    
    # Draw bounding box
    if show_bbox and bbox and len(bbox) == 4:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, BOX_THICKNESS)
    
    # Draw vertices
    if show_vertices and vertices:
        drawable_vertices = vertices[:]
        
        # Order vertices for squares and parallelograms
        if class_name in ['yellow_square', 'purple_parallelogram'] and len(drawable_vertices) == 4:
            drawable_vertices = order_vertices_for_polygon(drawable_vertices)
        
        for vx, vy in drawable_vertices:
            cv2.circle(frame, (vx, vy), 3, color, -1)

def draw_piece_filled(frame, piece_data, alpha=TRANSPARENCY_ALPHA):
    """
    Draw a filled piece on a frame with transparency
    
    Args:
        frame: OpenCV frame to draw on
        piece_data: Dictionary with 'class_name', 'vertices', and 'bbox'
        alpha: Transparency value (0.0 to 1.0)
    """
    class_name = piece_data['class_name']
    vertices = piece_data.get('vertices', [])
    
    if not vertices or len(vertices) < 3:
        return
    
    color = get_piece_color(class_name)
    drawable_vertices = vertices[:]
    
    # Order vertices for squares and parallelograms
    if class_name in ['yellow_square', 'purple_parallelogram'] and len(drawable_vertices) == 4:
        drawable_vertices = order_vertices_for_polygon(drawable_vertices)
    
    # Create overlay for transparency
    overlay = frame.copy()
    pts = np.array(drawable_vertices, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw vertex points
    for vx, vy in drawable_vertices:
        cv2.circle(frame, (vx, vy), 1, color, -1)

def create_reconstruction_view(frame_shape, pieces_data):
    """
    Create a reconstruction view showing only the filled pieces
    
    Args:
        frame_shape: Shape of the frame (height, width, channels)
        pieces_data: List of piece dictionaries
    
    Returns:
        OpenCV frame with filled pieces
    """
    reconstruction_view = np.zeros(frame_shape, dtype=np.uint8)
    
    for piece_data in pieces_data:
        draw_piece_filled(reconstruction_view, piece_data)
    
    return reconstruction_view