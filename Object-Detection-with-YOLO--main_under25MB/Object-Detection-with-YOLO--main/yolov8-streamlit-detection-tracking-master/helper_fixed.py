from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import numpy as np
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib in streamlit
from PIL import Image

def create_detection_report(task_type, model_name, conf, source_type, class_counts_over_time, df=None, media_path=None):
    """
    Create a PDF report of detection results.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Detection Statistics Report", styles['Title'])
    story.append(title)
    
    # Generated date
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = Paragraph(f"Generated on: {date_str}", styles['Normal'])
    story.append(date)
    
    # Add spacing
    story.append(Spacer(1, 20))
    
    # Summary section
    story.append(Paragraph("Summary", styles['Heading1']))
    
    # Create summary table
    summary_data = [
        ["Task Type", task_type],
        ["Model", model_name],
        ["Confidence Threshold", f"{conf:.2f}"],
        ["Source Type", source_type]
    ]
    
    if media_path:
        summary_data.append(["Media File", os.path.basename(media_path)])
    
    summary_table = Table(summary_data, colWidths=[150, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
    ]))
    story.append(summary_table)
    
    # Add spacing
    story.append(Spacer(1, 20))
    
    # Detection Results section
    story.append(Paragraph("Detection Results", styles['Heading1']))
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Power BI-like colors
    power_bi_colors = [
        '#01B8AA', '#374649', '#FD625E', '#F2C80F', 
        '#5F6B6D', '#8AD4EB', '#FE9666', '#A66999',
        '#3599B8', '#DFBFBF', '#4AC5BB', '#5F6B6D'
    ]
    
    # Set the style to be more like Power BI
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 3.5))
    
    if df is not None and not df.empty and len(df) > 1:
        # Use time series data for video/webcam
        for i, column in enumerate(df.columns[1:]):
            color_idx = i % len(power_bi_colors)
            plt.plot(df['Time'], df[column], marker='', 
                     linewidth=3, label=column, 
                     color=power_bi_colors[color_idx])
        plt.xlabel('Time (seconds)', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
    else:
        # For images, create a line connecting the object counts
        objects = list(class_counts_over_time.keys())
        counts = [class_counts_over_time[obj] for obj in objects]
        
        # For single points, add a line graph visualization
        x_range = range(len(objects))
        for i, (x, y) in enumerate(zip(x_range, counts)):
            color_idx = i % len(power_bi_colors)
            plt.plot([x], [y], marker='o', markersize=10, 
                     color=power_bi_colors[color_idx])
            
        # Add connecting lines with gradient alpha
        for i in range(len(objects) - 1):
            color_idx = i % len(power_bi_colors)
            plt.plot([i, i + 1], [counts[i], counts[i + 1]], 
                     color=power_bi_colors[color_idx], alpha=0.3)
        
        plt.xlabel('Object Classes', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.xticks(x_range, objects, rotation=45, ha='right')
    
    # Enhance the visual appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight', dpi=300)
    img_data.seek(0)
    plt.close()
    
    # Add the plot to the PDF
    img = RLImage(img_data)
    img.drawHeight = 300
    img.drawWidth = 500
    story.append(img)
    
    # Add spacing
    story.append(Spacer(1, 20))
    
    # Detailed Results section
    story.append(Paragraph("Detailed Results", styles['Heading1']))
    
    # Create detailed results table
    if df is not None and not df.empty and len(df) > 1:
        # For video/webcam, show the final counts
        final_counts = df.iloc[-1].drop('Time').to_dict()
        data = [[k, v] for k, v in final_counts.items()]
    else:
        # For images, show the counts directly
        data = [[k, v] for k, v in class_counts_over_time.items()]
    
    # Sort by count in descending order
    data.sort(key=lambda x: x[1], reverse=True)
    
    # Add header row
    data.insert(0, ['Object Class', 'Count'])
    
    # Create table
    table = Table(data, colWidths=[300, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
    ]))
    story.append(table)
    
    # Build the PDF
    doc.build(story)
    
    # Return the PDF as a BytesIO object
    buffer.seek(0)
    return buffer 