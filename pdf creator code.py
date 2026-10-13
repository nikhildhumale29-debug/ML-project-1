# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 08:29:38 2026

@author: dhuma
"""

from PIL import Image

def create_pdf_from_images(image_files, output_pdf):
    # Open all images and convert to RGB
    images = [Image.open(img).convert("RGB") for img in image_files]
    
    # Save as PDF (first image + rest appended)
    images[0].save(output_pdf, save_all=True, append_images=images[1:])
    print(f"PDF created successfully: {output_pdf}")

if __name__ == "__main__":
    image_files = [
        r"D:\New folder (2)\1.png",
        r"D:\New folder (2)\2.png",
        r"D:\New folder (2)\3.png",
        r"D:\New folder (2)\4.png",
        r"D:\New folder (2)\5.png",
        r"D:\New folder (2)\6.png",
        r"D:\New folder (2)\7.png",
        r"D:\New folder (2)\8.png",
        r"D:\New folder (2)\9.png",
        r"D:\New folder (2)\10.png"
    ]
    create_pdf_from_images(image_files, r"D:\New folder (2)\Gcp_Week2_Comput_Slides.pdf")
