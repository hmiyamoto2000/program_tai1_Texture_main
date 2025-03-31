# program_tai1_Texture_main

program_tai-main_for_Field_photo_diagnostic_imaging_texture

(These programs were created with outstanding contributions by Shingo Tamachi and Shion Yamada master's degree in Chiba University)

Firstly, confirm the filename "Setup Instruction.txt" and "requirements.txt". The libraries and modules should be set up.

#Texture analysis 

i) Preparation before the horizonal adjustment

Checking at the points of the lip and tail rays Storing the csv file with the above information within the working folder (the filename "center_line_tai.csv" in the folder name "input_csv" in the zip file)
Store the target photo in the folder name "fish_img".

Alter the background color (RGB,0:255:0) of the taget photo and store the altered photo in the folder name "anotation"

ii) Commands for the horizonal adjustment 

The Python command automatically rotates the photo based on the horizon by the points between the snout and tail rays, and the photos will be stored in the folder name "output" (python3 line_split_ver.py) 

iii) Preparation for the measuring range 

The divided photos with the measuring range (1/3 portion in the body) will be automatically stored in the folder name "anotation_split_img" and "anotation_split_img" 

iv) Commands for the evaluation within the range

The file with texture values will be stored in the folder name "2-texture_analysis" (python3 feature_value_image_whole.py)
