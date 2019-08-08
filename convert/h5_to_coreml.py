import coremltools
#corelml_model = coremltools.converters.keras.convert('keras-yolo3/model_data/pos_yolov3_tiny.h5', \
#    input_names='image', output_names='grid', input_name_shape_dict = {'input:0' : [1, 416, 416, 3]}, \
#    image_input_names = ['input:0'], image_scale=1/255.)


corelml_model = coremltools.converters.keras.convert('keras-yolo3/model_data/pos_yolov3_tiny.h5', \
                     input_names='image', output_names='grid', \
                     is_bgr = True, \
                     input_name_shape_dict = {'image' : [None, 416, 416, 3]}, \
                     image_scale = 1 / 255.0)

corelml_model.input_description['image'] = 'Input image'
corelml_model.output_description['grid'] = 'The 13x13 grid'

corelml_model.save('core_ml_yolov3-tiny.mlmodel')

