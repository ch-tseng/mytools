import os

classes = 277
cfg_folder = 'cfg-diabnext_300-277'
top_correct = 10

#------------------------------------------

cfg_obj_names = os.path.join(cfg_folder, 'labels.list')
cfg_obj_data = "obj.data"

if not os.path.exists( os.path.join(cfg_folder, "weights/")):
    os.makedirs(os.path.join(cfg_folder, "weights/"))

with open(os.path.join(cfg_folder, cfg_obj_data), 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train=" + str(os.path.join(cfg_folder, "train.txt"))+ '\n')
    the_file.write("valid=" + str(os.path.join(cfg_folder, "test.txt"))+ '\n')
    the_file.write("names=" + str(cfg_obj_names) + '\n')
    the_file.write("backup=" + str(os.path.join(cfg_folder, "weights/"))+'\n')
    the_file.write("top="+str(top_correct))

print(cfg_obj_names, "created.")
the_file.close()

