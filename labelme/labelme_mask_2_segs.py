def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        return shapely.geometry.shape(shape)
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def make_labelme_segments(filename, class_list, seg_list, img):
    img_shape = img.shape
    with open('labelme/std_labelme.json', 'r') as f:
        lines = f.read()
        main_subs = ''

        for id, seg in enumerate(seg_list):
            class_id = class_list[id]
            str_seg = ''
            #print('seg', seg)
            #print('test', seg.exterior.coords[:-1])
            for i, point in enumerate(seg.exterior.coords[:-1]):
                if i > 0: str_seg += ',\n'
                str_seg += '            [ {},{} ]'.format(point[0], point[1])

            with open('labelme/std_labelme_segs.json', 'r') as f2:
                subs = f2.read()

            print(str_seg, class_id)
            subs = subs.replace('%POINTS%', str_seg)
            subs = subs.replace('%LABEL%', str(class_id))

            if id>0: main_subs += ',\n'
            main_subs += '    '+subs

        lines = lines.replace('%SEGMENTS%', main_subs)
        lines = lines.replace('%IMAGEPATH%', img_name)
        lines = lines.replace('%IMAGEHEIGHT%', str(img_shape[0]))
        lines = lines.replace('%IMAGEWIDTH%', str(img_shape[1]))

    fname, fext = os.path.splitext(img_name)
    file_json_name = fname + '.json'

    cv2.imwrite(os.path.join(output_images, img_name), img)
    with open(os.path.join(output_labelme, file_json_name), 'wb+') as f:
        f.write(lines.encode('ascii'))

points = mask_to_polygons_layer(mask)
make_labelme_segments(img_name, class_list, seg_list, img)
