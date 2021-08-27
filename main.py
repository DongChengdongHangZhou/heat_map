import cv2
import numpy as np
import tifffile as tiff


def bbox_generator(img):
    '''
    return the bbox_list which contains the coordinates of 300 bounding boxes
    '''
    lst = []
    with open("heat_map_test.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            x1 = float(line[1:9])
            if line[1]==' ' and line[2] != ' ':
                x1 = float(line[2:9])
            if line[1]==' ' and line[2] == ' ':
                x1 = float(line[3:9])
            x2 = float(line[11:19])
            if line[11]==' ' and line[12] != ' ':
                x2 = float(line[12:19])
            if line[11]==' ' and line[12] == ' ':
                x2 = float(line[13:19])
            x3 = float(line[21:29])
            if line[21]==' ' and line[22] != ' ':
                x3 = float(line[22:29])
            if line[21]==' ' and line[22] == ' ':
                x3 = float(line[23:29])
            x4 = float(line[31:39])
            if line[31]==' ' and line[32] != ' ':
                x4 = float(line[32:39])
            if line[31]==' ' and line[32] == ' ':
                x4 = float(line[33:39])
            coordinate = []
            coordinate.append(x1)
            coordinate.append(x2)
            coordinate.append(x3)
            coordinate.append(x4)
            lst.append(coordinate)
    return lst

def pixel_add_in_bbox(count_map,x1,y1,x2,y2):
    '''
    the (x1,y1) means the coordinate of the top left corner
    while the (x2,y2) means the coordinate of the lower right corner

    '''
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    count_map[y1:y2,x1:x2] = count_map[y1:y2,x1:x2] + 1

def draw_heatmap(origin_img,count_map,save_path):
    ratio_origin_img = 0.3
    ratio_heat_img = 1 - ratio_origin_img
    brightness = 0
    count_map = count_map.astype(np.uint8)
    heat_img = cv2.applyColorMap(count_map, cv2.COLORMAP_JET)
    mix_img = cv2.addWeighted(origin_img, ratio_origin_img, heat_img, ratio_heat_img, brightness)
    cv2.imwrite(save_path,mix_img)

def bbox_visualizer(img,save_path):
    bbox_list = bbox_generator(img)
    count_map = np.zeros((img.shape[0],img.shape[1]),dtype=int)
    for i in range(len(bbox_list)):
        pixel_add_in_bbox(count_map,bbox_list[i][0],bbox_list[i][1],bbox_list[i][2],bbox_list[i][3])
    count_map = 255*count_map/count_map.max()
    count_map = cv2.GaussianBlur(count_map,(59,59),0)
    draw_heatmap(img,count_map,save_path)

def squeezed_feature_map_visualizer(img,feature_map_pre,feature_map_after,save_path_pre,save_path_after,save_path_diff):
    img = img.astype(np.uint8)
    def process(origin_image,tiff_image,save_path):
        im = tiff_image.copy()
        origin_copy = origin_image.copy()
        im_tiff = cv2.resize(im,(origin_image.shape[1],origin_image.shape[0]))
        im = (im_tiff - im_tiff.min())/(im_tiff.max()-im_tiff.min())*255
        im = im.astype(np.uint8)
        heat_img = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        # mix_img = cv2.addWeighted(origin_image, 0.3, heat_img, 0.7, 0)
        origin_copy[:,:,1] = origin_copy[:,:,0]
        origin_copy[:,:,2] = origin_copy[:,:,0]
        mix_img = 0.8*heat_img + origin_copy
        mix_img = mix_img*255/mix_img.max()
        cv2.imwrite(save_path,mix_img)
    
    process(img,feature_map_pre,save_path_pre)
    # process(img,feature_map_after,save_path_after)
    # process(img,feature_map_after-feature_map_pre,save_path_diff)




if __name__ == '__main__':
    # read_path = './source/1.jpg'
    # save_path = './target/1.jpg'
    # img = cv2.imread(read_path)
    # bbox_visualizer(img,save_path)
    image = cv2.imread('./goat/img_show.png')
    feature_map_pre = tiff.imread('./goat/det.tiff')
    feature_map_after = tiff.imread('./goat/sq.tiff')
    save_path_pre, save_path_after, save_path_diff = './goat/pre.jpg', './goat/after.jpg', './goat/diff.jpg'
    squeezed_feature_map_visualizer(image,feature_map_pre,feature_map_after,save_path_pre,save_path_after,save_path_diff)
