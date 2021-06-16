import cv2 as cv
import numpy as np
import pickle
import os

color_dict = {
    'soil': (255, 0, 0),
    'network': (0, 255, 0),
    'concrete': (0, 0, 255),
    'plate': (0, 255, 255)
}


def get_np_arr(points):
    result = []
    for i in range(len(points) - 1):
        if i % 2 == 1:
            continue

        result.append([points[i], points[i + 1]])

    pts = np.array(result, np.int32)

    pts = pts.reshape((-1, 1, 2))

    return pts


def draw_polygon(img, result):
    shape = img.shape
    blank_image = np.zeros((shape[0], shape[1], 3), np.uint8)

    # arr[0] = ['green network 100%', 'concrete 100%', 'soil 100%', 'green network 50%']
    # arr[1] = masks
    # result = [labels, masks]
    for i in range(len(result[0])):
        if 'soil' in result[0][i]:
            for p in result[1][i].polygons:
                pts = get_np_arr(p)
                cv.fillPoly(blank_image, [pts], color_dict['soil'])
        elif 'network' in result[0][i]:
            for p in result[1][i].polygons:
                pts = get_np_arr(p)
                cv.fillPoly(blank_image, [pts], color_dict['network'])
        elif 'concrete' in result[0][i]:
            for p in result[1][i].polygons:
                pts = get_np_arr(p)
                cv.fillPoly(blank_image, [pts], color_dict['concrete'])
        elif 'plate' in result[0][i]:
            for p in result[1][i].polygons:
                pts = get_np_arr(p)
                cv.fillPoly(blank_image, [pts], color_dict['plate'])
        else:
            pass

    output = img.copy()
    cv.addWeighted(blank_image, 0.5, output, 1, 0, output)

    return output


def _draw_polygon(points):
    save_path = '/mnt/nfs_share/soil_cover_datasets/soil_cover/test/test_result/'
    source_path = '/mnt/nfs_share/soil_cover_datasets/soil_cover/test/test/'
    for filename, arr in points.items():
        img = cv.imread(os.path.join(source_path, filename))
        shape = img.shape
        blank_image = np.zeros((shape[0], shape[1], 3), np.uint8)
        save_to = os.path.join(save_path, filename)

        # arr[0] = ['green network 100%', 'concrete 100%', 'soil 100%', 'green network 50%']
        # arr[1] = masks
        # arr = [labels, masks]
        for i in range(len(arr[0])):
            if 'soil' in arr[0][i]:
                for p in arr[1][i].polygons:
                    pts = get_np_arr(p)
                    cv.fillPoly(blank_image, [pts], color_dict['soil'])
            elif 'network' in arr[0][i]:
                for p in arr[1][i].polygons:
                    pts = get_np_arr(p)
                    cv.fillPoly(blank_image, [pts], color_dict['network'])
            elif 'concrete' in arr[0][i]:
                for p in arr[1][i].polygons:
                    pts = get_np_arr(p)
                    cv.fillPoly(blank_image, [pts], color_dict['concrete'])
            elif 'plate' in arr[0][i]:
                for p in arr[1][i].polygons:
                    pts = get_np_arr(p)
                    cv.fillPoly(blank_image, [pts], color_dict['plate'])
            else:
                pass

        output = img.copy()
        cv.addWeighted(blank_image, 0.5, output, 1, 0, output)

        # cv.imwrite(save_to, output)


#     cv.imwrite("/mnt/nfs_share/soil_cover/test.jpg", img)


if __name__ == "__main__":
    # test_path = "/mnt/nfs_share/soil_cover_datasets/soil_cover/test/test/"
    #
    # for root, dir, files in os.walk(test_path):
    #     names = files
    #
    # for n in names:
    #     image_path = os.path.join(test_path, n)
    #     img = cv.imread(image_path)

    # img = cv.imread("/mnt/nfs_share/soil_cover/soil_cover/吴中区住建局_尹西二村安置小区一标段东地块制高点球机_299149.jpg")
    #
    # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv.polylines(img, [pts], True, (0, 255, 255))
    # draw_polygon(img, points=test_points)
    with open("label_data", "rb") as f:
        label_data = pickle.load(f)

    draw_polygon(label_data)