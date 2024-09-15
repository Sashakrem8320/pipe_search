# Импорт необходимых библиотек
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import tqdm
import pandas as pd
import json
import os
import wget





with open("config.json", "r") as f:
    data = json.load(f)


# ФУНКЦИИ ПРОЦЕССА

count_pipe_arr = []

# Функция для проверки наложения по принципу IOU
def box_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(
        0, min(y2, y4) - max(y1, y3)
    )

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def get_bounding_rect(contour):  # функция для отрисовки определенных областей
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def get_intersection(
    rect1, rect2
):  # функция для нахождения координат пересечения областей (функция выше)
    x1, y1, x1_end, y1_end = rect1
    x2, y2, x2_end, y2_end = rect2
    x_int1 = max(x1, x2)
    y_int1 = max(y1, y2)
    x_int2 = min(x1_end, x2_end)
    y_int2 = min(y1_end, y2_end)
    if x_int1 < x_int2 and y_int1 < y_int2:
        return (x_int1, y_int1, x_int2, y_int2)
    else:
        return None


def subtract_contours(contour1, contour2):  # функция для объединения контуров
    mask1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)
    intersection = cv2.bitwise_and(mask1, mask2)
    mask1 = cv2.bitwise_and(mask1, cv2.bitwise_not(intersection))
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if contours else []


def filter_and_remove_overlapping_contours(
    contours, threshold=0.5
):  # функция для нахождения пересекающихся масок
    filtered_contours = []
    for i, contour in enumerate(contours):
        keep = True
        for j, existing_contour in enumerate(filtered_contours):
            if i != j:
                rect1 = get_bounding_rect(contour)
                rect2 = get_bounding_rect(existing_contour)
                intersection = get_intersection(rect1, rect2)
                if intersection:
                    contour = subtract_contours(contour, existing_contour)
                    if len(contour) == 0:
                        keep = False
                        break
        if keep:
            filtered_contours.append(contour)
    return filtered_contours


def remove_duplicates(masks_list):
    """Удаляет дубликаты из списка масок."""
    unique_masks = []
    masks_to_remove = set()
    for i, mask in enumerate(masks_list):
        for j, unique_mask in enumerate(unique_masks):
            if np.array_equal(mask, unique_mask):
                masks_to_remove.add(i)
                break
        else:
            unique_masks.append(mask)
    # Удаляем дубликаты из masks_list
    return [mask for i, mask in enumerate(masks_list) if i not in masks_to_remove]


test_folder = f"{data['path_to_folder_with_images']}/"




file_names = []
for root, dirs, files in os.walk(test_folder):
    for file in files:
        file_names.append(os.path.join(root, file))

masks_path = []
fin_mask = []  # Найденные маски
numb_img = len(file_names)  # Кол-во изображений для подсчета труб
for t in range(numb_img):
    torch.cuda.empty_cache()

    model = YOLO("best.pt")  # путь к модели
    img = cv2.imread(file_names[t])

    blur_size = 3
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(
        mask,
        (blur_size, blur_size),
        (img.shape[1] - blur_size, img.shape[0] - blur_size),
        255,
        -1,
    )
    mask = cv2.GaussianBlur(mask, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
    blurred_image = cv2.GaussianBlur(img, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
    img = np.where(mask[..., np.newaxis] == 255, img, blurred_image)

    results = model(img, imgsz=2048, iou=0.8, conf=0.61, verbose=False, max_det=10000)
    classes = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names

    # Проверка на пустоту
    if results[0].masks is not None:
        masks = results[0].masks.data
        num_masks = masks.shape[0]

        colors = [
            tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)
        ]

        combined_image = img.copy()
        num = 0
        object_sizes = []

        for i in range(
            int(num_masks / 100 * 20)
        ):  # Средний размер трубы на 20 % для выбора параметров модели
            color = colors[i]
            mask = masks[i].cpu()
            mask_resized = cv2.resize(
                np.array(mask),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            for c in range(3):
                combined_image[:, :, c] = np.where(
                    mask_resized > 0.5,
                    combined_image[:, :, c] * 1 + color[c] * 0,
                    combined_image[:, :, c],
                )

            class_index = int(classes[i])
            class_name = class_names[class_index]

            mask_contours, _ = cv2.findContours(
                mask_resized.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contour_areas = [cv2.contourArea(contour) for contour in mask_contours]

            # Проверка на пустоту
            if contour_areas != []:
                max_area = max(contour_areas)

                threshold_area = 0.1 * max_area
                filtered_contours = [
                    contour
                    for contour, area in zip(mask_contours, contour_areas)
                    if area > threshold_area
                ]

                filtered_contours = [
                    contour
                    for contour in filtered_contours
                    if cv2.arcLength(contour, True) / cv2.contourArea(contour) < 10
                ]

                all_contours = []

                for contour in filtered_contours:
                    num += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    object_size = (w, h)
                    object_sizes.append(object_size)
        mast = 0
        if object_size[0] < 40:
            total = num_masks
            progress = 0
            pbar = tqdm.tqdm(total=total, desc="Progress")
            # Если трубы меньше размера 40, то используем 1 решение

            resized_masks = []
            all_contours_list = []
            masks_list = []

            for i in range(num_masks):
                progress += 1  # Все маски труб
                pbar.update(1)
                color = colors[i]
                mask = masks[i].cpu()

                # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
                mask_resized = cv2.resize(
                    np.array(mask),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                resized_masks.append(mask_resized)

                for c in range(3):
                    combined_image[:, :, c] = np.where(
                        mask_resized > 0.5,
                        combined_image[:, :, c] * 1 + color[c] * 0,
                        combined_image[:, :, c],
                    )

                class_index = int(classes[i])
                class_name = class_names[class_index]

                mask_contours, _ = cv2.findContours(
                    mask_resized.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                contour_areas = [cv2.contourArea(contour) for contour in mask_contours]

                if contour_areas != []:
                    max_area = max(contour_areas)
                    threshold_area = 0.1 * max_area
                    filtered_contours = [
                        contour
                        for contour, area in zip(mask_contours, contour_areas)
                        if area > threshold_area
                    ]

                    filtered_contours = [
                        contour
                        for contour in filtered_contours
                        if cv2.arcLength(contour, True) / cv2.contourArea(contour) < 10
                    ]

                    all_contours = []

                    for contour in filtered_contours:
                        num += 1
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        object_size = (w, h)
                        object_sizes.append(object_size)

                        if (
                            0.6 < aspect_ratio < 1.05
                        ):  # Если маска похожа на круг рисуем эллипс, если нет,  то оставляем контур
                            center = (x + w // 2, y + h // 2)
                            axes = (w // 2, h // 2)
                            angle = 0
                            startAngle = 0
                            endAngle = 360
                            ellipse_poly = cv2.ellipse2Poly(
                                center, axes, angle, startAngle, endAngle, delta=1
                            )

                            all_contours.append(ellipse_poly)
                        else:
                            all_contours.append(contour)

                    all_contours_list.append(all_contours)
                    cv2.drawContours(
                        combined_image,
                        all_contours,
                        -1,
                        (0, 255, 0),
                        object_size[0] // 40,
                    )

                    # Создаем маску для каждого контура
                    for contour in all_contours:
                        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
                        masks_list.append(mask)

        else:  # Если размер трубы > 40, то используем 2 решение
            total = 4
            progress = 0
            pbar = tqdm.tqdm(total=total, desc="Progress")

            # Параметры для сравнения
            results_2048 = model(
                img, imgsz=1024, iou=0.4, conf=0.55, verbose=False, max_det=10000
            )
            # results_704 = model(img, imgsz=200, iou=0.9, conf=0.55, verbose=False, max_det=10000)
            results_704 = model(
                img, imgsz=224, iou=0.4, conf=0.55, verbose=False, max_det=10000
            )

            boxes_2048 = results_2048[0].boxes.xyxy.cpu().numpy()
            masks_2048 = (
                results_2048[0].masks.data.cpu().numpy()
            )  # Преобразуем к numpy для работы
            num_masks_2048 = masks_2048.shape[0]

            boxes_704 = results_704[0].boxes.xyxy.cpu().numpy()
            masks_704 = results_704[0].masks.data.cpu().numpy()
            num_masks_704 = masks_704.shape[0]

            # Объединение детекций
            combined_boxes = []
            progress += 1
            pbar.update(1)
            for box_2048 in boxes_2048:
                mask = masks[i].cpu()

                # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
                mask_resized = cv2.resize(
                    np.array(mask),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Наложение маски на изображение с определенной прозрачностью
                for c in range(3):  # Для каждого цветового канала
                    combined_image[:, :, c] = np.where(
                        mask_resized > 0.5,
                        combined_image[:, :, c] * 1 + color[c] * 0,
                        combined_image[:, :, c],
                    )

                # Получение класса для текущей маски
                class_index = int(classes[i])
                class_name = class_names[class_index]

                # Наложение контуров и подписей
                mask_contours_2048, _ = cv2.findContours(
                    mask_resized.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                overlap = False
                for box_704 in boxes_704:
                    iou = box_iou(box_2048, box_704)
                    if iou > 0.5:
                        mast += 1
                        overlap = True
                        break
                if not overlap:
                    combined_boxes.append(box_2048)
                else:
                    combined_boxes.extend(
                        [box for box in boxes_704 if box_iou(box, box_2048) < 0.5]
                    )

            progress += 1
            pbar.update(1)
            # Отрисовываем эллипсы для объединенных боксов
            combined_image = img.copy()
            points_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # Перебираем объединенные боксы и рисуем точки
            masks_list = []

            # Перебираем объединенные боксы и создаем маски для точек
            all_contours = []
            all_contours_list = []

            for box in combined_boxes:
                x1, y1, x2, y2 = box.astype(int)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                center = (x + w // 2, y + h // 2)
                axes = (w // 2, h // 2)
                angle = 0
                startAngle = 0
                endAngle = 360
                ellipse_poly = cv2.ellipse2Poly(
                    center, axes, angle, startAngle, endAngle, delta=1
                )
                all_contours.append(ellipse_poly)
            progress += 1
            pbar.update(1)

            # Фильтруем перекрывающиеся контуры

            filtered_contours = filter_and_remove_overlapping_contours(all_contours)

            # Визуализация отфильтрованных контуров

            # combined_imagee = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(combined_image, filtered_contours, -1, (0, 255, 0), 1)

            # Сохранение масок
            masks_list = []
            for contour in filtered_contours:

                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mask_area = cv2.countNonZero(mask)

                # Добавляем маску в список, если площадь больше 10
                if mask_area > 10:
                    masks_list.append(mask)

                    masks_list.append(mask)

            progress += 1
            pbar.update(1)

        # Сохранение количества труб
        masks_list = remove_duplicates(masks_list)

        output_folder = "masks"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the mask to a single JPG file in the specified folder
        mask_to_save = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask in masks_list:
            mask_to_save = np.logical_or(mask_to_save, mask).astype(np.uint8) * 255
        file_n = file_names[t].split(".")
        filename = f'mask_{((file_n[0]).split("/")[-1])}.jpg'
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, mask_to_save)

        count_pipe_arr.append(len(masks_list))
        path_mask = output_folder + "/" + filename
        masks_path.append(path_mask)

        output_folder = "prediction"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_n = file_names[t].split(".")
        filename = f'prediction_{((file_n[0]).split("/")[-1])}.jpg'
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, combined_image)

        print(f"\n Общее количество труб: {len(masks_list)}")

        cv2.imshow("Combined Detection", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        fin_mask.append(masks_list)

    else:
        masks = None
        print("Трубы не найдены.")

if input("Сохранение масок в файл npz [y/n]") == "y":  # Сохранение масок
    np.savez_compressed("masks.npz", *fin_mask)

    masks_data = np.load("masks.npz")
    mask_keys = masks_data.files
    masks_list = [masks_data[key] for key in mask_keys]

    if masks_list and masks_list[0] is not None:
        # Отображение масок для каждого изображения в отдельном окне
        for i, masks in enumerate(masks_list):
            combined_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
            for mask in masks:
                combined_mask += mask

            # Преобразование в формат uint8 для отображения
            combined_mask = (combined_mask > 0).astype(np.uint8) * 255

            # Получение размера объединенной маски
            height, width = combined_mask.shape

            # Изменение размера окна под размер изображения
            cv2.namedWindow(f"Image {i + 1}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Image {i + 1}", width, height)

            cv2.imshow(f"Image {i + 1}", combined_mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid mask data found in the file.")


# Мы выбрали YOLOv8 по нескольким значиным причинам:
# 1) YOLO обладает высокой точностью , в сравнении с другими моделями
# 2) у YOLO большая скорость , так как прогоняет изображения за один проход

results = []

for idx, fname in enumerate(count_pipe_arr):
    file_path = file_names[idx]
    results.append(
        dict(
            img_path=file_path,
            annot_path=masks_path[idx],
            pipe_class=0,
            pipe_count=count_pipe_arr[idx],
        )
    )
result_df = pd.DataFrame(results)
result_df.to_csv("submission.csv", index=False)
