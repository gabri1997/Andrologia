import os
import numpy as np
import argparse
from PIL import Image
import cv2

def find_components(input_path, dest_path, minimum_area_filter, save_flag, show_flag):

    print(f'Ho impostato il filtro di area minima a {minimum_area_filter}')
    extensions = {'.bmp', '.png', '.jpg'}
    print(f'Cerco le immagini con le seguenti estensioni: {extensions}')

    os.makedirs(dest_path, exist_ok=True)

    for folder in os.listdir(input_path):

        # Contatore globale per i risultati
        result_counter = 0  

        for file in os.listdir(os.path.join(input_path, folder)):

            if os.path.splitext(file)[1] in extensions:

                path = os.path.join(input_path, folder)
                image = np.array(Image.open(os.path.join(path, file)))
                print(f"Le dimensioni dell'immagine sono: {image.shape}")

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
                num_labels, labels = cv2.connectedComponents(binary)
                areas = [np.sum(labels == i) for i in range(num_labels)]

                # Filtro le aree superiori al filtro minimo (escludendo il background che ha indice sempre 0)
                areas_filtered = [(i, area) for i, area in enumerate(areas) if area > minimum_area_filter and i != 0]

                if len(areas_filtered) >= 1:
                    areas_filtered.sort(key=lambda x: x[1], reverse=True)
                    sorted_pics = [i for i, _ in areas_filtered[:2]]  

                    masks = []
                    for i in sorted_pics:
                        mask = (labels == i).astype(np.uint8) * 255
                        masks.append(mask)
                    
                    for mask_index, i in enumerate(sorted_pics):
                        mask = (labels == i).astype(np.uint8) * 255
                        x, y, w, h = cv2.boundingRect(mask)
                        print(f'Bounding box: x={x}, y={y}, w={w}, h={h}')

                        # Croppo l'immagine originale usando il bounding box
                        cropped_image = image[y:y+h, x:x+w]

                        if show_flag == 'True':
                            cv2.imshow("Result", cropped_image)
                            cv2.waitKey(500)
                            cv2.destroyAllWindows()

                        if save_flag:
                            save_folder_path = os.path.normpath(os.path.join(dest_path, folder))
                            os.makedirs(save_folder_path, exist_ok=True)

                            base_name = os.path.splitext(file)[0]
                            save_file_name = f"{base_name}_mask{mask_index}_result_{result_counter}.bmp"
                            save_path = os.path.join(save_folder_path, save_file_name)
                            cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(save_path, cropped_image_bgr)
                            print(f"Salvata immagine ritagliata in: {save_path}")

                            result_counter += 1  

                            # Questo è fisso per ogni immagine
                            crop_box_txt = (0, 110, 527, 158)
                            txt = image[crop_box_txt[1]:crop_box_txt[3], crop_box_txt[0]:crop_box_txt[2]]
                            txt_save_name = f"{base_name}_text_part.bmp"
                            txt_save_path = os.path.join(save_folder_path, txt_save_name)
                            cv2.imwrite(txt_save_path, txt)
                            print(f"Salvata parte di testo in: {txt_save_path}")
                else:
                    print("Non ci sono abbastanza componenti con aree superiori al filtro minimo.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--input_path', type=str, help="Percorso input")
    parser.add_argument('--output_path', type=str, help="Percorso output")
    parser.add_argument('--area_filter', type=int, help='Min area')
    parser.add_argument('--save_flag', type=str, help='Save the image')
    parser.add_argument('--show_flag', type=str, help='Show the image')

    # Parsing dei parametri nel file config.conf
    args = parser.parse_args(['@config.conf'])
    print(args)

    find_components(input_path=args.input_path, dest_path=args.output_path, minimum_area_filter=args.area_filter, save_flag=args.save_flag, show_flag=args.show_flag)
