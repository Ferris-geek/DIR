import os

def get_files_from_dir(dir):
    if not os.path.exists(dir):
        return ''

    file_paths = []

    for root, directories, files in os.walk(dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


if __name__ == '__main__':
    root = '/public/home/maofangyuan/dataset/Freiburg/test'
    file_paths = get_files_from_dir('/public/home/maofangyuan/dataset/Freiburg/test')
    with open('Freiburg_rgb_test.txt','w+') as f:
        for i in file_paths:
            i = i.replace('/public/home/maofangyuan/dataset/Freiburg/','')
            if 'ImagesRGB' in i:
                f.write(i+'\n')
            else:
                pass
    

